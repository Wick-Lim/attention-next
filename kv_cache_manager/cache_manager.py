"""Hierarchical KV cache manager with SRAM / HBM tiers."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from torch import Tensor
from typing import Dict, List, Optional, Tuple

from .importance_scorer import ImportanceScorer


@dataclass
class CacheEntry:
    """One token position's KV data across every model layer."""

    keys: List[Tensor]       # per-layer, each [num_heads, 1, head_dim]
    values: List[Tensor]     # per-layer, each [num_heads, 1, head_dim]
    rep_vec: Tensor          # [dim] representative vector for similarity
    importance: float = 0.75
    refs: int = 0
    token_id: int = -1       # vocabulary id (for debugging / display)
    position: int = -1       # original absolute position in the sequence


class CacheManager:
    """Two-tier (SRAM / HBM) KV-cache manager.

    *SRAM* holds the most important token positions and is used for
    actual attention computation.  *HBM* is a spill-over tier for
    evicted entries that may later be promoted back.

    Because user-space code cannot pin specific GPU memory to L2 cache,
    the two tiers are a *logical* split – both reside in ``device``
    memory.  The tier assignment still faithfully models the eviction /
    promotion policy described in the design document.
    """

    def __init__(
        self,
        sram_capacity: int,
        dim: int,
        device: str = "cuda",
        scorer: Optional[ImportanceScorer] = None,
    ):
        self.sram_capacity = sram_capacity
        self.dim = dim
        self.device = device
        self.scorer = scorer or ImportanceScorer()

        self.sram: List[CacheEntry] = []
        self.hbm: List[CacheEntry] = []

        # Running counters for hit-rate statistics.
        self._sram_hits: int = 0
        self._total_queries: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return len(self.sram) + len(self.hbm)

    def update(
        self,
        new_keys: List[Tensor],
        new_values: List[Tensor],
        rep_vec: Tensor,
        token_id: int = -1,
        position: int = -1,
    ) -> None:
        """Ingest a new token's KV and manage tier placement.

        Steps (following the design-document algorithm):
        1. Compute cosine similarity of *rep_vec* against all existing
           entries and update their importance scores (boost / decay).
        2. Create a new :class:`CacheEntry`.
        3. If SRAM is not full, place the new entry there; otherwise
           evict the lowest-importance SRAM entry to HBM if the new
           entry is more important.
        4. Run a promotion pass: any HBM entry whose importance exceeds
           ``promote_thresh`` (and there is room) is moved to SRAM.
        """
        self._total_queries += 1

        # --- 1. Update existing entries ---
        all_entries = self.sram + self.hbm
        if all_entries:
            cache_vecs = self._stack_rep_vecs(all_entries)
            current_imp = self._importance_tensor(all_entries)
            new_imp = self.scorer.compute_updates(rep_vec, cache_vecs, current_imp)

            for idx, entry in enumerate(all_entries):
                old = entry.importance
                entry.importance = new_imp[idx].item()
                if entry.importance > old:
                    entry.refs += 1

        # --- 2. New entry ---
        new_entry = CacheEntry(
            keys=[k.detach().clone() for k in new_keys],
            values=[v.detach().clone() for v in new_values],
            rep_vec=rep_vec.detach().clone(),
            importance=0.75,
            token_id=token_id,
            position=position,
        )

        # --- 3. Placement ---
        if len(self.sram) < self.sram_capacity:
            self.sram.append(new_entry)
            self._sram_hits += 1
        else:
            victim_idx = min(
                range(len(self.sram)), key=lambda i: self.sram[i].importance
            )
            victim = self.sram[victim_idx]

            if new_entry.importance > victim.importance:
                self.hbm.append(victim)
                self.sram[victim_idx] = new_entry
                self._sram_hits += 1
            else:
                self.hbm.append(new_entry)

        # --- 4. Promotion ---
        self._promote()

    def get_sram_kv(self) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Return SRAM-tier KV in HuggingFace ``past_key_values`` format.

        Each layer yields ``(key, value)`` with shape
        ``[batch=1, num_heads, sram_len, head_dim]``.
        """
        if not self.sram:
            return ()
        return self._entries_to_past_kv(self.sram)

    def get_all_kv(self) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Return *all* (SRAM + HBM) KV as ``past_key_values``."""
        entries = self.sram + self.hbm
        if not entries:
            return ()
        return self._entries_to_past_kv(entries)

    def reset(self) -> None:
        """Clear both tiers and reset statistics."""
        self.sram.clear()
        self.hbm.clear()
        self._sram_hits = 0
        self._total_queries = 0

    def stats(self) -> Dict[str, object]:
        """Return a snapshot of current cache statistics."""
        sram_imp = [e.importance for e in self.sram]
        hbm_imp = [e.importance for e in self.hbm]
        hit_rate = (
            self._sram_hits / self._total_queries
            if self._total_queries
            else 0.0
        )
        return {
            "sram_count": len(self.sram),
            "hbm_count": len(self.hbm),
            "sram_avg_importance": (
                sum(sram_imp) / len(sram_imp) if sram_imp else 0.0
            ),
            "hbm_avg_importance": (
                sum(hbm_imp) / len(hbm_imp) if hbm_imp else 0.0
            ),
            "sram_capacity_used": f"{len(self.sram)}/{self.sram_capacity}",
            "sram_hit_rate": f"{hit_rate:.2%}",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _promote(self) -> None:
        """Promote high-importance HBM entries to SRAM.

        If SRAM has room, entries are moved directly.  When SRAM is full
        a swap is attempted: the highest-importance HBM entry replaces
        the lowest-importance SRAM entry (if strictly better).
        """
        if not self.hbm:
            return

        promoted: List[int] = []
        for idx, entry in enumerate(self.hbm):
            if entry.importance <= self.scorer.promote_thresh:
                continue

            if len(self.sram) < self.sram_capacity:
                # Room available – move directly.
                self.sram.append(entry)
                promoted.append(idx)
            else:
                # SRAM full – try swapping with the weakest SRAM entry.
                min_sram_idx = min(
                    range(len(self.sram)),
                    key=lambda i: self.sram[i].importance,
                )
                if entry.importance > self.sram[min_sram_idx].importance:
                    demoted = self.sram[min_sram_idx]
                    self.sram[min_sram_idx] = entry
                    self.hbm[idx] = demoted
                    promoted.append(idx)   # mark original slot as handled

        # Remove promoted entries (iterate in reverse to keep indices valid).
        # Entries that were swapped already replaced their HBM slot with the
        # demoted entry, so we must NOT pop those.  Only pop entries that
        # were moved (not swapped).
        moved = [i for i in promoted if i < len(self.hbm) and self.hbm[i] not in self.sram]
        # Simpler: rebuild HBM list excluding entries now in SRAM.
        sram_set = set(id(e) for e in self.sram)
        self.hbm = [e for e in self.hbm if id(e) not in sram_set]

    def _stack_rep_vecs(self, entries: List[CacheEntry]) -> Tensor:
        return torch.stack([e.rep_vec for e in entries])

    def _importance_tensor(self, entries: List[CacheEntry]) -> Tensor:
        return torch.tensor(
            [e.importance for e in entries],
            device=self.device,
            dtype=torch.float32,
        )

    def _entries_to_past_kv(
        self, entries: List[CacheEntry]
    ) -> Tuple[Tuple[Tensor, Tensor], ...]:
        """Pack a list of :class:`CacheEntry` into ``past_key_values``."""
        num_layers = len(entries[0].keys)
        result: List[Tuple[Tensor, Tensor]] = []
        for layer_idx in range(num_layers):
            layer_k = torch.cat(
                [e.keys[layer_idx] for e in entries], dim=1  # seq_len dim
            ).unsqueeze(0)  # add batch dim → [1, heads, seq, head_dim]
            layer_v = torch.cat(
                [e.values[layer_idx] for e in entries], dim=1
            ).unsqueeze(0)
            result.append((layer_k, layer_v))
        return tuple(result)
