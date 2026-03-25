"""Hook into HuggingFace GPT-2 to use managed KV cache during generation."""

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Optional, Tuple

from .cache_manager import CacheManager


# ------------------------------------------------------------------
# Helper: extract per-position KV and representative vectors
# ------------------------------------------------------------------

def _extract_position_kv(
    past_key_values: Tuple[Tuple[Tensor, Tensor], ...],
    position: int,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Pull out one position's key/value slices across all layers.

    Args:
        past_key_values: HF-style tuple.  Each layer entry is
            ``(key, value)`` with shape ``[batch, heads, seq, head_dim]``.
        position: index along the seq dimension to extract.

    Returns:
        ``(keys, values)`` – lists (one per layer) of tensors with shape
        ``[heads, 1, head_dim]`` (batch dimension removed).
    """
    keys: List[Tensor] = []
    values: List[Tensor] = []
    for layer_kv in past_key_values:
        # layer_kv[0]: [batch, heads, seq, head_dim]
        keys.append(layer_kv[0][0, :, position : position + 1, :])
        values.append(layer_kv[1][0, :, position : position + 1, :])
    return keys, values


def _make_rep_vec(keys_layer0: Tensor) -> Tensor:
    """Create a representative vector from the first layer's key tensor.

    Args:
        keys_layer0: ``[num_heads, 1, head_dim]``

    Returns:
        Flat vector ``[num_heads * head_dim]``.
    """
    return keys_layer0.squeeze(1).reshape(-1)


# ------------------------------------------------------------------
# AttentionHook – thin wrapper that patches ``model.forward``
# ------------------------------------------------------------------

class AttentionHook:
    """Intercept a GPT-2 model's ``forward`` to route KV through a
    :class:`CacheManager`.

    Usage::

        hook = AttentionHook(model, cache_manager)
        hook.register()
        # … run generation / evaluation …
        hook.remove()

    When registered the hook replaces ``past_key_values`` fed into
    ``model.forward`` with the SRAM-tier KV from *cache_manager* and
    injects the correct ``position_ids`` so that learned position
    embeddings stay consistent.
    """

    def __init__(self, model, cache_manager: CacheManager):
        self.model = model
        self.cache = cache_manager
        self._original_forward = None
        self._position: int = 0  # tracks absolute position

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int) -> None:
        self._position = value

    def register(self) -> None:
        """Monkey-patch ``model.forward`` to inject managed KV cache."""
        if self._original_forward is not None:
            return  # already registered

        self._original_forward = self.model.forward
        cache = self.cache
        hook = self  # capture for closure

        def _patched_forward(*args, **kwargs):
            # Only intervene when using cached generation
            # (i.e. single-token input with existing cache).
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if (
                input_ids is not None
                and input_ids.shape[-1] == 1
                and cache.total_tokens > 0
            ):
                kwargs["past_key_values"] = cache.get_sram_kv()
                kwargs["position_ids"] = torch.tensor(
                    [[hook._position]], device=input_ids.device
                )

            outputs = hook._original_forward(*args, **kwargs)

            # After forward: grab the new KV (last position) and feed to cache.
            if (
                input_ids is not None
                and input_ids.shape[-1] == 1
                and outputs.past_key_values
            ):
                past_kv = outputs.past_key_values
                seq_len = past_kv[0][0].shape[2]
                keys, values = _extract_position_kv(past_kv, seq_len - 1)
                rep = _make_rep_vec(keys[0])
                cache.update(
                    keys, values, rep,
                    token_id=input_ids[0, -1].item(),
                    position=hook._position,
                )
                hook._position += 1

            return outputs

        self.model.forward = _patched_forward

    def remove(self) -> None:
        """Restore the original ``model.forward``."""
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None


# ------------------------------------------------------------------
# ManagedGenerator – standalone generation loop (no monkey-patching)
# ------------------------------------------------------------------

class ManagedGenerator:
    """Generate text using a CacheManager-backed KV cache.

    This is the recommended high-level API for the MVP.  Unlike
    :class:`AttentionHook` it does *not* modify the model object; instead
    it drives its own generation loop.
    """

    def __init__(self, model, tokenizer, cache_manager: CacheManager):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache_manager
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """Generate continuation for *prompt* using managed KV cache.

        Returns the generated text (prompt excluded).
        """
        self.cache.reset()

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        # --- Prefill: process full prompt at once ---
        outputs = self.model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values

        for pos in range(prompt_len):
            keys, values = _extract_position_kv(past_kv, pos)
            rep = _make_rep_vec(keys[0])
            self.cache.update(
                keys, values, rep,
                token_id=input_ids[0, pos].item(),
                position=pos,
            )

        # First generated token
        next_token = self._sample(outputs.logits[:, -1, :], temperature, top_k)
        generated_ids = [next_token.item()]
        position = prompt_len

        # --- Decode loop ---
        for _ in range(max_new_tokens - 1):
            managed_past = self.cache.get_sram_kv()
            position_ids = torch.tensor([[position]], device=self.device)

            outputs = self.model(
                next_token.unsqueeze(0),
                past_key_values=managed_past,
                position_ids=position_ids,
                use_cache=True,
            )

            # Extract new KV (last position in returned cache)
            new_past = outputs.past_key_values
            seq_len = new_past[0][0].shape[2]
            keys, values = _extract_position_kv(new_past, seq_len - 1)
            rep = _make_rep_vec(keys[0])
            self.cache.update(
                keys, values, rep,
                token_id=next_token.item(),
                position=position,
            )

            next_token = self._sample(outputs.logits[:, -1, :], temperature, top_k)
            generated_ids.append(next_token.item())
            position += 1

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Perplexity evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_perplexity(
        self,
        text: str,
        max_length: int = 512,
    ) -> dict:
        """Compute perplexity of *text* using managed KV cache.

        Processes token-by-token so the cache management policy affects
        every position.  Returns a dict with ``managed_ppl`` and
        ``baseline_ppl`` (full-attention reference).
        """
        import torch.nn.functional as F

        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, :max_length]
        seq_len = input_ids.shape[1]

        # ---- Baseline: full-attention perplexity ----
        baseline_outputs = self.model(input_ids)
        shift_logits = baseline_outputs.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        baseline_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        baseline_ppl = torch.exp(baseline_loss).item()

        # ---- Managed: incremental with cache management ----
        # Feed token t_i, logits predict t_{i+1}.  Accumulate NLL for
        # t_{i+1} and add t_i's KV to the cache after each step.
        self.cache.reset()
        total_nll = 0.0
        count = 0

        for i in range(seq_len - 1):
            managed_past = self.cache.get_sram_kv() or None
            position_ids = torch.tensor([[i]], device=self.device)

            out = self.model(
                input_ids[:, i : i + 1],
                past_key_values=managed_past,
                position_ids=position_ids,
                use_cache=True,
            )

            # NLL for the *next* token (t_{i+1})
            logits = out.logits[:, -1, :]
            target = input_ids[:, i + 1]
            nll = F.cross_entropy(logits, target)
            total_nll += nll.item()
            count += 1

            # Add t_i's KV to cache
            new_past = out.past_key_values
            new_seq = new_past[0][0].shape[2]
            keys, values = _extract_position_kv(new_past, new_seq - 1)
            rep = _make_rep_vec(keys[0])
            self.cache.update(
                keys, values, rep,
                token_id=input_ids[0, i].item(),
                position=i,
            )

        import math

        managed_ppl = math.exp(total_nll / count) if count else float("inf")

        return {
            "baseline_ppl": baseline_ppl,
            "managed_ppl": managed_ppl,
            "ppl_ratio": managed_ppl / baseline_ppl if baseline_ppl else float("inf"),
            "sequence_length": seq_len,
            "cache_stats": self.cache.stats(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(logits: Tensor, temperature: float, top_k: int) -> Tensor:
        """Sample a single token from *logits* with temperature + top-k."""
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        if top_k > 0:
            top_vals, _ = logits.topk(top_k)
            logits[logits < top_vals[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
