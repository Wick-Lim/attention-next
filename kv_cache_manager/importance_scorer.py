"""Importance scoring for KV cache entries based on cosine similarity."""

import torch
import torch.nn.functional as F
from torch import Tensor


class ImportanceScorer:
    """Computes importance score updates for cached token positions.

    When a new token arrives, existing cache entries that are semantically
    similar (cosine similarity > threshold) get an importance boost, while
    dissimilar entries decay.  This drives the SRAM/HBM tier decisions in
    CacheManager.
    """

    def __init__(
        self,
        sim_threshold: float = 0.30,
        boost: float = 0.35,
        decay: float = 0.04,
        promote_thresh: float = 0.68,
    ):
        self.sim_threshold = sim_threshold
        self.boost = boost
        self.decay = decay
        self.promote_thresh = promote_thresh

    def compute_similarity(self, query_vec: Tensor, cache_vecs: Tensor) -> Tensor:
        """Batch cosine similarity between *query_vec* and every row in *cache_vecs*.

        Args:
            query_vec: ``[dim]`` or ``[1, dim]``
            cache_vecs: ``[n, dim]``

        Returns:
            Tensor of shape ``[n]`` with cosine similarities in [-1, 1].
        """
        if query_vec.dim() == 1:
            query_vec = query_vec.unsqueeze(0)
        # F.cosine_similarity broadcasts along dim=-1
        return F.cosine_similarity(query_vec, cache_vecs, dim=-1)

    def compute_updates(
        self,
        query_vec: Tensor,
        cache_vecs: Tensor,
        current_importance: Tensor,
    ) -> Tensor:
        """Return updated importance scores for every cache entry.

        Entries whose cosine similarity with *query_vec* exceeds
        ``sim_threshold`` receive a boost proportional to similarity;
        all others decay by a fixed amount.  The result is clamped to
        [0.08, 1.0].

        Args:
            query_vec: ``[dim]`` – new token's representative vector.
            cache_vecs: ``[n, dim]`` – stacked representative vectors of
                existing cache entries.
            current_importance: ``[n]`` – current importance scores.

        Returns:
            ``[n]`` tensor of new importance scores.
        """
        sims = self.compute_similarity(query_vec, cache_vecs)

        new_importance = torch.where(
            sims > self.sim_threshold,
            current_importance + sims * self.boost,
            current_importance - self.decay,
        )

        return new_importance.clamp(min=0.08, max=1.0)
