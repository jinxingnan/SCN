"""
Fusion Functions for Candidate Vector Construction

Implements different fusion strategies to combine relation and tail entity embeddings
into candidate vectors for navigation matching.

Following the SCN methodology:
- Candidate vector c = Fusion(e_r, e_t)
- RotatE: Hadamard product (element-wise multiplication)
- TransE: Vector addition
- Weighted: Linear combination (λ * r + (1-λ) * t), relation-dominant

SCN recommends "weighted" fusion with λ=0.7:
- Relations (like 'directed_by') have clearer semantic categories
- Entity vectors serve as auxiliary disambiguation
- Over-reliance on entity vectors leads to overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RotatEFusion(nn.Module):
    """
    RotatE-style fusion using Hadamard (element-wise) product.

    In RotatE, the scoring function is based on complex rotations:
        h ∘ r ≈ t  (where ∘ is element-wise product in complex space)

    For navigation, we want to represent "going via relation r to entity t",
    which can be captured by r ⊙ t (Hadamard product).

    The output is L2-normalized to ensure unit norm for cosine similarity.
    """

    def forward(self, e_r: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """
        Compute fused candidate vector using Hadamard product.

        Args:
            e_r: Relation embedding, shape [..., dim]
            e_t: Tail entity embedding, shape [..., dim]

        Returns:
            Fused candidate vector, shape [..., dim], L2-normalized
        """
        # Element-wise multiplication
        fused = e_r * e_t

        # L2 normalize to unit sphere
        return F.normalize(fused, p=2, dim=-1)


class TransEFusion(nn.Module):
    """
    TransE-style fusion using vector addition.

    In TransE, the scoring function is based on translation:
        h + r ≈ t

    For navigation, we represent "going via relation r to entity t"
    as r + t (vector addition).

    The output is L2-normalized to ensure unit norm for cosine similarity.
    """

    def forward(self, e_r: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """
        Compute fused candidate vector using vector addition.

        Args:
            e_r: Relation embedding, shape [..., dim]
            e_t: Tail entity embedding, shape [..., dim]

        Returns:
            Fused candidate vector, shape [..., dim], L2-normalized
        """
        # Vector addition
        fused = e_r + e_t

        # L2 normalize to unit sphere
        return F.normalize(fused, p=2, dim=-1)


class WeightedFusion(nn.Module):
    """
    Weighted linear combination fusion for SCN.

    Formula: c = Normalize(λ * r + (1-λ) * t)

    This fusion strategy emphasizes relation semantics over entity specifics:
    - λ (lambda_weight) typically set to 0.7
    - Relations have clearer semantic categories (e.g., 'directed_by', 'acted_in')
    - LLM can more easily infer relation type from the question
    - Entity vectors provide auxiliary disambiguation

    The output is L2-normalized to ensure unit norm for cosine similarity.
    """

    def __init__(self, lambda_weight: float = 0.7):
        """
        Initialize WeightedFusion.

        Args:
            lambda_weight: Weight for relation embedding (default: 0.7)
                          Higher values give more weight to relation semantics
                          Recommended range: [0.5, 0.9]
        """
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(self, e_r: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """
        Compute fused candidate vector using weighted linear combination.

        Args:
            e_r: Relation embedding, shape [..., dim]
            e_t: Tail entity embedding, shape [..., dim]

        Returns:
            Fused candidate vector, shape [..., dim], L2-normalized
        """
        # Weighted linear combination
        fused = self.lambda_weight * e_r + (1 - self.lambda_weight) * e_t

        # L2 normalize to unit sphere
        return F.normalize(fused, p=2, dim=-1)

    def __repr__(self) -> str:
        return f"WeightedFusion(lambda_weight={self.lambda_weight})"


def get_fusion(fusion_type: str, lambda_weight: Optional[float] = None) -> nn.Module:
    """
    Factory function to get the appropriate fusion module.

    Args:
        fusion_type: "rotate", "transe", or "weighted"
        lambda_weight: Weight for relation in weighted fusion (default: 0.7)
                      Only used when fusion_type is "weighted"

    Returns:
        Fusion module instance

    Raises:
        ValueError: If fusion_type is not recognized
    """
    if fusion_type == "rotate":
        return RotatEFusion()
    elif fusion_type == "transe":
        return TransEFusion()
    elif fusion_type == "weighted":
        weight = lambda_weight if lambda_weight is not None else 0.7
        return WeightedFusion(lambda_weight=weight)
    else:
        valid_types = ["rotate", "transe", "weighted"]
        raise ValueError(
            f"Unknown fusion type: '{fusion_type}'. Must be one of {valid_types}"
        )


def fuse_candidates_batch(
    relation_embeddings: torch.Tensor,
    tail_embeddings: torch.Tensor,
    fusion_type: str = "rotate",
    lambda_weight: float = 0.7,
) -> torch.Tensor:
    """
    Batch fusion of relation and tail embeddings into candidate vectors.

    This is a functional interface for batch processing without module instantiation.

    Args:
        relation_embeddings: Relation embeddings, shape [batch, num_candidates, dim]
        tail_embeddings: Tail entity embeddings, shape [batch, num_candidates, dim]
        fusion_type: "rotate", "transe", or "weighted"
        lambda_weight: Weight for relation in weighted fusion (default: 0.7)

    Returns:
        Fused candidate vectors, shape [batch, num_candidates, dim], L2-normalized

    Raises:
        ValueError: If fusion_type is not recognized
    """
    if fusion_type == "rotate":
        fused = relation_embeddings * tail_embeddings
    elif fusion_type == "transe":
        fused = relation_embeddings + tail_embeddings
    elif fusion_type == "weighted":
        fused = lambda_weight * relation_embeddings + (1 - lambda_weight) * tail_embeddings
    else:
        valid_types = ["rotate", "transe", "weighted"]
        raise ValueError(
            f"Unknown fusion type: '{fusion_type}'. Must be one of {valid_types}"
        )

    return F.normalize(fused, p=2, dim=-1)
