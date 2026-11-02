"""
Input Projector for SCN

Maps KGE vectors to LLM embedding space as "soft tokens".
This enables direct injection of entity semantics into the LLM
without relying on entity text names.

Architecture:
    KGE vector (d_kg) → MLP → LLM soft token (d_llm)

Key design choices:
    - Dropout(0.3): Prevents model from memorizing KGE vector patterns
    - LayerNorm: Stabilizes the output distribution to match LLM embeddings
    - GELU activation: Matches LLM internal activation function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InputProjector(nn.Module):
    """
    Maps KGE vectors to LLM embedding space.

    This projector transforms entity embeddings from the KGE space
    into soft tokens that can be directly concatenated with text embeddings
    and processed by the LLM.

    Attributes:
        kge_dim: Dimension of KGE embeddings (typically 512 for RotatE)
        llm_dim: Dimension of LLM embeddings (4096 for Llama-2-7B)
        hidden_dim: Intermediate hidden dimension
        dropout: Dropout probability for regularization
    """

    def __init__(
        self,
        kge_dim: int,
        llm_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Initialize Input Projector.

        Args:
            kge_dim: Dimension of KGE embeddings
            llm_dim: Dimension of LLM embeddings
            hidden_dim: Hidden layer dimension (default: average of kge_dim and llm_dim)
            dropout: Dropout probability (default: 0.3, important for preventing overfitting)
        """
        super().__init__()

        self.kge_dim = kge_dim
        self.llm_dim = llm_dim
        self.dropout_rate = dropout

        # Default hidden dimension: average of input and output
        if hidden_dim is None:
            hidden_dim = (kge_dim + llm_dim) // 2

        self.hidden_dim = hidden_dim

        # Two-layer MLP with dropout
        self.linear1 = nn.Linear(kge_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, llm_dim)
        self.layer_norm = nn.LayerNorm(llm_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, kge_vec: torch.Tensor) -> torch.Tensor:
        """
        Project KGE vector to LLM embedding space.

        Args:
            kge_vec: KGE embedding vector
                     Shape: [d_kg] for single entity
                     Shape: [batch, d_kg] for batched entities

        Returns:
            soft_token: LLM-compatible embedding
                        Shape: [d_llm] for single entity
                        Shape: [batch, d_llm] for batched entities
        """
        # First linear layer
        x = self.linear1(kge_vec)

        # Activation
        x = self.activation(x)

        # Dropout (critical for preventing overfitting to KGE patterns)
        x = self.dropout(x)

        # Second linear layer
        x = self.linear2(x)

        # Layer normalization (stabilizes output distribution)
        x = self.layer_norm(x)

        return x

    def project_entity(self, kge_vec: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: project a single entity vector.

        Args:
            kge_vec: Entity KGE embedding, shape [d_kg]

        Returns:
            soft_token: LLM soft token, shape [d_llm]
        """
        return self.forward(kge_vec)

    def project_batch(self, kge_vecs: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: project a batch of entity vectors.

        Args:
            kge_vecs: Batch of KGE embeddings, shape [batch, d_kg]

        Returns:
            soft_tokens: Batch of LLM soft tokens, shape [batch, d_llm]
        """
        return self.forward(kge_vecs)

    @classmethod
    def from_config(cls, config) -> "InputProjector":
        """
        Create InputProjector from SCNConfig.

        Args:
            config: SCNConfig object with kge_dim, llm_hidden_size, input_projector_dropout

        Returns:
            Initialized InputProjector
        """
        return cls(
            kge_dim=config.kge_dim,
            llm_dim=config.llm_hidden_size,
            hidden_dim=getattr(config, "input_projector_hidden_dim", None),
            dropout=getattr(config, "input_projector_dropout", 0.3),
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"InputProjector(\n"
            f"  kge_dim={self.kge_dim},\n"
            f"  llm_dim={self.llm_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )
