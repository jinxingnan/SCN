"""
Projector Module

Maps LLM hidden states to the KGE (Knowledge Graph Embedding) space.
This is the core trainable component that bridges language understanding
and knowledge graph navigation.

Following SCN methodology:
    v_nav = P_ω(h_last)

Where:
    h_last: Last token hidden state from LLM [batch, llm_hidden_size]
    P_ω: MLP projector with learnable parameters ω
    v_nav: Navigation vector in KGE space [batch, kge_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .config import SCNConfig


class Projector(nn.Module):
    """
    MLP Projector from LLM space to KGE space.

    Architecture:
        Input (llm_hidden_size)
            → Linear → Activation → Dropout
            → Linear → Activation → Dropout
            → ... (configurable depth)
            → Linear → L2 Normalize
        Output (kge_dim)

    The output is always L2-normalized to enable cosine similarity matching
    with candidate vectors.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize projector.

        Args:
            input_dim: Input dimension (LLM hidden size, e.g., 4096)
            output_dim: Output dimension (KGE dimension, e.g., 768)
            hidden_dims: List of hidden layer dimensions, e.g., [2048, 1024]
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final projection layer (no activation, no dropout - will be normalized)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project LLM hidden state to KGE space.

        Args:
            h: LLM hidden state, shape [batch, input_dim] or [input_dim]

        Returns:
            Navigation vector in KGE space, shape [batch, output_dim] or [output_dim]
            The output is L2-normalized.
        """
        # Handle single vector input
        squeeze_output = False
        if h.dim() == 1:
            h = h.unsqueeze(0)
            squeeze_output = True

        # MLP forward
        v = self.mlp(h)

        # L2 normalize to unit sphere
        v = F.normalize(v, p=2, dim=-1)

        if squeeze_output:
            v = v.squeeze(0)

        return v

    @classmethod
    def from_config(cls, config: SCNConfig) -> "Projector":
        """
        Create projector from SCN config.

        Args:
            config: SCN configuration

        Returns:
            Projector instance
        """
        return cls(
            input_dim=config.llm_hidden_size,
            output_dim=config.kge_dim,
            hidden_dims=config.projector_hidden_dims,
            dropout=config.projector_dropout,
            activation=config.projector_activation,
        )

    def get_output_dim(self) -> int:
        """Return output dimension."""
        return self.output_dim

    def get_num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
