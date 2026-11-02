"""
Candidate Builder

Constructs candidate vectors for navigation matching.
Given a current entity and its neighbors in the subgraph,
builds fused candidate vectors using the Fusion function.

Following SCN methodology:
    c_i = Fusion(e_r, e_t) for each neighbor (h, r, t)

The STOP action is represented by a learnable vector v_stop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Set

from embeddings.kge_loader import KGELoader
from embeddings.fusion import get_fusion


class CandidateBuilder(nn.Module):
    """
    Builds candidate vectors for action selection.

    For a given current entity h and its outgoing edges {(h, r_i, t_i)},
    constructs candidate vectors:
        - c_i = Fusion(e_{r_i}, e_{t_i}) for each neighbor
        - v_stop = learnable STOP vector

    The candidate matrix C includes v_stop as the first row (index 0).
    """

    def __init__(
        self,
        kge_loader: KGELoader,
        kge_dim: int,
        fusion_type: str = "weighted",
        lambda_weight: float = 0.7,
    ):
        """
        Initialize candidate builder.

        Args:
            kge_loader: KGE embedding loader
            kge_dim: Dimension of KGE embeddings
            fusion_type: Fusion type ("rotate", "transe", or "weighted")
            lambda_weight: Weight for relation in weighted fusion (default: 0.7)
        """
        super().__init__()

        self.kge_loader = kge_loader
        self.kge_dim = kge_dim
        self.fusion_type = fusion_type
        self.lambda_weight = lambda_weight

        # Fusion function (pass lambda_weight for weighted fusion)
        self.fusion = get_fusion(fusion_type, lambda_weight=lambda_weight)

        # Learnable STOP vector
        self.v_stop = nn.Parameter(torch.randn(kge_dim))
        nn.init.normal_(self.v_stop, mean=0, std=0.02)

    def get_neighbors(
        self,
        current_entity: str,
        subgraph: Set[Tuple[str, str, str]],
        visited: Set[str],
    ) -> List[Tuple[str, str, str]]:
        """
        Get valid neighbor triples from current entity.

        Args:
            current_entity: Current entity label
            subgraph: Set of (head, relation, tail) triples
            visited: Set of already visited entity labels (for cycle prevention)

        Returns:
            List of valid (head, relation, tail) triples where:
            - head == current_entity
            - tail not in visited
            - Both relation and tail exist in KGE vocabulary
        """
        neighbors = []

        for h, r, t in subgraph:
            # Must start from current entity
            if h != current_entity:
                continue

            # Cycle prevention: don't revisit entities
            if t in visited:
                continue

            # Must have valid embeddings
            if not self.kge_loader.has_relation(r):
                continue
            if not self.kge_loader.has_entity(t):
                continue

            neighbors.append((h, r, t))

        return neighbors

    def build_candidate_matrix(
        self,
        neighbors: List[Tuple[str, str, str]],
    ) -> Tuple[torch.Tensor, List[Tuple[str, str, str]]]:
        """
        Build candidate matrix from neighbor triples.

        Args:
            neighbors: List of (head, relation, tail) triples

        Returns:
            Tuple of:
            - Candidate matrix C of shape [num_candidates + 1, kge_dim]
              where row 0 is v_stop
            - List of triples corresponding to rows 1..N (same order as neighbors)
        """
        device = self.v_stop.device

        # Start with normalized STOP vector
        v_stop_normalized = F.normalize(self.v_stop, p=2, dim=-1)
        candidates = [v_stop_normalized]

        # Build candidate vector for each neighbor
        for h, r, t in neighbors:
            # Get embeddings (already normalized by KGELoader)
            e_r = self.kge_loader.get_relation_embedding(r)
            e_t = self.kge_loader.get_entity_embedding(t)

            # Fuse: c = Fusion(e_r, e_t)
            c = self.fusion(e_r, e_t)

            candidates.append(c)

        # Stack into matrix
        C = torch.stack(candidates, dim=0)  # [N+1, kge_dim]

        return C, neighbors

    def build_candidate_matrix_no_stop(
        self,
        neighbors: List[Tuple[str, str, str]],
    ) -> Tuple[torch.Tensor, List[Tuple[str, str, str]]]:
        """
        Build candidate matrix WITHOUT STOP vector (for Fixed-k mode).

        In Fixed-k mode, STOP should not be in the action space during the
        first k-1 steps. This method builds a candidate matrix containing
        only expansion actions (neighbors), so softmax is computed only
        over valid expansions.

        Args:
            neighbors: List of (head, relation, tail) triples

        Returns:
            Tuple of:
            - Candidate matrix C of shape [num_candidates, kge_dim] (NO STOP)
              Returns None if neighbors is empty
            - List of triples corresponding to rows (same order as neighbors)
        """
        if not neighbors:
            return None, []

        candidates = []

        # Build candidate vector for each neighbor (NO STOP)
        for h, r, t in neighbors:
            # Get embeddings (already normalized by KGELoader)
            e_r = self.kge_loader.get_relation_embedding(r)
            e_t = self.kge_loader.get_entity_embedding(t)

            # Fuse: c = Fusion(e_r, e_t)
            c = self.fusion(e_r, e_t)

            candidates.append(c)

        # Stack into matrix
        C = torch.stack(candidates, dim=0)  # [N, kge_dim]

        return C, neighbors

    def build_candidates_for_entity(
        self,
        current_entity: str,
        subgraph: Set[Tuple[str, str, str]],
        visited: Set[str],
    ) -> Tuple[torch.Tensor, List[Tuple[str, str, str]]]:
        """
        Build candidate matrix for a given entity.

        Convenience method combining get_neighbors and build_candidate_matrix.

        Args:
            current_entity: Current entity label
            subgraph: Set of (head, relation, tail) triples
            visited: Set of visited entity labels

        Returns:
            Tuple of:
            - Candidate matrix C of shape [num_candidates + 1, kge_dim]
            - List of neighbor triples (index 0 = STOP, indices 1..N = neighbors)
        """
        neighbors = self.get_neighbors(current_entity, subgraph, visited)
        return self.build_candidate_matrix(neighbors)

    def get_candidate_for_triple(
        self,
        triple: Tuple[str, str, str],
    ) -> torch.Tensor:
        """
        Get candidate vector for a single triple.

        Args:
            triple: (head, relation, tail) triple

        Returns:
            Candidate vector of shape [kge_dim]
        """
        h, r, t = triple

        e_r = self.kge_loader.get_relation_embedding(r)
        e_t = self.kge_loader.get_entity_embedding(t)

        return self.fusion(e_r, e_t)

    def get_stop_vector(self) -> torch.Tensor:
        """
        Get the normalized STOP vector.

        Returns:
            STOP vector of shape [kge_dim], L2-normalized
        """
        return F.normalize(self.v_stop, p=2, dim=-1)
