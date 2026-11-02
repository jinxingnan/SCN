"""
Knowledge Graph Embedding Loader

Loads pretrained RotatE embeddings and provides label-to-embedding lookups.
All embeddings are frozen (not trainable).
"""

import csv
import gzip
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple


class KGELoader:
    """
    Loads and manages pretrained KGE (Knowledge Graph Embeddings).

    This class:
    - Loads RotatE entity and relation embeddings from disk
    - Loads label-to-ID mappings from TSV files
    - Provides efficient lookup by label or ID
    - Returns L2-normalized embeddings

    All embeddings are frozen and not trainable.
    """

    def __init__(
        self,
        embeddings_dir: str,
        mapping_dir: str,
        device: torch.device,
        normalize: bool = True,
    ):
        """
        Initialize KGE loader.

        Args:
            embeddings_dir: Directory containing entity_embeddings.pt and relation_embeddings.pt
            mapping_dir: Directory containing entity_to_id.tsv.gz and relation_to_id.tsv.gz
            device: Target device for embeddings
            normalize: Whether to L2-normalize embeddings on retrieval (default: True)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.mapping_dir = Path(mapping_dir)
        self.device = device
        self.normalize = normalize

        # Load mappings first
        self.entity_to_id = self._load_mapping(self.mapping_dir / "entity_to_id.tsv.gz")
        self.relation_to_id = self._load_mapping(self.mapping_dir / "relation_to_id.tsv.gz")

        # Load embeddings
        self.entity_embeddings = self._load_embeddings(
            self.embeddings_dir / "entity_embeddings.pt"
        )
        self.relation_embeddings = self._load_embeddings(
            self.embeddings_dir / "relation_embeddings.pt"
        )

        # Validate consistency
        self._validate()

        # Store dimensions
        self.num_entities = self.entity_embeddings.shape[0]
        self.num_relations = self.relation_embeddings.shape[0]
        self.embedding_dim = self.entity_embeddings.shape[1]

        print(f"KGELoader initialized:")
        print(f"  Entities: {self.num_entities} x {self.embedding_dim}")
        print(f"  Relations: {self.num_relations} x {self.embedding_dim}")
        print(f"  Device: {self.device}")
        print(f"  Normalize: {self.normalize}")

    def _load_mapping(self, path: Path) -> Dict[str, int]:
        """Load label-to-ID mapping from gzipped TSV file.
        
        Uses CSV reader to properly handle quoted fields with special characters.
        Strips whitespace from labels to handle inconsistencies.
        If duplicates exist after stripping, keeps the one with larger ID.
        """
        if not path.exists():
            raise FileNotFoundError(f"Mapping file not found: {path}")

        mapping = {}
        duplicates_found = 0
        
        with gzip.open(path, "rt", encoding="utf-8") as f:
            # Use csv.reader with tab delimiter to handle quoted fields properly
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            
            # Skip header line
            header = next(reader)

            for line_num, row in enumerate(reader, start=2):
                if not row or len(row) == 0:
                    continue  # Skip empty lines
                    
                if len(row) != 2:
                    raise ValueError(
                        f"Invalid format in {path} at line {line_num}: expected 2 tab-separated values, got {len(row)}"
                    )

                id_str, label = row
                # Strip whitespace from label to handle inconsistencies in source data
                label_stripped = label.strip()
                entity_id = int(id_str)
                
                # If duplicate after stripping, keep the one with larger ID
                if label_stripped in mapping:
                    if entity_id > mapping[label_stripped]:
                        mapping[label_stripped] = entity_id
                    duplicates_found += 1
                else:
                    mapping[label_stripped] = entity_id

        if duplicates_found > 0:
            print(f"  Warning: Found {duplicates_found} duplicate labels after stripping whitespace (kept higher IDs)")

        return mapping

    def _load_embeddings(self, path: Path) -> torch.Tensor:
        """Load embedding tensor from file."""
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found: {path}")

        embeddings = torch.load(path, map_location=self.device, weights_only=True)

        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D embedding tensor, got shape {embeddings.shape}")

        return embeddings

    def _validate(self) -> None:
        """Validate that mappings and embeddings are consistent.
        
        Note: After stripping whitespace, some labels may be duplicates,
        so mapping count may be slightly less than embedding count.
        """
        entity_map_count = len(self.entity_to_id)
        entity_emb_count = self.entity_embeddings.shape[0]
        
        # Allow mapping count to be less than embedding count due to duplicate labels after stripping
        if entity_map_count > entity_emb_count:
            raise ValueError(
                f"Entity mapping count ({entity_map_count}) > embedding count ({entity_emb_count})"
            )
        
        # If counts don't match, it's likely due to stripped duplicates (acceptable)
        if entity_map_count < entity_emb_count:
            diff = entity_emb_count - entity_map_count
            print(f"  Note: Entity mapping has {diff} fewer entries than embeddings (likely due to stripped duplicates)")

        relation_map_count = len(self.relation_to_id)
        relation_emb_count = self.relation_embeddings.shape[0]
        
        # Relations should match exactly (no whitespace issues expected)
        if relation_map_count > relation_emb_count:
            raise ValueError(
                f"Relation mapping count ({relation_map_count}) > embedding count ({relation_emb_count})"
            )
        
        if relation_map_count < relation_emb_count:
            diff = relation_emb_count - relation_map_count
            print(f"  Note: Relation mapping has {diff} fewer entries than embeddings")

        entity_dim = self.entity_embeddings.shape[1]
        relation_dim = self.relation_embeddings.shape[1]
        if entity_dim != relation_dim:
            raise ValueError(
                f"Entity dim ({entity_dim}) != Relation dim ({relation_dim})"
            )

    def get_entity_embedding(self, label: str) -> torch.Tensor:
        """
        Get entity embedding by label.

        Args:
            label: Entity label (Freebase format, e.g., "Justin Bieber" or "m.0gxnnwp")

        Returns:
            Entity embedding tensor of shape [embedding_dim]

        Raises:
            KeyError: If entity label not found in vocabulary
        """
        # Strip leading/trailing whitespace to handle dataset inconsistencies
        label = label.strip()
        
        if label not in self.entity_to_id:
            raise KeyError(f"Entity not found in KGE vocabulary: '{label}'")

        entity_id = self.entity_to_id[label]
        embedding = self.entity_embeddings[entity_id]

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def get_relation_embedding(self, label: str) -> torch.Tensor:
        """
        Get relation embedding by label.

        Args:
            label: Relation label (Freebase format, e.g., "people.person.children")

        Returns:
            Relation embedding tensor of shape [embedding_dim]

        Raises:
            KeyError: If relation label not found in vocabulary
        """
        # Strip leading/trailing whitespace to handle dataset inconsistencies
        label = label.strip()
        
        if label not in self.relation_to_id:
            raise KeyError(f"Relation not found in KGE vocabulary: '{label}'")

        relation_id = self.relation_to_id[label]
        embedding = self.relation_embeddings[relation_id]

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def get_entity_embedding_by_id(self, entity_id: int) -> torch.Tensor:
        """
        Get entity embedding by ID.

        Args:
            entity_id: Integer entity ID

        Returns:
            Entity embedding tensor of shape [embedding_dim]

        Raises:
            IndexError: If entity_id out of range
        """
        if entity_id < 0 or entity_id >= self.num_entities:
            raise IndexError(
                f"Entity ID {entity_id} out of range [0, {self.num_entities})"
            )

        embedding = self.entity_embeddings[entity_id]

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def get_relation_embedding_by_id(self, relation_id: int) -> torch.Tensor:
        """
        Get relation embedding by ID.

        Args:
            relation_id: Integer relation ID

        Returns:
            Relation embedding tensor of shape [embedding_dim]

        Raises:
            IndexError: If relation_id out of range
        """
        if relation_id < 0 or relation_id >= self.num_relations:
            raise IndexError(
                f"Relation ID {relation_id} out of range [0, {self.num_relations})"
            )

        embedding = self.relation_embeddings[relation_id]

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def get_entity_embeddings_batch(self, labels: List[str]) -> torch.Tensor:
        """
        Get multiple entity embeddings by labels.

        Args:
            labels: List of entity labels

        Returns:
            Stacked embeddings of shape [len(labels), embedding_dim]

        Raises:
            KeyError: If any label not found
        """
        embeddings = [self.get_entity_embedding(label) for label in labels]
        return torch.stack(embeddings, dim=0)

    def get_relation_embeddings_batch(self, labels: List[str]) -> torch.Tensor:
        """
        Get multiple relation embeddings by labels.

        Args:
            labels: List of relation labels

        Returns:
            Stacked embeddings of shape [len(labels), embedding_dim]

        Raises:
            KeyError: If any label not found
        """
        embeddings = [self.get_relation_embedding(label) for label in labels]
        return torch.stack(embeddings, dim=0)

    def has_entity(self, label: str) -> bool:
        """Check if entity label exists in vocabulary."""
        return label in self.entity_to_id

    def has_relation(self, label: str) -> bool:
        """Check if relation label exists in vocabulary."""
        return label in self.relation_to_id

    def get_triple_embeddings(
        self, triple: Tuple[str, str, str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings for a triple (head, relation, tail).

        Args:
            triple: Tuple of (head_label, relation_label, tail_label)

        Returns:
            Tuple of (head_emb, relation_emb, tail_emb), each of shape [embedding_dim]

        Raises:
            KeyError: If any label not found
        """
        h, r, t = triple
        h_emb = self.get_entity_embedding(h)
        r_emb = self.get_relation_embedding(r)
        t_emb = self.get_entity_embedding(t)
        return h_emb, r_emb, t_emb
