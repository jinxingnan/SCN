"""
Dataset for SCN Training

Loads samples from a single JSONL file containing:
- Question and topic entities
- Gold reasoning paths (ground_truth_paths)
- Answer entities
- Subgraph triples (graph)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Set, Optional
from torch.utils.data import Dataset, DataLoader


@dataclass
class SCNSample:
    """
    A single training/evaluation sample for SCN.

    Attributes:
        sample_id: Unique identifier
        question: Natural language question
        topic_entities: Starting entities from entity linking
        gold_path: Gold reasoning path as list of (h, r, t) triples
        answer_entity: The correct answer entity
        all_answers: All valid answers (for evaluation)
        subgraph_triples: Local subgraph as set of triples
        alternative_paths: Other valid paths (can be used for multi-path training)
    """
    sample_id: str
    question: str
    topic_entities: List[str]
    gold_path: List[Tuple[str, str, str]]
    answer_entity: str
    all_answers: List[str]
    subgraph_triples: Set[Tuple[str, str, str]]
    alternative_paths: List[List[Tuple[str, str, str]]] = field(default_factory=list)


class SCNDataset(Dataset):
    """
    PyTorch Dataset for SCN training.

    Loads samples from a single JSONL file with format:
    {
        "id": "WebQTrn-0",
        "question": "what is the name of justin bieber brother",
        "answer": ["Jaxon Bieber"],
        "q_entity": ["Justin Bieber"],
        "a_entity": ["Jaxon Bieber"],
        "graph": [["entity1", "relation", "entity2"], ...],
        "ground_truth_paths": [[["h1", "r1", "t1"], ["h2", "r2", "t2"]], ...]
    }
    """

    def __init__(
        self,
        data_file: str,
        split_ratio: float = 1.0,
        split_type: str = "train",
        random_seed: int = 42,
        use_multi_path_per_pair: bool = False,
        skip_no_paths: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_file: Path to JSONL file
            split_ratio: Ratio for train/val split (1.0 = use all data)
            split_type: "train" or "val" (only used if split_ratio < 1.0)
            random_seed: Random seed for splitting
            use_multi_path_per_pair: If True, train all paths for same (topic, answer) pair
                                     If False, only one path per (topic, answer) pair
            skip_no_paths: If True (default), skip samples without ground_truth_paths
                          If False, load all samples (for inference/evaluation)
        """
        self.data_file = Path(data_file)
        self.split_ratio = split_ratio
        self.split_type = split_type
        self.random_seed = random_seed
        self.use_multi_path_per_pair = use_multi_path_per_pair
        self.skip_no_paths = skip_no_paths

        # Load data
        self.samples: List[SCNSample] = []
        self._load_data()

        print(f"SCNDataset loaded: {len(self.samples)} samples from {self.data_file.name}")
        print(f"  Multi-path per pair: {self.use_multi_path_per_pair}")

    def _load_data(self) -> None:
        """Load data from JSONL file."""
        all_samples = []

        with open(self.data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Extract fields
                sample_id = data["id"]
                question = data["question"]
                topic_entities = data["q_entity"]
                all_answers = data.get("a_entity", data.get("answer", []))

                # Get subgraph
                graph = data["graph"]
                subgraph = set()
                for triple in graph:
                    if len(triple) == 3:
                        subgraph.add((str(triple[0]), str(triple[1]), str(triple[2])))

                # Get ground truth paths
                ground_truth_paths = data.get("ground_truth_paths", [])
                if not ground_truth_paths and self.skip_no_paths:
                    # Skip samples without ground truth paths (only if skip_no_paths=True)
                    continue

                # Group paths by their starting topic entity
                # This enables multi-path training from all topic entities
                topic_entity_set = set(topic_entities)
                paths_by_topic = {topic: [] for topic in topic_entities}

                for path_raw in ground_truth_paths:
                    if path_raw and len(path_raw) > 0:
                        path_start = path_raw[0][0]
                        if path_start in topic_entity_set:
                            path_tuples = [tuple(step) for step in path_raw]
                            paths_by_topic[path_start].append(path_tuples)

                # Collect all valid paths (from any topic entity)
                # Mode 1: use_multi_path_per_pair=True -> use ALL paths
                # Mode 2: use_multi_path_per_pair=False -> one path per (topic, answer) pair
                all_valid_paths = []

                if self.use_multi_path_per_pair:
                    # Original logic: collect ALL paths
                    for topic in topic_entities:
                        all_valid_paths.extend(paths_by_topic[topic])
                else:
                    # New logic: Only one path per (topic_entity, answer_entity) pair
                    # Group paths by (topic, answer) pair
                    paths_by_pair = {}  # {(topic, answer): [paths]}

                    for topic in topic_entities:
                        for path in paths_by_topic[topic]:
                            # Get answer entity (last entity in path)
                            answer = path[-1][2]
                            pair_key = (topic, answer)

                            if pair_key not in paths_by_pair:
                                paths_by_pair[pair_key] = []
                            paths_by_pair[pair_key].append(path)

                    # Select one path per pair (use first path for determinism)
                    for pair_key, paths in paths_by_pair.items():
                        all_valid_paths.append(paths[0])

                # If no valid paths found, skip this sample (only if skip_no_paths=True)
                if not all_valid_paths and self.skip_no_paths:
                    continue

                # Store the first path as gold_path
                # alternative_paths = all remaining paths (for multi-path training)
                # If no paths available (inference mode), use empty defaults
                gold_path = all_valid_paths[0] if all_valid_paths else []
                alternative_paths = all_valid_paths[1:] if len(all_valid_paths) > 1 else []

                # Determine answer entity from gold path
                if gold_path:
                    answer_entity = gold_path[-1][2]  # tail of last triple
                elif all_answers:
                    answer_entity = all_answers[0]
                else:
                    answer_entity = topic_entities[0] if topic_entities else ""

                # Create sample
                sample = SCNSample(
                    sample_id=sample_id,
                    question=question,
                    topic_entities=topic_entities,
                    gold_path=gold_path,
                    answer_entity=answer_entity,
                    all_answers=all_answers,
                    subgraph_triples=subgraph,
                    alternative_paths=alternative_paths,
                )

                all_samples.append(sample)

        # Apply train/val split if needed
        if self.split_ratio < 1.0:
            import random
            random.seed(self.random_seed)
            random.shuffle(all_samples)

            split_idx = int(len(all_samples) * self.split_ratio)

            if self.split_type == "train":
                self.samples = all_samples[:split_idx]
            else:  # val
                self.samples = all_samples[split_idx:]
        else:
            self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SCNSample:
        return self.samples[idx]

    def get_sample_by_id(self, sample_id: str) -> Optional[SCNSample]:
        """Get sample by ID (linear search)."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None


def collate_fn(batch: List[SCNSample]) -> List[SCNSample]:
    """
    Collate function that returns samples as-is.

    Since samples have variable-length paths and subgraphs,
    we process them individually rather than batching tensors.
    """
    return batch


def create_dataloaders(
    train_file: str,
    val_file: Optional[str] = None,
    train_ratio: float = 0.9,
    batch_size: int = 1,
    num_workers: int = 0,
    use_multi_path_per_pair: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file (optional)
                  If None, will split train_file using train_ratio
        train_ratio: Train/val split ratio (only used if val_file is None)
        batch_size: Batch size (default: 1)
        num_workers: Number of data loading workers
        use_multi_path_per_pair: If True, train all paths for same (topic, answer) pair

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if val_file is not None:
        # Use separate files
        train_dataset = SCNDataset(
            data_file=train_file,
            use_multi_path_per_pair=use_multi_path_per_pair,
        )
        val_dataset = SCNDataset(
            data_file=val_file,
            use_multi_path_per_pair=use_multi_path_per_pair,
        )
    else:
        # Split single file
        train_dataset = SCNDataset(
            data_file=train_file,
            split_ratio=train_ratio,
            split_type="train",
            use_multi_path_per_pair=use_multi_path_per_pair,
        )
        val_dataset = SCNDataset(
            data_file=train_file,
            split_ratio=train_ratio,
            split_type="val",
            use_multi_path_per_pair=use_multi_path_per_pair,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
