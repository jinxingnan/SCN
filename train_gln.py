"""
SCN Training Script

Main entry point for training the SCN (Generative Latent Navigation) model.

Training uses teacher forcing with InfoNCE contrastive loss:
- At each step, compute navigation vector from question + path history
- Compute InfoNCE loss against gold action with hard/easy negatives
- Aggregate losses across all steps in the path
"""
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from scn.config import SCNConfig
from scn.model import SCNReasoner
from scn.losses import InfoNCELoss, MultiPositiveInfoNCELoss
from scn.state import StateManager, trace_gold_path, ReasoningState
from scn.training_utils import get_step_training_data
from embeddings.kge_loader import KGELoader
from data.dataset import SCNDataset, SCNSample, create_dataloaders


class SCNTrainer:
    """
    Trainer for SCN model using teacher forcing with InfoNCE loss.
    """

    def __init__(
        self,
        model: SCNReasoner,
        config: SCNConfig,
        device: torch.device,
    ):
        """
        Initialize trainer.

        Args:
            model: SCN model
            config: Training configuration
            device: Target device
        """
        self.model = model
        self.config = config
        self.device = device

        # Loss function - use Multi-Positive InfoNCE for multi-answer questions
        self.loss_fn = MultiPositiveInfoNCELoss(
            temperature=config.temperature,
            hard_negative_weight=config.hard_negative_weight,
        )

        # Single-positive objective for validation
        self.single_loss_fn = InfoNCELoss(
            temperature=config.temperature,
            hard_negative_weight=config.hard_negative_weight,
        )

        # Optimizer with parameter groups
        param_groups = model.get_parameter_groups()
        self.optimizer = AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = None  # Will be set after knowing total steps

        # Gradient accumulation
        self.grad_accum_steps = config.gradient_accumulation_steps
        self.accum_count = 0

        # Metrics tracking
        self.step_count = 0
        self.epoch_count = 0

    def set_scheduler(self, total_steps: int) -> None:
        """Set learning rate scheduler after knowing total training steps."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )

    def _train_single_path(
        self,
        question: str,
        path: List[Tuple[str, str, str]],
        all_answers: List[str],
        state_manager: StateManager,
    ) -> List[torch.Tensor]:
        """
        Train on a single path and return step losses.

        Args:
            question: Natural language question
            path: Gold path as list of (h, r, t) triples
            all_answers: All valid answers for this question
            state_manager: StateManager for getting valid actions

        Returns:
            List of step losses (including terminal STOP loss)
        """
        # Path start entity is the first triple's head
        topic_entity = path[0][0]

        # Trace through path to get states at each step
        states = trace_gold_path(topic_entity, path)

        step_losses = []

        # === Phase 1: Train on path steps (move forward) ===
        for step_idx in range(len(path)):
            current_state = states[step_idx]
            gold_triple = path[step_idx]

            # Get navigation vector for current state
            v_nav = self.model.get_navigation_vector(
                question=question,
                path_history=current_state.path_history,
                current_entity=current_state.get_current_entity(),
                use_soft_prompt=True,
            )

            # Get valid neighbors from current entity
            neighbors = state_manager.get_valid_actions(current_state)

            # Raise error if no valid neighbors (data should be clean)
            if not neighbors:
                raise ValueError(
                    f"Step {step_idx}: No valid neighbors found! "
                    f"Current entity: {current_state.get_current_entity()}, "
                    f"Gold triple: {gold_triple}"
                )

            # Use get_step_training_data for proper positive/negative classification
            step_data = get_step_training_data(
                current_state=current_state,
                gold_triple=gold_triple,
                neighbors=neighbors,
                all_answers=all_answers,
                step_idx=step_idx,
                path_length=len(path),
                include_stop_in_negatives=True,
            )

            # Build positives
            c_pos_list = []
            for triple in step_data["positive_triples"]:
                c = self.model.candidate_builder.get_candidate_for_triple(triple)
                c_pos_list.append(c)

            if not c_pos_list:
                raise ValueError(
                    f"Step {step_idx}: No positive triples found! "
                    f"Gold triple: {gold_triple}, Neighbors: {neighbors}"
                )

            c_positives = torch.stack(c_pos_list, dim=0)

            # Build easy negatives (v_stop)
            c_easy_neg_list = []
            if step_data["include_stop_in_negatives"]:
                v_stop = self.model.candidate_builder.get_stop_vector()
                c_easy_neg_list.append(v_stop)
            c_easy_negatives = torch.stack(c_easy_neg_list, dim=0) if c_easy_neg_list else None

            # Build regular negatives (triples NOT leading to answers)
            c_neg_list = []
            for triple in step_data["negative_triples"]:
                c = self.model.candidate_builder.get_candidate_for_triple(triple)
                c_neg_list.append(c)
            c_negatives = torch.stack(c_neg_list, dim=0) if c_neg_list else None

            # Combine easy negatives with regular negatives
            if c_easy_negatives is not None and c_negatives is not None:
                c_all_negatives = torch.cat([c_easy_negatives, c_negatives], dim=0)
            elif c_easy_negatives is not None:
                c_all_negatives = c_easy_negatives
            else:
                c_all_negatives = c_negatives

            step_loss = self.loss_fn(v_nav, c_positives, c_all_negatives)
            step_losses.append(step_loss)

        # === Phase 2: Train terminal STOP step ===
        answer_entity = path[-1][2]
        visited_on_path = {triple[0] for triple in path}

        terminal_state = ReasoningState(
            frontier=answer_entity,
            visited=visited_on_path,
            path_history=list(path),
        )

        v_nav_terminal = self.model.get_navigation_vector(
            question=question,
            path_history=list(path),
            current_entity=answer_entity,
            use_soft_prompt=True,
        )

        terminal_neighbors = state_manager.get_valid_actions(terminal_state)

        v_stop = self.model.candidate_builder.get_stop_vector()
        c_positives_terminal = v_stop.unsqueeze(0)

        c_neg_terminal_list = []
        for triple in terminal_neighbors:
            c = self.model.candidate_builder.get_candidate_for_triple(triple)
            c_neg_terminal_list.append(c)

        c_negatives_terminal = torch.stack(c_neg_terminal_list, dim=0) if c_neg_terminal_list else None

        terminal_loss = self.loss_fn(v_nav_terminal, c_positives_terminal, c_negatives_terminal)
        step_losses.append(terminal_loss)

        return step_losses

    def train_step_for_sample(
        self,
        sample: SCNSample,
    ) -> Dict[str, float]:
        """
        Train on a single sample using teacher forcing with Multi-Positive InfoNCE.

        Multi-path training: trains on gold_path + all alternative_paths.
        Uses path-level gradient accumulation to avoid OOM:
        - Each path does forward + backward independently
        - Gradients accumulate on parameters
        - Memory usage = single path (not all paths)

        Args:
            sample: Training sample

        Returns:
            Dict of metrics for this sample
        """
        self.model.train()

        # Get sample data
        question = sample.question
        gold_path = sample.gold_path
        alternative_paths = sample.alternative_paths
        all_answers = sample.all_answers
        subgraph = sample.subgraph_triples

        # If gold path is empty, return zero loss
        if not gold_path:
            return {
                "loss": 0.0,
                "num_steps": 0,
                "num_paths": 0,
                "path_length": 0,
            }

        # Collect all paths to train
        all_paths = [gold_path] + alternative_paths
        num_paths = len(all_paths)

        # State manager for getting valid actions (shared across all paths)
        state_manager = StateManager(subgraph)

        # Train each path with path-level gradient accumulation
        # Each path does forward + backward independently to save memory
        accumulated_loss = 0.0
        total_steps = 0

        for path in all_paths:
            if not path:
                raise ValueError(f"Empty path in sample {sample.sample_id}")

            path_step_losses = self._train_single_path(
                question=question,
                path=path,
                all_answers=all_answers,
                state_manager=state_manager,
            )

            # Average all step losses equally
            step_losses_tensor = torch.stack(path_step_losses)
            path_loss = step_losses_tensor.mean()

            # Scale by: 1/num_paths (for path averaging) * 1/grad_accum_steps (for batch accumulation)
            scaled_path_loss = path_loss / (num_paths * self.grad_accum_steps)

            # Backward for this path only (gradients accumulate on parameters)
            scaled_path_loss.backward()

            # Track metrics (detached from computation graph)
            accumulated_loss += path_loss.item()
            total_steps += len(path_step_losses)

        # Return averaged loss for logging
        return {
            "loss": accumulated_loss / num_paths,
            "num_steps": total_steps,
            "num_paths": num_paths,
            "path_length": len(gold_path),
        }

    def _extract_step_hard_negatives(
        self,
        step_idx: int,
        gold_triple: Tuple[str, str, str],
        alternative_paths: List[List[Tuple]],
    ) -> Optional[torch.Tensor]:
        """
        Extract hard negatives for a specific step.

        Hard negatives at step i are triples from alternative_paths
        that have the same position (step_idx) but different relation.

        Args:
            step_idx: Current step index
            gold_triple: Gold triple at this step
            alternative_paths: List of alternative valid paths

        Returns:
            Tensor of hard negative candidates or None
        """
        gold_h, gold_r, gold_t = gold_triple

        hard_neg_triples = []

        for path in alternative_paths:
            # Path must have enough steps
            if step_idx >= len(path):
                continue

            neg_triple = path[step_idx]
            neg_h, neg_r, neg_t = neg_triple

            # Hard negative: same target entity, different relation
            # Or: same head, different relation
            if neg_h == gold_h and neg_r != gold_r:
                hard_neg_triples.append(neg_triple)

        if not hard_neg_triples:
            return None

        # Build candidate vectors
        hard_neg_candidates = []
        for triple in hard_neg_triples:
            # Check if triple exists in KGE vocabulary
            h, r, t = triple
            if (self.model.kge_loader.has_relation(r) and
                self.model.kge_loader.has_entity(t)):
                c = self.model.candidate_builder.get_candidate_for_triple(triple)
                hard_neg_candidates.append(c)

        if not hard_neg_candidates:
            return None

        return torch.stack(hard_neg_candidates, dim=0)

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        # Clip gradients
        nn.utils.clip_grad_norm_(
            self.model.get_trainable_parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        self.step_count += 1
        self.accum_count = 0

    def train_epoch(
        self,
        train_loader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dict of aggregated metrics
        """
        self.model.train()
        self.epoch_count = epoch

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_paths = 0
        num_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            # Process each sample in batch
            for sample in batch:
                metrics = self.train_step_for_sample(sample)

                epoch_loss += metrics["loss"]
                epoch_steps += metrics["num_steps"]
                epoch_paths += metrics.get("num_paths", 1)
                num_samples += 1

                self.accum_count += 1

                # Gradient accumulation
                if self.accum_count >= self.grad_accum_steps:
                    self.optimizer_step()

            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = epoch_loss / max(num_samples, 1)
                avg_paths = epoch_paths / max(num_samples, 1)
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | Paths/Sample: {avg_paths:.2f} | LR: {lr:.2e}"
                )

        # Final optimizer step if there are remaining gradients
        if self.accum_count > 0:
            self.optimizer_step()

        return {
            "epoch_loss": epoch_loss / max(num_samples, 1),
            "total_steps": epoch_steps,
            "total_paths": epoch_paths,
            "avg_paths_per_sample": epoch_paths / max(num_samples, 1),
            "num_samples": num_samples,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader,
    ) -> Dict[str, float]:
        """
        Validate model on validation set.

        Performs two types of evaluation:
        1. Teacher forcing: Loss and path accuracy (original metrics)
        2. Beam search inference: Answer accuracy and reasoning statistics (new metrics)

        Args:
            val_loader: Validation data loader

        Returns:
            Dict of validation metrics including answer accuracy
        """
        self.model.eval()

        # Teacher forcing metrics (original)
        total_loss = 0.0
        total_steps = 0
        num_samples_with_gold_path = 0
        correct_first_step = 0
        correct_full_path = 0

        # Answer accuracy metrics (KGQA standard)
        hits_at_1_count = 0
        answer_correct_any_beam = 0
        num_samples_with_answers = 0

        # F1 metrics (KGQA standard)
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        # Reasoning statistics (new)
        total_path_length = 0
        total_reasoning_steps = 0
        stop_count = 0
        total_beam_diversity = 0
        no_answer_count = 0

        # Create beam search inference
        from scn.inference import BeamSearchInference
        beam_search = BeamSearchInference(
            model=self.model,
            beam_size=5,
            max_hops=5,
        )

        total_samples = 0

        for batch in val_loader:
            for sample in batch:
                total_samples += 1

                question = sample.question
                topic_entities = sample.topic_entities
                gold_path = sample.gold_path
                subgraph = sample.subgraph_triples
                all_answers = sample.all_answers

                # ========================================
                # Part 1: Teacher forcing evaluation (original metrics)
                # ========================================
                if gold_path and len(gold_path) > 0:
                    topic_entity = topic_entities[0]
                    states = trace_gold_path(topic_entity, gold_path)
                    state_manager = StateManager(subgraph)

                    step_losses = []
                    all_correct = True

                    for step_idx in range(len(gold_path)):
                        current_state = states[step_idx]
                        gold_triple = gold_path[step_idx]

                        # Get navigation vector (SCN uses soft prompt)
                        v_nav = self.model.get_navigation_vector(
                            question=question,
                            path_history=current_state.path_history,
                            current_entity=current_state.get_current_entity(),
                            use_soft_prompt=True,
                        )

                        neighbors = state_manager.get_valid_actions(current_state)

                        # Process this step only if there are valid neighbors
                        if neighbors and len(neighbors) > 0:
                            # Build candidate matrix
                            candidate_matrix, neighbor_list = self.model.candidate_builder.build_candidate_matrix(
                                neighbors
                            )

                            # Compute action probabilities
                            log_probs = self.model.compute_action_log_probs(v_nav, candidate_matrix)

                            # Find gold action index
                            gold_idx = -1
                            for i, triple in enumerate(neighbor_list):
                                if triple == gold_triple:
                                    gold_idx = i + 1  # +1 because STOP is at index 0
                                    break

                            # Process only if gold action found in neighbors
                            if gold_idx != -1:
                                # Check if prediction is correct
                                pred_idx = torch.argmax(log_probs).item()
                                if pred_idx != gold_idx:
                                    all_correct = False

                                # First step accuracy
                                if step_idx == 0 and pred_idx == gold_idx:
                                    correct_first_step += 1

                                # Compute loss
                                c_positive = self.model.candidate_builder.get_candidate_for_triple(gold_triple)

                                easy_neg_list = []
                                for triple in neighbors:
                                    if triple != gold_triple:
                                        c = self.model.candidate_builder.get_candidate_for_triple(triple)
                                        easy_neg_list.append(c)

                                c_easy_negatives = None
                                if len(easy_neg_list) > 0:
                                    c_easy_negatives = torch.stack(easy_neg_list, dim=0)

                                # Compute the single-positive validation loss
                                step_loss = self.single_loss_fn(
                                    v_nav=v_nav,
                                    c_positive=c_positive,
                                    c_easy_negatives=c_easy_negatives,
                                    c_hard_negatives=None,
                                )
                                step_losses.append(step_loss)
                            else:
                                all_correct = False
                        else:
                            all_correct = False

                    # Aggregate teacher forcing metrics
                    if len(step_losses) > 0:
                        sample_loss = torch.stack(step_losses).mean().item()
                        total_loss += sample_loss
                        total_steps += len(step_losses)

                    if all_correct:
                        correct_full_path += 1

                    num_samples_with_gold_path += 1

                # ========================================
                # Part 2: Beam search inference (answer accuracy)
                # ========================================
                if all_answers and len(all_answers) > 0 and topic_entities and len(topic_entities) > 0:
                    # Run beam search
                    result = beam_search.search(
                        question=question,
                        topic_entities=topic_entities,
                        subgraph=subgraph,
                    )

                    # Hits@1: Top-1 strict matching
                    if result.best_answer and result.best_answer in all_answers:
                        hits_at_1_count += 1

                    # F1 Score: Compute precision, recall, F1 for Top-K beam answers
                    # Collect unique answers from Top-K beam paths
                    # IMPORTANT: Only use paths that have STOP signal (completed paths)
                    predicted_answers = set()
                    K = 5  # Use Top-5 beam paths (or beam_size)
                    for beam_path in result.all_paths[:K]:
                        # Only count answers from paths that chose to STOP
                        # This avoids counting incomplete/forced paths
                        if beam_path.is_stopped:
                            answer = beam_path.get_answer()
                            if answer:
                                predicted_answers.add(answer)

                    gold_answers = set(all_answers)

                    if len(predicted_answers) == 0 and len(gold_answers) == 0:
                        # Both empty: perfect match
                        precision = 1.0
                        recall = 1.0
                        f1 = 1.0
                    elif len(predicted_answers) == 0:
                        # No prediction
                        precision = 0.0
                        recall = 0.0
                        f1 = 0.0
                    elif len(gold_answers) == 0:
                        # No gold answer (shouldn't happen)
                        precision = 0.0
                        recall = 0.0
                        f1 = 0.0
                    else:
                        # Normal case: compute precision, recall, F1
                        true_positives = len(predicted_answers & gold_answers)
                        precision = true_positives / len(predicted_answers)

                        # Recall with normalization: denominator is min(K, |gold|)
                        # This ensures fairness for questions with |gold| > K
                        effective_gold_count = min(K, len(gold_answers))
                        recall = true_positives / effective_gold_count

                        if precision + recall == 0:
                            f1 = 0.0
                        else:
                            f1 = 2 * precision * recall / (precision + recall)

                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1

                    # Any-in-beam loose matching
                    any_correct = False
                    for beam_path in result.all_paths:
                        path_answer = beam_path.get_answer()
                        if path_answer and path_answer in all_answers:
                            any_correct = True
                            break

                    if any_correct:
                        answer_correct_any_beam += 1

                    # Collect reasoning statistics
                    path_length = len(result.best_path)
                    total_path_length += path_length
                    total_reasoning_steps += result.num_steps

                    # Check if model chose STOP (path stopped before max_hops)
                    if result.num_steps < 5:
                        # Find if best path in beam is stopped
                        for beam_path in result.all_paths:
                            if beam_path.get_answer() == result.best_answer and beam_path.is_stopped:
                                stop_count += 1
                                break

                    # Beam diversity: count unique answers in beam
                    unique_answers = set()
                    for beam_path in result.all_paths:
                        ans = beam_path.get_answer()
                        if ans:
                            unique_answers.add(ans)
                    total_beam_diversity += len(unique_answers)

                    # No answer (empty path - model stopped immediately)
                    if path_length == 0:
                        no_answer_count += 1

                    num_samples_with_answers += 1

        # Compute final metrics
        return {
            # Teacher forcing metrics (original)
            "val_loss": total_loss / max(num_samples_with_gold_path, 1),
            "first_step_acc": correct_first_step / max(num_samples_with_gold_path, 1),
            "full_path_acc": correct_full_path / max(num_samples_with_gold_path, 1),

            # KGQA standard metrics (WebQSP/CWQ)
            "hits_at_1": hits_at_1_count / max(num_samples_with_answers, 1),
            "precision": total_precision / max(num_samples_with_answers, 1),
            "recall": total_recall / max(num_samples_with_answers, 1),
            "f1": total_f1 / max(num_samples_with_answers, 1),

            # Additional answer metrics
            "answer_acc_any_beam": answer_correct_any_beam / max(num_samples_with_answers, 1),

            # Reasoning statistics
            "avg_path_length": total_path_length / max(num_samples_with_answers, 1),
            "avg_reasoning_steps": total_reasoning_steps / max(num_samples_with_answers, 1),
            "stop_rate": stop_count / max(num_samples_with_answers, 1),
            "avg_beam_diversity": total_beam_diversity / max(num_samples_with_answers, 1),
            "no_answer_rate": no_answer_count / max(num_samples_with_answers, 1),

            # Sample counts
            "num_samples": total_samples,
            "num_samples_with_gold_path": num_samples_with_gold_path,
            "num_samples_with_answers": num_samples_with_answers,
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "step": self.step_count,
            "model_state_dict": {
                "input_projector": self.model.input_projector.state_dict(),
                "output_projector": self.model.output_projector.state_dict(),
                "candidate_builder": self.model.candidate_builder.state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config.to_dict(),
        }

        # Save LoRA weights separately if enabled
        if self.config.use_lora:
            # Use consistent naming: lora_best for best_model.pt, lora_epoch_X for periodic checkpoints
            checkpoint_name = Path(path).stem  # e.g., "best_model" or "checkpoint_epoch_5"
            if checkpoint_name == "best_model":
                lora_path = Path(path).parent / "lora_best"
            else:
                lora_path = Path(path).parent / f"lora_epoch_{epoch}"
            self.model.llm.save_lora_weights(str(lora_path))
            checkpoint["lora_path"] = str(lora_path)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        model_state = checkpoint["model_state_dict"]

        # Load SCN components (required)
        self.model.input_projector.load_state_dict(model_state["input_projector"])
        self.model.output_projector.load_state_dict(model_state["output_projector"])
        self.model.candidate_builder.load_state_dict(model_state["candidate_builder"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.step_count = checkpoint["step"]

        # Load LoRA weights if available
        if "lora_path" in checkpoint and self.config.use_lora:
            lora_path = Path(checkpoint["lora_path"])

            # Handle relative path: resolve relative to checkpoint directory
            if not lora_path.is_absolute():
                checkpoint_dir = Path(path).parent
                lora_path = checkpoint_dir / lora_path.name  # e.g., lora_best

            self.model.llm.load_lora_weights(str(lora_path))

        return checkpoint


def train(
    config: SCNConfig,
    train_file: str,
    val_file: Optional[str],
    train_ratio: float,
    kge_embeddings_dir: str,
    kge_mapping_dir: str,
    output_dir: str,
    resume_from: Optional[str] = None,
) -> None:
    """
    Main training function.

    Args:
        config: SCN configuration
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file (if None, splits train_file)
        train_ratio: Train/val split ratio (only used if val_file is None)
        kge_embeddings_dir: Path to KGE embeddings
        kge_mapping_dir: Path to KGE mappings
        output_dir: Output directory for checkpoints
        resume_from: Optional checkpoint to resume from
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config.to_yaml(str(output_path / "config.yaml"))

    # Load KGE embeddings
    print("Loading KGE embeddings...")
    kge_loader = KGELoader(
        embeddings_dir=kge_embeddings_dir,
        mapping_dir=kge_mapping_dir,
        device=device,
    )

    # Create model
    print("Creating model...")
    model = SCNReasoner(
        config=config,
        kge_loader=kge_loader,
        device=device,
    )

    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"Parameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        train_ratio=train_ratio,
        batch_size=config.batch_size,
        use_multi_path_per_pair=config.use_multi_path_per_pair,
    )

    # Create trainer
    trainer = SCNTrainer(model, config, device)

    # Calculate total steps and set scheduler
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    trainer.set_scheduler(total_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        print(f"Resuming from {resume_from}")
        checkpoint = trainer.load_checkpoint(resume_from)
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    best_hits_at_1 = 0.0  # Use Hits@1 as main metric (KGQA standard)

    print(f"\nStarting training for {config.num_epochs} epochs...")
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(val_loader.dataset)}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print()

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['epoch_loss']:.4f}")
        print(f"  Samples: {train_metrics['num_samples']}")
        print(f"  Total paths trained: {train_metrics['total_paths']}")
        print(f"  Avg paths/sample: {train_metrics['avg_paths_per_sample']:.2f}")

        # Validate
        if (epoch + 1) % config.val_interval == 0:
            val_metrics = trainer.validate(val_loader)
            print(f"\nValidation metrics:")
            print(f"  [KGQA Standard Metrics (WebQSP/CWQ)]")
            print(f"    Hits@1: {val_metrics['hits_at_1']:.4f}")
            print(f"    F1: {val_metrics['f1']:.4f}")
            print(f"    Precision: {val_metrics['precision']:.4f}")
            print(f"    Recall: {val_metrics['recall']:.4f}")
            print(f"  [Teacher Forcing Metrics]")
            print(f"    Loss: {val_metrics['val_loss']:.4f}")
            print(f"    First step accuracy: {val_metrics['first_step_acc']:.4f}")
            print(f"    Full path accuracy: {val_metrics['full_path_acc']:.4f}")
            print(f"  [Additional Metrics]")
            print(f"    Any-in-beam accuracy: {val_metrics['answer_acc_any_beam']:.4f}")
            print(f"    Avg path length: {val_metrics['avg_path_length']:.2f}")
            print(f"    Avg reasoning steps: {val_metrics['avg_reasoning_steps']:.2f}")
            print(f"    STOP rate: {val_metrics['stop_rate']:.4f}")
            print(f"    Avg beam diversity: {val_metrics['avg_beam_diversity']:.2f}")
            print(f"    No answer rate: {val_metrics['no_answer_rate']:.4f}")
            print(f"  [Sample Counts]")
            print(f"    Total: {val_metrics['num_samples']}")
            print(f"    With gold path: {val_metrics['num_samples_with_gold_path']}")
            print(f"    With answers: {val_metrics['num_samples_with_answers']}")

            # Save best model based on Hits@1 (KGQA standard metric)
            if val_metrics["hits_at_1"] > best_hits_at_1:
                best_hits_at_1 = val_metrics["hits_at_1"]
                trainer.save_checkpoint(
                    str(output_path / "best_model.pt"),
                    epoch,
                    {**train_metrics, **val_metrics},
                )
                print(f"  -> New best model saved! (Hits@1: {best_hits_at_1:.4f}, F1: {val_metrics['f1']:.4f})")

        # Save periodic checkpoint
        if not config.save_best_only:
            trainer.save_checkpoint(
                str(output_path / f"checkpoint_epoch_{epoch + 1}.pt"),
                epoch,
                train_metrics,
            )

    print("\nTraining complete!")
    print(f"Best Hits@1: {best_hits_at_1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train SCN model")

    # Config file (only configuration source)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )

    # Resume from directory (will load config.yaml and best_model.pt from directory)
    parser.add_argument(
        "--resume_from_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint to resume from (will load config.yaml and best_model.pt)",
    )

    # Optional overrides for dataset paths when resuming
    parser.add_argument(
        "--override_train_file",
        type=str,
        default=None,
        help="Override train file path when resuming (optional)",
    )
    parser.add_argument(
        "--override_val_file",
        type=str,
        default=None,
        help="Override validation file path when resuming (optional)",
    )
    parser.add_argument(
        "--override_num_epochs",
        type=int,
        default=None,
        help="Override total number of epochs when resuming (optional)",
    )

    args = parser.parse_args()

    # Handle resume_from_dir (takes precedence over config and resume_from)
    resume_checkpoint_path = None
    if args.resume_from_dir:
        resume_dir = Path(args.resume_from_dir)
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        
        # Load config from the checkpoint directory
        config_path = resume_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found in resume directory: {config_path}")
        
        print(f"Resuming from directory: {resume_dir}")
        print(f"Loading config from {config_path}")
        config = SCNConfig.from_yaml(str(config_path))
        
        # Set checkpoint path to best_model.pt from the directory
        resume_checkpoint_path = resume_dir / "best_model.pt"
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_path}")
        print(f"Loading checkpoint from {resume_checkpoint_path}")
    else:
        # Load config from YAML file (normal flow)
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"Loading config from {config_path}")
        config = SCNConfig.from_yaml(str(config_path))

    # Add timestamp to output_dir (always create new directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.output_dir}_{timestamp}"
    print(f"Output directory: {output_dir}")
    if args.resume_from_dir:
        print(f"Note: Continuing training from {args.resume_from_dir}")
        print(f"      New checkpoints will be saved to: {output_dir}")

    # Apply dataset overrides if specified
    if args.override_train_file:
        print(f"Overriding train file: {config.train_file} -> {args.override_train_file}")
        config.train_file = args.override_train_file
    if args.override_val_file:
        print(f"Overriding validation file: {config.val_file} -> {args.override_val_file}")
        config.val_file = args.override_val_file
    if args.override_num_epochs:
        print(f"Overriding num_epochs: {config.num_epochs} -> {args.override_num_epochs}")
        config.num_epochs = args.override_num_epochs

    # Run training (all parameters from config)
    train(
        config=config,
        train_file=config.train_file,
        val_file=config.val_file,
        train_ratio=config.train_ratio,
        kge_embeddings_dir=config.kge_embeddings_dir,
        kge_mapping_dir=config.kge_mapping_dir,
        output_dir=output_dir,
        resume_from=str(resume_checkpoint_path) if resume_checkpoint_path else None,
    )


if __name__ == "__main__":
    main()
