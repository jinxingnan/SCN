"""
Inference Module for SCN

Implements beam search for multi-hop reasoning in the vector space.

Key features:
- Vector-based action selection using cosine similarity
- Beam search for exploring multiple paths
- Path aggregation using LogSumExp for answer scoring
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional
from collections import defaultdict

from .model import SCNReasoner
from .state import ReasoningState, StateManager


@dataclass
class BeamPath:
    """
    A single path in the beam.

    Attributes:
        path: List of (h, r, t) triples traversed
        frontier: Current entity
        visited: Set of visited entities
        score: Accumulated log-probability
        is_stopped: Whether STOP was selected
        navigation_vectors: List of navigation vectors used at each step
    """
    path: List[Tuple[str, str, str]] = field(default_factory=list)
    frontier: str = ""
    visited: Set[str] = field(default_factory=set)
    score: float = 0.0
    is_stopped: bool = False
    navigation_vectors: List[torch.Tensor] = field(default_factory=list)

    def copy(self) -> "BeamPath":
        """Create a deep copy."""
        return BeamPath(
            path=list(self.path),
            frontier=self.frontier,
            visited=set(self.visited),
            score=self.score,
            is_stopped=self.is_stopped,
            navigation_vectors=list(self.navigation_vectors),
        )

    def get_answer(self) -> Optional[str]:
        """Get the answer entity (last entity in path)."""
        if self.path:
            return self.path[-1][2]
        return self.frontier


@dataclass
class InferenceResult:
    """
    Result of beam search inference.

    Attributes:
        best_answer: The predicted answer entity
        best_path: The path to the best answer
        best_score: Score of the best answer
        all_answers: Dict mapping answer entities to aggregated scores
        all_paths: List of all completed paths in the beam
        num_steps: Number of reasoning steps taken
        navigation_vectors: Navigation vectors of the best path
    """
    best_answer: Optional[str]
    best_path: List[Tuple[str, str, str]]
    best_score: float
    all_answers: Dict[str, float]
    all_paths: List[BeamPath]
    num_steps: int
    navigation_vectors: List[torch.Tensor] = field(default_factory=list)


class BeamSearchInference:
    """
    Beam search inference for SCN.

    Performs multi-hop reasoning by:
    1. Maintaining a beam of top-k paths
    2. At each step, expanding paths with valid actions
    3. Pruning to keep top-k paths by score
    4. Aggregating paths to answer entities using LogSumExp
    """

    def __init__(
        self,
        model: SCNReasoner,
        beam_size: int = 5,
        max_hops: int = 10,
        temperature: float = 1.0,
    ):
        """
        Initialize beam search.

        Args:
            model: SCN model for scoring actions
            beam_size: Number of paths to keep in beam
            max_hops: Maximum number of reasoning steps
            temperature: Temperature for action selection (inference)
        """
        self.model = model
        self.beam_size = beam_size
        self.max_hops = max_hops
        self.temperature = temperature

    @torch.no_grad()
    def search(
        self,
        question: str,
        topic_entities: List[str],
        subgraph: Set[Tuple[str, str, str]],
        use_soft_prompt: bool = True,
        entity_map: Optional[Dict[str, str]] = None,
        use_neighbor_context: bool = False,
    ) -> InferenceResult:
        """
        Perform beam search to find the best answer.

        Args:
            question: Natural language question
            topic_entities: Starting entities
            subgraph: Local knowledge graph
            use_soft_prompt: Whether to use soft tokens (SCN) or text-only
            entity_map: Optional mapping to obfuscate entity names
            use_neighbor_context: Whether to inject neighbor list into text prompt

        Returns:
            InferenceResult with best answer and path
        """
        self.model.eval()

        # Initialize beam with topic entities
        beam: List[BeamPath] = []
        for entity in topic_entities:
            beam.append(BeamPath(
                path=[],
                frontier=entity,
                visited={entity},
                score=0.0,
                is_stopped=False,
                navigation_vectors=[],
            ))

        # State manager for getting valid actions
        state_manager = StateManager(subgraph)

        # Beam search
        num_steps = 0

        for step in range(self.max_hops):
            num_steps = step + 1

            # Check if all paths are stopped
            if all(p.is_stopped for p in beam):
                break

            # Expand each path
            all_candidates: List[BeamPath] = []

            for path in beam:
                # Keep stopped paths as-is
                if path.is_stopped:
                    all_candidates.append(path)
                    continue

                # Create state for getting valid actions
                state = ReasoningState(
                    frontier=path.frontier,
                    visited=path.visited,
                    path_history=path.path,
                )

                # Get valid neighbors
                neighbors = state_manager.get_valid_actions(state)

                # If no valid neighbors, mark as stopped
                if not neighbors:
                    stopped_path = path.copy()
                    stopped_path.is_stopped = True
                    all_candidates.append(stopped_path)
                    continue

                # Construct neighbor context if requested
                context_text = None
                if use_neighbor_context and neighbors:
                    # Limit context size to avoid OOM or excessive length
                    max_context_neighbors = 20
                    context_parts = []
                    
                    # Sort by relation to be deterministic
                    sorted_neighbors = sorted(neighbors, key=lambda x: x[1])
                    
                    for i, (h, r, t) in enumerate(sorted_neighbors):
                        if i >= max_context_neighbors:
                            context_parts.append("... (more)")
                            break
                        
                        # Apply masking to tail entity
                        neighbor_name = t
                        if entity_map and t in entity_map:
                            neighbor_name = entity_map[t]
                        
                        context_parts.append(f"{r} -> {neighbor_name}")
                    
                    # Current entity masking
                    current_name = path.frontier
                    if entity_map and current_name in entity_map:
                        current_name = entity_map[current_name]
                        
                    context_text = f"Context: [{current_name}] connects to: " + ", ".join(context_parts)

                # Get navigation vector (SCN: with current entity soft prompt)
                v_nav = self.model.get_navigation_vector(
                    question=question,
                    path_history=path.path,
                    current_entity=path.frontier,
                    use_soft_prompt=use_soft_prompt,
                    entity_map=entity_map,
                    context_text=context_text,
                )
                
                # Clone v_nav for storage
                v_nav_stored = v_nav.detach().cpu()

                # Build candidate matrix
                candidate_matrix, neighbor_list = self.model.candidate_builder.build_candidate_matrix(
                    neighbors
                )

                # Compute log-probabilities
                # NOTE: Temperature is already applied in model.compute_action_logits()
                # Do NOT apply temperature again here - that was a bug causing train/inference mismatch
                log_probs = self.model.compute_action_log_probs(v_nav, candidate_matrix)

                # STOP action (index 0)
                stop_score = path.score + log_probs[0].item()
                stopped_path = path.copy()
                stopped_path.is_stopped = True
                stopped_path.score = stop_score
                # STOP action doesn't produce a NEW navigation vector for the next step,
                # but we record the one used to DECIDE to stop.
                stopped_path.navigation_vectors.append(v_nav_stored)
                all_candidates.append(stopped_path)

                # Expand to neighbors (indices 1..N)
                for i, (h, r, t) in enumerate(neighbor_list):
                    action_score = log_probs[i + 1].item()
                    new_path = BeamPath(
                        path=path.path + [(h, r, t)],
                        frontier=t,
                        visited=path.visited | {t},
                        score=path.score + action_score,
                        is_stopped=False,
                        navigation_vectors=path.navigation_vectors + [v_nav_stored],
                    )
                    all_candidates.append(new_path)

            # Prune to top-k by score
            all_candidates.sort(key=lambda p: p.score, reverse=True)
            beam = all_candidates[:self.beam_size]

        # Aggregate paths to answers using LogSumExp
        answer_scores = self._aggregate_paths(beam)

        # Get best answer
        best_path_obj = None
        if answer_scores:
            best_answer = max(answer_scores, key=answer_scores.get)
            best_score = answer_scores[best_answer]

            # Find best path to this answer
            best_path = []
            for path in beam:
                if path.get_answer() == best_answer:
                    best_path = path.path
                    best_path_obj = path
                    break
        else:
            # Fallback: use topic entity
            best_answer = topic_entities[0] if topic_entities else None
            best_score = float("-inf")
            best_path = []
            
        # Extract navigation vectors from best path
        navigation_vectors = best_path_obj.navigation_vectors if best_path_obj else []

        return InferenceResult(
            best_answer=best_answer,
            best_path=best_path,
            best_score=best_score,
            all_answers=answer_scores,
            all_paths=beam,
            num_steps=num_steps,
            navigation_vectors=navigation_vectors,
        )

    def _aggregate_paths(self, beam: List[BeamPath]) -> Dict[str, float]:
        """
        Aggregate path scores to answer entities using LogSumExp.

        Multiple paths can lead to the same answer entity.
        We aggregate their scores: score(a) = LogSumExp(scores of paths to a)

        Args:
            beam: List of completed paths

        Returns:
            Dict mapping answer entities to aggregated scores
        """
        answer_to_scores: Dict[str, List[float]] = defaultdict(list)

        for path in beam:
            answer = path.get_answer()
            if answer:
                answer_to_scores[answer].append(path.score)

        # LogSumExp aggregation
        aggregated: Dict[str, float] = {}
        for answer, scores in answer_to_scores.items():
            scores_tensor = torch.tensor(scores)
            aggregated[answer] = torch.logsumexp(scores_tensor, dim=0).item()

        return aggregated

    @torch.no_grad()
    def greedy_search(
        self,
        question: str,
        topic_entities: List[str],
        subgraph: Set[Tuple[str, str, str]],
    ) -> InferenceResult:
        """
        Greedy search (beam_size=1) for fast inference.

        Args:
            question: Natural language question
            topic_entities: Starting entities
            subgraph: Local knowledge graph

        Returns:
            InferenceResult with predicted answer
        """
        self.model.eval()

        # Start with first topic entity
        current_entity = topic_entities[0]
        path: List[Tuple[str, str, str]] = []
        visited = {current_entity}
        total_score = 0.0
        navigation_vectors = []

        state_manager = StateManager(subgraph)

        for step in range(self.max_hops):
            # Create state
            state = ReasoningState(
                frontier=current_entity,
                visited=visited,
                path_history=path,
            )

            # Get valid neighbors
            neighbors = state_manager.get_valid_actions(state)

            # No valid actions -> stop
            if not neighbors:
                break

            # Get navigation vector (SCN: with current entity soft prompt)
            v_nav = self.model.get_navigation_vector(
                question=question,
                path_history=path,
                current_entity=current_entity,
                use_soft_prompt=True,
            )
            navigation_vectors.append(v_nav.detach().cpu())

            # Build candidate matrix
            candidate_matrix, neighbor_list = self.model.candidate_builder.build_candidate_matrix(
                neighbors
            )

            # Get log-probabilities
            log_probs = self.model.compute_action_log_probs(v_nav, candidate_matrix)

            # Greedy selection
            best_idx = torch.argmax(log_probs).item()
            best_log_prob = log_probs[best_idx].item()

            # STOP selected
            if best_idx == 0:
                total_score += best_log_prob
                break

            # Take action
            selected_triple = neighbor_list[best_idx - 1]
            h, r, t = selected_triple

            path.append(selected_triple)
            visited.add(t)
            current_entity = t
            total_score += best_log_prob

        # Determine answer
        answer = path[-1][2] if path else current_entity
        
        # Create best path object for consistency
        best_path_obj = BeamPath(
            path=path,
            frontier=current_entity,
            visited=visited,
            score=total_score,
            is_stopped=True,
            navigation_vectors=navigation_vectors
        )

        return InferenceResult(
            best_answer=answer,
            best_path=path,
            best_score=total_score,
            all_answers={answer: total_score},
            all_paths=[best_path_obj],
            num_steps=len(path),
            navigation_vectors=navigation_vectors,
        )


def evaluate_sample(
    model: SCNReasoner,
    question: str,
    topic_entities: List[str],
    gold_answer: str,
    all_answers: List[str],
    subgraph: Set[Tuple[str, str, str]],
    beam_size: int = 5,
    max_hops: int = 10,
) -> Dict[str, any]:
    """
    Evaluate model on a single sample.

    Args:
        model: SCN model
        question: Natural language question
        topic_entities: Starting entities
        gold_answer: Ground truth answer
        all_answers: All valid answers
        subgraph: Local knowledge graph
        beam_size: Beam size for search
        max_hops: Maximum reasoning steps

    Returns:
        Dict with evaluation metrics
    """
    inference = BeamSearchInference(
        model=model,
        beam_size=beam_size,
        max_hops=max_hops,
    )

    result = inference.search(
        question=question,
        topic_entities=topic_entities,
        subgraph=subgraph,
    )

    # Check correctness
    predicted = result.best_answer
    is_correct = predicted in all_answers if all_answers else predicted == gold_answer

    # Check if gold answer is in beam
    gold_in_beam = gold_answer in result.all_answers

    # Get rank of gold answer
    sorted_answers = sorted(result.all_answers.items(), key=lambda x: x[1], reverse=True)
    gold_rank = -1
    for i, (ans, _) in enumerate(sorted_answers):
        if ans == gold_answer:
            gold_rank = i + 1
            break

    return {
        "predicted_answer": predicted,
        "gold_answer": gold_answer,
        "is_correct": is_correct,
        "gold_in_beam": gold_in_beam,
        "gold_rank": gold_rank,
        "predicted_score": result.best_score,
        "path_length": len(result.best_path),
        "num_steps": result.num_steps,
        "path": result.best_path,
    }
