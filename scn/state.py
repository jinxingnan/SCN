"""
Reasoning State Management

Manages the state during multi-hop reasoning, including:
- Current frontier entities
- Visited entities (for cycle prevention)
- Path history
- Accumulated scores

"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional
from copy import deepcopy


@dataclass
class ReasoningState:
    """
    State of reasoning at a given step.

    Attributes:
        frontier: Current entity (single entity in SCN)
        visited: Set of visited entity labels (for cycle prevention)
        path_history: List of (head, relation, tail) triples traversed
        score: Accumulated log-probability
        is_stopped: Whether STOP action was selected
    """
    frontier: str
    visited: Set[str] = field(default_factory=set)
    path_history: List[Tuple[str, str, str]] = field(default_factory=list)
    score: float = 0.0
    is_stopped: bool = False

    def copy(self) -> "ReasoningState":
        """Create a deep copy of the state."""
        return ReasoningState(
            frontier=self.frontier,
            visited=set(self.visited),
            path_history=list(self.path_history),
            score=self.score,
            is_stopped=self.is_stopped,
        )

    def get_current_entity(self) -> str:
        """Get the current entity (frontier)."""
        return self.frontier

    def get_path_length(self) -> int:
        """Get the number of hops taken."""
        return len(self.path_history)

    def get_answer(self) -> Optional[str]:
        """
        Get the answer entity (last entity in path).

        Returns:
            Last entity reached, or frontier if path is empty
        """
        if self.path_history:
            return self.path_history[-1][2]  # tail of last triple
        return self.frontier


class StateManager:
    """
    Manages state transitions during reasoning.

    Handles:
    - Creating initial states from topic entities
    - Applying actions (triples) to states
    - Cycle prevention via visited set
    """

    def __init__(self, subgraph: Set[Tuple[str, str, str]]):
        """
        Initialize state manager.

        Args:
            subgraph: Set of (head, relation, tail) triples in the local subgraph
        """
        self.subgraph = subgraph

        # Build index: entity -> outgoing triples
        self.outgoing_index = {}
        for h, r, t in subgraph:
            if h not in self.outgoing_index:
                self.outgoing_index[h] = []
            self.outgoing_index[h].append((h, r, t))

    def create_initial_state(self, topic_entity: str) -> ReasoningState:
        """
        Create initial reasoning state from topic entity.

        Args:
            topic_entity: Starting entity label

        Returns:
            Initial ReasoningState
        """
        return ReasoningState(
            frontier=topic_entity,
            visited={topic_entity},
            path_history=[],
            score=0.0,
            is_stopped=False,
        )

    def create_initial_states(self, topic_entities: List[str]) -> List[ReasoningState]:
        """
        Create initial states for multiple topic entities.

        Args:
            topic_entities: List of starting entity labels

        Returns:
            List of initial ReasoningStates
        """
        return [self.create_initial_state(e) for e in topic_entities]

    def get_valid_actions(
        self,
        state: ReasoningState,
    ) -> List[Tuple[str, str, str]]:
        """
        Get valid actions (triples) from current state.

        Args:
            state: Current reasoning state

        Returns:
            List of valid (head, relation, tail) triples
            Excludes triples that would revisit entities
        """
        current_entity = state.get_current_entity()

        # Get all outgoing triples
        candidates = self.outgoing_index.get(current_entity, [])

        # Filter out triples that revisit entities (cycle prevention)
        valid = []
        for h, r, t in candidates:
            if t not in state.visited:
                valid.append((h, r, t))

        return valid

    def apply_action(
        self,
        state: ReasoningState,
        action: Tuple[str, str, str],
        log_prob: float,
    ) -> ReasoningState:
        """
        Apply an action to create a new state.

        Args:
            state: Current state
            action: (head, relation, tail) triple to apply
            log_prob: Log-probability of this action

        Returns:
            New state after applying the action
        """
        h, r, t = action

        new_state = state.copy()
        new_state.frontier = t
        new_state.visited.add(t)
        new_state.path_history.append(action)
        new_state.score += log_prob

        return new_state

    def apply_stop(
        self,
        state: ReasoningState,
        log_prob: float,
    ) -> ReasoningState:
        """
        Apply STOP action to create a terminal state.

        Args:
            state: Current state
            log_prob: Log-probability of STOP action

        Returns:
            New terminal state
        """
        new_state = state.copy()
        new_state.is_stopped = True
        new_state.score += log_prob

        return new_state

    def is_terminal(self, state: ReasoningState) -> bool:
        """
        Check if state is terminal (no more actions possible).

        A state is terminal if:
        - STOP was selected, or
        - No valid actions remain

        Args:
            state: Current state

        Returns:
            True if state is terminal
        """
        if state.is_stopped:
            return True

        valid_actions = self.get_valid_actions(state)
        return len(valid_actions) == 0


def trace_gold_path(
    topic_entity: str,
    gold_path: List[Tuple[str, str, str]],
) -> List[ReasoningState]:
    """
    Trace through gold path to get states at each step.

    Useful for teacher forcing training.

    Args:
        topic_entity: Starting entity
        gold_path: List of gold (h, r, t) triples

    Returns:
        List of states: [initial_state, state_after_step_1, ..., state_after_step_n]
    """
    states = []

    # Initial state
    state = ReasoningState(
        frontier=topic_entity,
        visited={topic_entity},
        path_history=[],
        score=0.0,
        is_stopped=False,
    )
    states.append(state.copy())

    # Apply each step
    for h, r, t in gold_path:
        state.frontier = t
        state.visited.add(t)
        state.path_history.append((h, r, t))
        states.append(state.copy())

    return states
