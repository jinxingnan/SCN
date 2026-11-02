"""
Training Utilities for SCN

Helper functions for training with multi-positive supervision and STOP action.
"""

from typing import Set, Tuple, List


def check_triple_leads_to_answer(
    triple: Tuple[str, str, str],
    all_answers: Set[str],
) -> bool:
    """
    Check if a triple directly leads to a correct answer entity.

    This is a direct check: the tail entity must be in the answer set.

    Args:
        triple: (head, relation, tail) triple
        all_answers: Set of correct answer entity labels

    Returns:
        True if tail entity is a correct answer, False otherwise
    """
    if not triple or not all_answers:
        return False

    tail_entity = triple[2]
    return tail_entity in all_answers


def find_positive_triples(
    neighbors: List[Tuple[str, str, str]],
    all_answers: Set[str],
) -> List[Tuple[str, str, str]]:
    """
    Find all neighbor triples that lead to correct answers.

    Args:
        neighbors: List of valid neighbor triples from current entity
        all_answers: Set of correct answer entity labels

    Returns:
        List of triples where tail entity is in answer set
    """
    if not neighbors or not all_answers:
        return []

    positive_triples = []
    for triple in neighbors:
        if check_triple_leads_to_answer(triple, all_answers):
            positive_triples.append(triple)

    return positive_triples


def find_negative_triples(
    neighbors: List[Tuple[str, str, str]],
    all_answers: Set[str],
) -> List[Tuple[str, str, str]]:
    """
    Find all neighbor triples that do NOT lead to correct answers.

    Args:
        neighbors: List of valid neighbor triples from current entity
        all_answers: Set of correct answer entity labels

    Returns:
        List of triples where tail entity is NOT in answer set
    """
    if not neighbors:
        return []

    negative_triples = []
    for triple in neighbors:
        if not check_triple_leads_to_answer(triple, all_answers):
            negative_triples.append(triple)

    return negative_triples


def validate_training_step(
    gold_triple: Tuple[str, str, str],
    positive_triples: List[Tuple[str, str, str]],
    negative_triples: List[Tuple[str, str, str]],
    step_idx: int,
    path_length: int,
    is_final_step: bool = False,
) -> None:
    """
    Validate training step data to catch errors early.

    Raises ValueError if data is invalid (no silent skipping).

    Args:
        gold_triple: The gold triple for this step
        positive_triples: List of positive triples
        negative_triples: List of negative triples
        step_idx: Current step index
        path_length: Total path length
        is_final_step: Whether this is the final step (leads to answer)

    Raises:
        ValueError: If validation fails
    """
    # Must have at least one positive
    if not positive_triples:
        raise ValueError(
            f"Step {step_idx}: No positive triples found! "
            f"Gold triple: {gold_triple}"
        )

    # Gold triple should be in positives
    if gold_triple not in positive_triples:
        raise ValueError(
            f"Step {step_idx}: gold_triple {gold_triple} is not in positive_triples. "
            f"This indicates data inconsistency."
        )


def get_step_training_data(
    current_state,
    gold_triple: Tuple[str, str, str],
    neighbors: List[Tuple[str, str, str]],
    all_answers: Set[str],
    step_idx: int,
    path_length: int,
    include_stop_in_negatives: bool = True,
) -> dict:
    """
    Prepare training data for a single step.

    Multi-hop handling:
        - Intermediate steps (step_idx < path_length - 1):
          gold_triple is the ONLY positive (it's on the path to answer)
        - Final step (step_idx == path_length - 1):
          All triples leading to answers are positives (Multi-Positive)

    We NEVER train v_stop as a positive sample in teacher forcing!
    The model should learn to prefer forward actions during gold_path,
    and will naturally choose STOP when no good forward option exists.

    Args:
        current_state: Current reasoning state
        gold_triple: The gold triple for this step
        neighbors: List of valid neighbor triples
        all_answers: Set of correct answer entities
        step_idx: Current step index
        path_length: Total path length
        include_stop_in_negatives: Whether to include STOP in negatives

    Returns:
        Dict with keys:
            - "positive_triples": List of positive triples
            - "negative_triples": List of negative triples
            - "include_stop_in_negatives": Boolean

    Raises:
        ValueError: If data is invalid
    """
    is_final_step = (step_idx == path_length - 1)

    if is_final_step:
        # Final step: Multi-Positive (all triples leading to answers)
        positive_triples = find_positive_triples(neighbors, all_answers)
        negative_triples = find_negative_triples(neighbors, all_answers)

        # If no positive triples found, use gold_triple as fallback
        # (This handles edge cases where gold_path[-1] tail is not in all_answers)
        if not positive_triples and gold_triple in neighbors:
            positive_triples = [gold_triple]
            negative_triples = [t for t in neighbors if t != gold_triple]
    else:
        # Intermediate step: gold_triple is the only positive
        # (it's on the path to answer, but doesn't directly lead to one)
        positive_triples = [gold_triple]
        negative_triples = [t for t in neighbors if t != gold_triple]

    # Validate
    validate_training_step(
        gold_triple=gold_triple,
        positive_triples=positive_triples,
        negative_triples=negative_triples,
        step_idx=step_idx,
        path_length=path_length,
        is_final_step=is_final_step,
    )

    return {
        "positive_triples": positive_triples,
        "negative_triples": negative_triples,
        "include_stop_in_negatives": include_stop_in_negatives,
    }
