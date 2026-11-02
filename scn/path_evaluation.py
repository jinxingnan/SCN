"""
Comprehensive Path Evaluation Metrics

This module provides functions to evaluate reasoning paths from beam search:
- Core exact matching
- Four-quadrant analysis
- Top-K path matching
- Partial path matching (prefix, suffix, step-wise, edit distance)
- Confidence & score analysis
- Beam diversity
- Length analysis
"""

from typing import List, Dict, Set, Tuple


def answer_has_gold_paths(answer: str, all_gold_paths: List[List[tuple]]) -> bool:
    """
    Check if a given answer has corresponding ground_truth_paths.
    
    Args:
        answer: The answer entity to check
        all_gold_paths: List of gold paths
        
    Returns:
        True if at least one gold path ends with this answer, False otherwise
    """
    if not all_gold_paths:
        return False
    
    for gold_path in all_gold_paths:
        if not gold_path:
            continue
        # The last element of the path should be the answer entity
        last_element = gold_path[-1]
        # Handle both tuple format (head, relation, tail) and string format
        if isinstance(last_element, tuple) and len(last_element) >= 3:
            path_tail = last_element[2]  # (head, relation, tail)
        elif isinstance(last_element, str):
            path_tail = last_element
        else:
            continue
            
        if path_tail == answer:
            return True
    
    return False


def compute_edit_distance(path1: List[tuple], path2: List[tuple]) -> int:
    """Compute edit distance (Levenshtein distance) between two paths."""
    m, n = len(path1), len(path2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if path1[i-1] == path2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


def compute_path_overlap(path1: List[tuple], path2: List[tuple]) -> Dict[str, float]:
    """
    Compute prefix, suffix, and step-wise overlap between two paths.

    Returns:
        Dict with prefix_match_rate, suffix_match_rate, stepwise_accuracy
    """
    if not path1 or not path2:
        return {
            "prefix_match_rate": 0.0,
            "suffix_match_rate": 0.0,
            "stepwise_accuracy": 0.0,
        }

    # Prefix matching
    prefix_len = 0
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            prefix_len += 1
        else:
            break
    prefix_match_rate = prefix_len / max(len(path1), len(path2))

    # Suffix matching
    suffix_len = 0
    for i in range(1, min(len(path1), len(path2)) + 1):
        if path1[-i] == path2[-i]:
            suffix_len += 1
        else:
            break
    suffix_match_rate = suffix_len / max(len(path1), len(path2))

    # Step-wise accuracy (how many steps match at any position)
    matches = 0
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            matches += 1
    stepwise_accuracy = matches / max(len(path1), len(path2))

    return {
        "prefix_match_rate": prefix_match_rate,
        "suffix_match_rate": suffix_match_rate,
        "stepwise_accuracy": stepwise_accuracy,
    }


def compute_comprehensive_path_metrics(
    beam_paths: List,  # List of objects with .get_answer(), .path, .score attributes
    gold_answers: List[str],
    all_gold_paths: List[List[tuple]],
    top1_answer: str = None,  # LogSumExp-Top1 answer for dedicated evaluation
) -> Dict:
    """
    Compute comprehensive path evaluation metrics.

    This function implements ALL path metrics including:
    - Core exact matching metrics
    - Four-quadrant analysis
    - Top-K path matching (using LogSumExp-Top1)
    - Partial path matching (prefix, suffix, step-wise)
    - Edit distance analysis
    - Confidence analysis
    - Beam diversity

    Args:
        beam_paths: List of beam path objects (must have get_answer(), path, score attributes)
        gold_answers: Gold standard answers
        all_gold_paths: All valid gold paths
        top1_answer: LogSumExp-Top1 answer (optional, for better path matching evaluation)

    Returns:
        Dict with comprehensive path metrics organized by category
    """
    path_metrics = {}

    # Note: Even if gold_answers or all_gold_paths is empty, we still compute metrics
    # The metrics will naturally be 0/False through normal computation
    gold_answers_set = set(gold_answers) if gold_answers else set()

    # =================================================================
    # CATEGORY 1: CORE EXACT MATCHING METRICS (必须有)
    # =================================================================
    core_metrics = {}

    # 分析每条beam路径
    all_predictions = []  # 所有预测路径
    correct_answer_predictions = []  # 答案正确的路径
    correct_path_predictions = []  # 路径正确的路径
    both_correct_predictions = []  # 答案和路径都正确的路径
    found_gold_path_indices = set()  # 被找到的gold path索引

    for beam_path in beam_paths:
        answer = beam_path.get_answer()
        path = beam_path.path

        answer_correct = answer in gold_answers_set if answer else False
        path_correct = False
        matched_gold_idx = -1

        # Check if path matches any gold path
        # SPECIAL CASE: If answer is correct but has no corresponding ground_truth_paths,
        # default to path_correct = True (don't penalize model for incomplete dataset)
        if answer_correct and answer:
            # Check if this answer has corresponding ground_truth_paths
            has_gold_paths = answer_has_gold_paths(answer, all_gold_paths)
            
            if has_gold_paths:
                # Normal case: check if predicted path matches any gold path
                for idx, gold_path in enumerate(all_gold_paths):
                    if path == gold_path:
                        path_correct = True
                        matched_gold_idx = idx
                        found_gold_path_indices.add(idx)
                        break
            else:
                # Special case: answer is correct but no gold paths available
                # Default to path_correct = True (benefit of doubt)
                path_correct = True
                matched_gold_idx = -2  # Special marker for "no gold paths available"
        else:
            # Answer is wrong, check if path happens to be correct anyway
            for idx, gold_path in enumerate(all_gold_paths):
                if path == gold_path:
                    path_correct = True
                    matched_gold_idx = idx
                    found_gold_path_indices.add(idx)
                    break

        all_predictions.append({
            "answer": answer,
            "path": path,
            "answer_correct": answer_correct,
            "path_correct": path_correct,
            "matched_gold_idx": matched_gold_idx,
            "score": beam_path.score,
        })

        if answer_correct:
            correct_answer_predictions.append(all_predictions[-1])
        if path_correct:
            correct_path_predictions.append(all_predictions[-1])
        if answer_correct and path_correct:
            both_correct_predictions.append(all_predictions[-1])

    # 1a. Any Path Match in Beam
    core_metrics["any_path_match_in_beam"] = len(both_correct_predictions) > 0

    # 1b. Path Precision for All Predictions
    core_metrics["total_predictions"] = len(all_predictions)
    core_metrics["path_correct_count"] = len(correct_path_predictions)
    core_metrics["path_precision_all"] = (
        len(correct_path_predictions) / len(all_predictions)
        if all_predictions else 0.0
    )

    # 1c. Path Precision for Correct Answers (最重要！)
    core_metrics["answer_correct_count"] = len(correct_answer_predictions)
    core_metrics["both_correct_count"] = len(both_correct_predictions)
    core_metrics["path_precision_for_correct_answers"] = (
        len(both_correct_predictions) / len(correct_answer_predictions)
        if correct_answer_predictions else 0.0
    )

    # 1d. Path Recall (Gold Path Coverage Rate)
    core_metrics["found_gold_path_count"] = len(found_gold_path_indices)
    core_metrics["total_gold_path_count"] = len(all_gold_paths)
    core_metrics["path_recall"] = (
        len(found_gold_path_indices) / len(all_gold_paths)
        if all_gold_paths else 0.0
    )

    path_metrics["core_exact_matching"] = core_metrics

    # =================================================================
    # CATEGORY 2: FOUR-QUADRANT ANALYSIS
    # =================================================================
    quadrant_metrics = {}

    both_correct = len(both_correct_predictions)
    correct_answer_wrong_path = len(correct_answer_predictions) - both_correct
    wrong_answer_correct_path = len(correct_path_predictions) - both_correct
    both_wrong = len(all_predictions) - both_correct - correct_answer_wrong_path - wrong_answer_correct_path

    quadrant_metrics["both_correct"] = both_correct
    quadrant_metrics["correct_answer_wrong_path"] = correct_answer_wrong_path
    quadrant_metrics["wrong_answer_correct_path"] = wrong_answer_correct_path
    quadrant_metrics["both_wrong"] = both_wrong

    # 比例
    total = len(all_predictions)
    quadrant_metrics["both_correct_rate"] = both_correct / total if total > 0 else 0.0
    quadrant_metrics["correct_answer_wrong_path_rate"] = correct_answer_wrong_path / total if total > 0 else 0.0
    quadrant_metrics["wrong_answer_correct_path_rate"] = wrong_answer_correct_path / total if total > 0 else 0.0
    quadrant_metrics["both_wrong_rate"] = both_wrong / total if total > 0 else 0.0

    path_metrics["quadrant_analysis"] = quadrant_metrics

    # =================================================================
    # CATEGORY 3: TOP-K PATH MATCHING (Using LogSumExp-Top1)
    # =================================================================
    topk_metrics = {}

    # Path Match @ 1 using LogSumExp-Top1 answer
    # 找到所有指向 top1_answer 的路径，检查是否至少有一条匹配 gold path
    # SPECIAL CASE: If top1_answer is correct but has no corresponding ground_truth_paths,
    # default to path_correct = True
    if top1_answer:
        top1_paths = [
            pred for pred in all_predictions
            if pred["answer"] == top1_answer
        ]
        
        # Check if top1_answer has corresponding ground_truth_paths
        has_gold_paths = answer_has_gold_paths(top1_answer, all_gold_paths)
        
        if has_gold_paths:
            # Normal case: at least one path should be correct
            top1_path_correct = any(pred["path_correct"] for pred in top1_paths)
        else:
            # Special case: answer correct but no gold paths available
            # If answer is in gold_answers, default to path_correct = True
            if top1_answer in gold_answers_set:
                top1_path_correct = True
            else:
                top1_path_correct = False
        
        topk_metrics["path_match_at_1"] = top1_path_correct
        topk_metrics["num_paths_to_top1_answer"] = len(top1_paths)
    else:
        # Fallback: 使用 beam rank 1
        topk_paths = all_predictions[:1]
        has_match = any(p["answer_correct"] and p["path_correct"] for p in topk_paths)
        topk_metrics["path_match_at_1"] = has_match
        topk_metrics["num_paths_to_top1_answer"] = 0

    # Path Match @ 3/5 (beam-based, 保持原逻辑)
    for k in [3, 5]:
        topk_paths = all_predictions[:k]
        has_match = any(p["answer_correct"] and p["path_correct"] for p in topk_paths)
        topk_metrics[f"path_match_at_{k}"] = has_match

    # Rank of Best Correct Path (第一个完全正确路径的排名)
    best_correct_rank = -1
    for i, pred in enumerate(all_predictions):
        if pred["answer_correct"] and pred["path_correct"]:
            best_correct_rank = i + 1
            break
    topk_metrics["best_correct_path_rank"] = best_correct_rank
    topk_metrics["best_correct_path_mrr"] = 1.0 / best_correct_rank if best_correct_rank > 0 else 0.0

    path_metrics["topk_matching"] = topk_metrics

    # =================================================================
    # CATEGORY 4: PARTIAL PATH MATCHING (软匹配)
    # =================================================================
    partial_metrics = {}

    if all_predictions and all_gold_paths:
        # 对于top-1路径，计算与最近gold path的部分匹配
        best_pred_path = all_predictions[0]["path"]

        # 找到最近的gold path（编辑距离最小）
        min_edit_dist = float('inf')
        closest_gold_path = None
        best_overlap = None

        for gold_path in all_gold_paths:
            edit_dist = compute_edit_distance(best_pred_path, gold_path)
            if edit_dist < min_edit_dist:
                min_edit_dist = edit_dist
                closest_gold_path = gold_path
                best_overlap = compute_path_overlap(best_pred_path, gold_path)

        partial_metrics["min_edit_distance"] = min_edit_dist
        partial_metrics["normalized_edit_distance"] = (
            min_edit_dist / max(len(best_pred_path), len(closest_gold_path))
            if closest_gold_path else 1.0
        )

        if best_overlap:
            partial_metrics["prefix_match_rate"] = best_overlap["prefix_match_rate"]
            partial_metrics["suffix_match_rate"] = best_overlap["suffix_match_rate"]
            partial_metrics["stepwise_accuracy"] = best_overlap["stepwise_accuracy"]
        else:
            partial_metrics["prefix_match_rate"] = 0.0
            partial_metrics["suffix_match_rate"] = 0.0
            partial_metrics["stepwise_accuracy"] = 0.0

        # 对所有完全正确的预测，计算平均部分匹配
        if both_correct_predictions:
            avg_prefix = 0.0
            avg_suffix = 0.0
            avg_stepwise = 0.0

            for pred in both_correct_predictions:
                pred_path = pred["path"]
                gold_idx = pred["matched_gold_idx"]
                if gold_idx >= 0:
                    overlap = compute_path_overlap(pred_path, all_gold_paths[gold_idx])
                    avg_prefix += overlap["prefix_match_rate"]
                    avg_suffix += overlap["suffix_match_rate"]
                    avg_stepwise += overlap["stepwise_accuracy"]

            n = len(both_correct_predictions)
            partial_metrics["avg_prefix_match_for_correct"] = avg_prefix / n
            partial_metrics["avg_suffix_match_for_correct"] = avg_suffix / n
            partial_metrics["avg_stepwise_for_correct"] = avg_stepwise / n

    path_metrics["partial_matching"] = partial_metrics

    # =================================================================
    # CATEGORY 5: CONFIDENCE & SCORE ANALYSIS
    # =================================================================
    confidence_metrics = {}

    if both_correct_predictions:
        correct_scores = [p["score"] for p in both_correct_predictions]
        confidence_metrics["avg_score_correct_paths"] = sum(correct_scores) / len(correct_scores)
        confidence_metrics["max_score_correct_paths"] = max(correct_scores)
        confidence_metrics["min_score_correct_paths"] = min(correct_scores)

    wrong_predictions = [p for p in all_predictions if not (p["answer_correct"] and p["path_correct"])]
    if wrong_predictions:
        wrong_scores = [p["score"] for p in wrong_predictions]
        confidence_metrics["avg_score_wrong_paths"] = sum(wrong_scores) / len(wrong_scores)
        confidence_metrics["max_score_wrong_paths"] = max(wrong_scores)
        confidence_metrics["min_score_wrong_paths"] = min(wrong_scores)

    # Score gap (正确 vs 错误)
    if both_correct_predictions and wrong_predictions:
        confidence_metrics["score_gap"] = (
            confidence_metrics["avg_score_correct_paths"] -
            confidence_metrics["avg_score_wrong_paths"]
        )

    path_metrics["confidence_analysis"] = confidence_metrics

    # =================================================================
    # CATEGORY 6: BEAM DIVERSITY
    # =================================================================
    diversity_metrics = {}

    # Unique paths
    unique_paths = set()
    for pred in all_predictions:
        path_tuple = tuple(pred["path"])
        unique_paths.add(path_tuple)
    diversity_metrics["unique_path_count"] = len(unique_paths)
    diversity_metrics["path_diversity_rate"] = len(unique_paths) / len(all_predictions) if all_predictions else 0.0

    # Unique answers
    unique_answers = set()
    for pred in all_predictions:
        if pred["answer"]:
            unique_answers.add(pred["answer"])
    diversity_metrics["unique_answer_count"] = len(unique_answers)
    diversity_metrics["answer_diversity_rate"] = len(unique_answers) / len(all_predictions) if all_predictions else 0.0

    path_metrics["diversity"] = diversity_metrics

    # =================================================================
    # CATEGORY 7: LENGTH ANALYSIS
    # =================================================================
    length_metrics = {}

    if all_predictions:
        pred_lengths = [len(p["path"]) for p in all_predictions]
        length_metrics["avg_predicted_length"] = sum(pred_lengths) / len(pred_lengths)
        length_metrics["max_predicted_length"] = max(pred_lengths)
        length_metrics["min_predicted_length"] = min(pred_lengths)

    if all_gold_paths:
        gold_lengths = [len(p) for p in all_gold_paths]
        length_metrics["avg_gold_length"] = sum(gold_lengths) / len(gold_lengths)
        length_metrics["max_gold_length"] = max(gold_lengths)
        length_metrics["min_gold_length"] = min(gold_lengths)

    if all_predictions and all_gold_paths:
        best_pred_len = len(all_predictions[0]["path"])
        avg_gold_len = length_metrics["avg_gold_length"]
        length_metrics["length_error"] = abs(best_pred_len - avg_gold_len)

    path_metrics["length_analysis"] = length_metrics

    return path_metrics
