"""
KGQA Evaluation Metrics Module

提供 LogSumExp + 阈值的 F1 计算，用于测试评估。
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
from collections import defaultdict


@dataclass
class F1Result:
    """F1 计算结果"""
    precision: float
    recall: float
    f1: float
    predicted_answers: Set[str]
    gold_answers: Set[str]
    true_positives: int


def compute_answer_scores_logsumexp(
    all_paths: List,
    only_stopped: bool = True,
) -> Dict[str, float]:
    """
    LogSumExp 聚合路径分数

    与 Hits@1 使用相同的聚合逻辑。
    多条路径指向同一答案时，该答案得分更高。

    Args:
        all_paths: BeamPath 对象列表
        only_stopped: 是否只使用 STOP 路径

    Returns:
        Dict[答案 -> LogSumExp 聚合分数]
    """
    answer_to_scores: Dict[str, List[float]] = defaultdict(list)

    for path in all_paths:
        # 检查是否需要过滤非 STOP 路径
        should_skip = only_stopped and not path.is_stopped
        if not should_skip:
            answer = path.get_answer()
            if answer:
                answer_to_scores[answer].append(path.score)

    # LogSumExp 聚合
    aggregated: Dict[str, float] = {}
    for answer, scores in answer_to_scores.items():
        scores_tensor = torch.tensor(scores)
        aggregated[answer] = torch.logsumexp(scores_tensor, dim=0).item()

    return aggregated


def filter_answers_by_threshold(
    answer_scores: Dict[str, float],
    threshold_delta: float = 2.0,
) -> Set[str]:
    """
    置信度阈值过滤

    只保留得分在 threshold_delta 范围内的答案。
    threshold_delta = 2.0 相当于 exp(-2) ≈ 13.5% 相对概率阈值。

    Args:
        answer_scores: Dict[答案 -> 得分]
        threshold_delta: 阈值 (log space)

    Returns:
        高置信度答案集合
    """
    if len(answer_scores) == 0:
        return set()

    best_score = max(answer_scores.values())
    threshold = best_score - threshold_delta

    filtered_answers = set()
    for answer, score in answer_scores.items():
        if score >= threshold:
            filtered_answers.add(answer)

    return filtered_answers


def compute_f1_with_logsumexp(
    all_paths: List,
    gold_answers: List[str],
    k: int = 5,
    threshold_delta: float = 2.0,
    only_stopped: bool = True,
) -> F1Result:
    """
    LogSumExp + 阈值的 F1 计算

    关键改进：
    1. LogSumExp 聚合：与 Hits@1 使用相同逻辑
    2. 置信度阈值：只保留高置信度答案
    3. Recall 归一化：使用 min(K, |gold|) 确保公平性

    步骤:
    1. LogSumExp 聚合路径分数
    2. 置信度阈值过滤
    3. 计算 Precision, Recall, F1

    Args:
        all_paths: BeamPath 对象列表
        gold_answers: 金标准答案列表
        k: Top-K 值
        threshold_delta: 置信度阈值 (默认 2.0)
        only_stopped: 是否只使用 STOP 路径

    Returns:
        F1Result
    """
    gold_answers_set = set(gold_answers)

    # Step 1: LogSumExp 聚合
    answer_scores = compute_answer_scores_logsumexp(all_paths, only_stopped)

    # Step 2: 阈值过滤
    if len(answer_scores) > 0:
        predicted_answers = filter_answers_by_threshold(answer_scores, threshold_delta)
    else:
        predicted_answers = set()

    # Step 3: 计算 P, R, F1
    num_predicted = len(predicted_answers)
    num_gold = len(gold_answers_set)

    if num_predicted == 0 and num_gold == 0:
        # 两者都为空：完美匹配
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        true_positives = 0
    elif num_predicted == 0:
        # 没有预测
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        true_positives = 0
    elif num_gold == 0:
        # 没有金标准答案
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        true_positives = 0
    else:
        # 正常情况
        true_positives = len(predicted_answers & gold_answers_set)
        precision = true_positives / num_predicted

        # Recall 归一化：denominator is min(K, |gold|)
        effective_gold_count = min(k, num_gold)
        recall = true_positives / effective_gold_count

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return F1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        predicted_answers=predicted_answers,
        gold_answers=gold_answers_set,
        true_positives=true_positives,
    )


# 默认阈值配置
DEFAULT_THRESHOLD_DELTA = 2.0  # exp(-2) ≈ 13.5% 相对概率阈值
