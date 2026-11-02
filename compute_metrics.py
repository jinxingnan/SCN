"""
Compute evaluation metrics from saved inference results.

This script reads beam search results from run_inference.py and computes
metrics with a specified threshold without re-running model inference.

Usage:
    python compute_metrics.py \
        --inference_file results/inference_raw.json \
        --threshold_delta 2.0 \
        --output_file results/metrics_threshold_2.0.json
"""

import argparse
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

from scn.metrics import compute_f1_with_logsumexp
from scn.path_evaluation import compute_comprehensive_path_metrics


@dataclass
class BeamPath:
    """Reconstructed BeamPath object from JSON data."""
    answer: str
    score: float
    is_stopped: bool
    path: List[tuple]
    
    def get_answer(self) -> str:
        return self.answer


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


def compute_sample_metrics(
    beam_paths: List[BeamPath],
    gold_answers: List[str],
    all_gold_paths: List[List[tuple]],
    beam_size: int,
    threshold_delta: float,
) -> Dict:
    """
    Compute metrics for a single sample.

    Args:
        beam_paths: List of BeamPath objects
        gold_answers: Gold standard answers
        all_gold_paths: All valid gold paths
        beam_size: Beam size (K)
        threshold_delta: Threshold for F1 calculation

    Returns:
        Dict of metrics for this sample
    """
    metrics = {}
    
    # ============================================================
    # STEP 1: LogSumExp aggregation for all stopped answers
    # ============================================================
    # Note: This computation is independent of gold_answers
    # Even if gold_answers is empty, we still compute predictions
    answer_scores = {}
    for path in beam_paths:
        if path.answer and path.is_stopped:
            if path.answer not in answer_scores:
                answer_scores[path.answer] = []
            answer_scores[path.answer].append(path.score)
    
    aggregated_scores = {}
    for answer, scores in answer_scores.items():
        if scores:
            aggregated_scores[answer] = torch.logsumexp(
                torch.tensor(scores), dim=0
            ).item()
    
    sorted_answers = sorted(
        aggregated_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # ============================================================
    # STEP 2: Hits@1 using LogSumExp-Top1
    # ============================================================
    top1_answer = sorted_answers[0][0] if sorted_answers else None
    metrics["hits@1"] = top1_answer in gold_answers if top1_answer else False
    
    # ============================================================
    # STEP 3: F1/Precision/Recall using LogSumExp + threshold
    # ============================================================
    f1_result = compute_f1_with_logsumexp(
        all_paths=beam_paths,
        gold_answers=gold_answers,
        k=beam_size,
        threshold_delta=threshold_delta,
        only_stopped=True,
    )
    
    metrics["precision"] = f1_result.precision
    metrics["recall"] = f1_result.recall
    metrics["f1"] = f1_result.f1
    metrics["predicted_answers_topk"] = list(f1_result.predicted_answers)
    
    # ============================================================
    # STEP 4: MRR and Hits@K using LogSumExp ranking
    # ============================================================
    gold_rank = -1
    for i, (ans, _) in enumerate(sorted_answers):
        if ans in gold_answers:
            gold_rank = i + 1
            break
    
    metrics["gold_rank"] = gold_rank

    if gold_rank != -1:
        metrics["mrr_contribution"] = 1.0 / gold_rank
        metrics["hits@3"] = gold_rank <= 3
        metrics["hits@5"] = gold_rank <= 5
        metrics["hits@10"] = gold_rank <= 10
    else:
        metrics["mrr_contribution"] = 0.0
        metrics["hits@3"] = False
        metrics["hits@5"] = False
        metrics["hits@10"] = False
    
    # ============================================================
    # STEP 5: Path evaluation for LogSumExp-Top1 answer
    # ============================================================
    # Get all paths that lead to top1_answer
    top1_paths = [
        path for path in beam_paths
        if path.answer == top1_answer and path.is_stopped
    ]
    
    # Check if any top1 path matches any gold path
    # SPECIAL CASE: If top1_answer is correct but has no corresponding ground_truth_paths,
    # default to path_correct = True (don't penalize model for incomplete dataset)
    top1_path_correct = False
    if top1_answer and top1_answer in gold_answers:
        # Check if this answer has corresponding ground_truth_paths
        has_gold_paths = answer_has_gold_paths(top1_answer, all_gold_paths)
        
        if has_gold_paths:
            # Normal case: check if any predicted path matches gold paths
            if top1_paths:
                for top1_path in top1_paths:
                    if top1_path.path in all_gold_paths:
                        top1_path_correct = True
                        break
        else:
            # Special case: answer is correct but no gold paths available
            # Default to path_correct = True (benefit of doubt)
            top1_path_correct = True
    
    metrics["top1_path_correct"] = top1_path_correct
    metrics["num_top1_paths"] = len(top1_paths)
    
    # ============================================================
    # STEP 6: Any-in-beam accuracy (auxiliary metric)
    # ============================================================
    any_correct = any(
        path.answer in gold_answers 
        for path in beam_paths 
        if path.answer
    )
    metrics["any_in_beam"] = any_correct

    # ============================================================
    # COMPREHENSIVE PATH METRICS
    # ============================================================
    path_metrics = compute_comprehensive_path_metrics(
        beam_paths=beam_paths,
        gold_answers=gold_answers,
        all_gold_paths=all_gold_paths,
        top1_answer=top1_answer,  # Pass top1_answer for better evaluation
    )
    metrics["path_metrics"] = path_metrics

    return metrics


def compute_metrics_from_inference(
    inference_data: dict,
    threshold_delta: float,
    verbose: bool = True,
) -> Dict:
    """
    Compute metrics from saved inference results.
    
    Args:
        inference_data: Inference results from run_inference.py
        threshold_delta: Threshold delta for F1 calculation
        verbose: Print progress
        
    Returns:
        Dict with aggregated metrics
    """
    metadata = inference_data["metadata"]
    samples = inference_data["samples"]
    beam_size = metadata["beam_size"]
    
    if verbose:
        print(f"Computing metrics for {len(samples)} samples...")
        print(f"Threshold delta: {threshold_delta}")
    
    # Aggregate metrics
    total_samples = 0
    samples_with_answers = 0
    
    hits_at_1_count = 0
    top1_path_correct_count = 0  # NEW: LogSumExp-Top1 path correctness
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    hits_at_3_count = 0
    hits_at_5_count = 0
    hits_at_10_count = 0
    mrr_sum = 0.0
    gold_in_beam_count = 0
    answer_any_beam_count = 0

    # Path metrics aggregation
    path_aggregators = {
        # Core exact matching
        "any_path_match_in_beam_count": 0,
        "path_precision_all_sum": 0.0,
        "path_precision_for_correct_answers_sum": 0.0,
        "path_recall_sum": 0.0,

        # Quadrant analysis
        "both_correct_sum": 0,
        "correct_answer_wrong_path_sum": 0,
        "wrong_answer_correct_path_sum": 0,
        "both_wrong_sum": 0,
        "both_correct_rate_sum": 0.0,

        # Top-K matching
        "path_match_at_1_count": 0,
        "path_match_at_3_count": 0,
        "path_match_at_5_count": 0,
        "best_correct_path_mrr_sum": 0.0,

        # Partial matching
        "prefix_match_rate_sum": 0.0,
        "suffix_match_rate_sum": 0.0,
        "stepwise_accuracy_sum": 0.0,
        "normalized_edit_distance_sum": 0.0,

        # Confidence analysis
        "avg_score_correct_paths_sum": 0.0,
        "avg_score_wrong_paths_sum": 0.0,
        "score_gap_sum": 0.0,
        "confidence_samples_count": 0,

        # Diversity
        "unique_path_count_sum": 0,
        "unique_answer_count_sum": 0,
        "path_diversity_rate_sum": 0.0,
        "answer_diversity_rate_sum": 0.0,

        # Length
        "avg_predicted_length_sum": 0.0,
        "avg_gold_length_sum": 0.0,
        "length_error_sum": 0.0,
    }

    detailed_results = []
    
    for sample in samples:
        total_samples += 1

        gold_answers = sample["gold_answers"]
        # Note: Don't skip samples with empty gold_answers
        # They should participate in computation normally
        # All metrics will naturally be 0/False through normal computation
        
        # Count samples with answers for statistics
        if gold_answers:
            samples_with_answers += 1

        # Get all valid gold paths
        all_gold_paths = []
        if "all_gold_paths" in sample and sample["all_gold_paths"]:
            all_gold_paths = [
                [tuple(triple) for triple in path]
                for path in sample["all_gold_paths"]
            ]
        elif "gold_path" in sample and sample["gold_path"]:
            # Fallback for old format
            all_gold_paths = [[tuple(triple) for triple in sample["gold_path"]]]

        # Reconstruct BeamPath objects
        beam_paths = []
        for path_info in sample["beam_paths"]:
            path_tuples = [tuple(triple) for triple in path_info["path"]]
            beam_path = BeamPath(
                answer=path_info["answer"],
                score=path_info["score"],
                is_stopped=path_info["is_stopped"],
                path=path_tuples,
            )
            beam_paths.append(beam_path)

        # Compute metrics
        sample_metrics = compute_sample_metrics(
            beam_paths=beam_paths,
            gold_answers=gold_answers,
            all_gold_paths=all_gold_paths,
            beam_size=beam_size,
            threshold_delta=threshold_delta,
        )
        
        # Aggregate
        if sample_metrics.get("hits@1", False):
            hits_at_1_count += 1
        
        if sample_metrics.get("top1_path_correct", False):
            top1_path_correct_count += 1
        
        total_precision += sample_metrics.get("precision", 0.0)
        total_recall += sample_metrics.get("recall", 0.0)
        total_f1 += sample_metrics.get("f1", 0.0)
        
        if sample_metrics.get("any_in_beam", False):
            answer_any_beam_count += 1
        
        if sample_metrics.get("gold_rank", -1) != -1:
            gold_in_beam_count += 1
            mrr_sum += sample_metrics["mrr_contribution"]

            if sample_metrics.get("hits@3", False):
                hits_at_3_count += 1
            if sample_metrics.get("hits@5", False):
                hits_at_5_count += 1
            if sample_metrics.get("hits@10", False):
                hits_at_10_count += 1

        # Aggregate comprehensive path metrics
        pm = sample_metrics.get("path_metrics", {})

        # Core exact matching
        if pm.get("core_exact_matching", {}).get("any_path_match_in_beam", False):
            path_aggregators["any_path_match_in_beam_count"] += 1
        path_aggregators["path_precision_all_sum"] += pm.get("core_exact_matching", {}).get("path_precision_all", 0.0)
        path_aggregators["path_precision_for_correct_answers_sum"] += pm.get("core_exact_matching", {}).get("path_precision_for_correct_answers", 0.0)
        path_aggregators["path_recall_sum"] += pm.get("core_exact_matching", {}).get("path_recall", 0.0)

        # Quadrant analysis
        quad = pm.get("quadrant_analysis", {})
        path_aggregators["both_correct_sum"] += quad.get("both_correct", 0)
        path_aggregators["correct_answer_wrong_path_sum"] += quad.get("correct_answer_wrong_path", 0)
        path_aggregators["wrong_answer_correct_path_sum"] += quad.get("wrong_answer_correct_path", 0)
        path_aggregators["both_wrong_sum"] += quad.get("both_wrong", 0)
        path_aggregators["both_correct_rate_sum"] += quad.get("both_correct_rate", 0.0)

        # Top-K matching
        topk = pm.get("topk_matching", {})
        if topk.get("path_match_at_1", False):
            path_aggregators["path_match_at_1_count"] += 1
        if topk.get("path_match_at_3", False):
            path_aggregators["path_match_at_3_count"] += 1
        if topk.get("path_match_at_5", False):
            path_aggregators["path_match_at_5_count"] += 1
        path_aggregators["best_correct_path_mrr_sum"] += topk.get("best_correct_path_mrr", 0.0)

        # Partial matching
        partial = pm.get("partial_matching", {})
        path_aggregators["prefix_match_rate_sum"] += partial.get("prefix_match_rate", 0.0)
        path_aggregators["suffix_match_rate_sum"] += partial.get("suffix_match_rate", 0.0)
        path_aggregators["stepwise_accuracy_sum"] += partial.get("stepwise_accuracy", 0.0)
        path_aggregators["normalized_edit_distance_sum"] += partial.get("normalized_edit_distance", 0.0)

        # Confidence analysis
        conf = pm.get("confidence_analysis", {})
        if "avg_score_correct_paths" in conf:
            path_aggregators["avg_score_correct_paths_sum"] += conf["avg_score_correct_paths"]
            path_aggregators["confidence_samples_count"] += 1
        if "avg_score_wrong_paths" in conf:
            path_aggregators["avg_score_wrong_paths_sum"] += conf["avg_score_wrong_paths"]
        if "score_gap" in conf:
            path_aggregators["score_gap_sum"] += conf["score_gap"]

        # Diversity
        div = pm.get("diversity", {})
        path_aggregators["unique_path_count_sum"] += div.get("unique_path_count", 0)
        path_aggregators["unique_answer_count_sum"] += div.get("unique_answer_count", 0)
        path_aggregators["path_diversity_rate_sum"] += div.get("path_diversity_rate", 0.0)
        path_aggregators["answer_diversity_rate_sum"] += div.get("answer_diversity_rate", 0.0)

        # Length
        length = pm.get("length_analysis", {})
        path_aggregators["avg_predicted_length_sum"] += length.get("avg_predicted_length", 0.0)
        path_aggregators["avg_gold_length_sum"] += length.get("avg_gold_length", 0.0)
        path_aggregators["length_error_sum"] += length.get("length_error", 0.0)

        # Save detailed result
        detailed_results.append({
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "gold_answers": gold_answers,
            "num_gold_paths": len(all_gold_paths),
            "metrics": sample_metrics,
        })
    
    # Compute final metrics
    # Use total_samples as denominator to include empty answer samples
    n_with_answers = max(samples_with_answers, 1)
    n_total = max(total_samples, 1)
    n_conf = max(path_aggregators["confidence_samples_count"], 1)

    final_metrics = {
        # =================================================================
        # ANSWER METRICS
        # =================================================================
        "answer_metrics": {
            # KGQA Standard Metrics (use n_total to include empty answer samples)
            "hits@1": hits_at_1_count / n_total,
            "f1": total_f1 / n_total,
            "precision": total_precision / n_total,
            "recall": total_recall / n_total,

            # Additional Answer Metrics
            "hits@3": hits_at_3_count / n_total,
            "hits@5": hits_at_5_count / n_total,
            "hits@10": hits_at_10_count / n_total,
            "mrr": mrr_sum / n_total,
            "gold_in_beam": gold_in_beam_count / n_total,
            "answer_any_beam": answer_any_beam_count / n_total,
            
            # LogSumExp-Top1 Path Correctness
            "top1_path_correct": top1_path_correct_count / n_total,
        },

        # =================================================================
        # PATH METRICS
        # =================================================================
        "path_metrics": {
            # Category 1: Core Exact Matching
            "core_exact_matching": {
                "any_path_match_in_beam": path_aggregators["any_path_match_in_beam_count"] / n_total,
                "path_precision_all": path_aggregators["path_precision_all_sum"] / n_total,
                "path_precision_for_correct_answers": path_aggregators["path_precision_for_correct_answers_sum"] / n_total,
                "path_recall": path_aggregators["path_recall_sum"] / n_total,
            },

            # Category 2: Four-Quadrant Analysis
            "quadrant_analysis": {
                "both_correct_count": path_aggregators["both_correct_sum"],
                "correct_answer_wrong_path_count": path_aggregators["correct_answer_wrong_path_sum"],
                "wrong_answer_correct_path_count": path_aggregators["wrong_answer_correct_path_sum"],
                "both_wrong_count": path_aggregators["both_wrong_sum"],
                "both_correct_rate": path_aggregators["both_correct_rate_sum"] / n_total,
            },

            # Category 3: Top-K Path Matching
            "topk_matching": {
                "path_match@1": path_aggregators["path_match_at_1_count"] / n_total,
                "path_match@3": path_aggregators["path_match_at_3_count"] / n_total,
                "path_match@5": path_aggregators["path_match_at_5_count"] / n_total,
                "best_correct_path_mrr": path_aggregators["best_correct_path_mrr_sum"] / n_total,
            },

            # Category 4: Partial Path Matching
            "partial_matching": {
                "avg_prefix_match_rate": path_aggregators["prefix_match_rate_sum"] / n_total,
                "avg_suffix_match_rate": path_aggregators["suffix_match_rate_sum"] / n_total,
                "avg_stepwise_accuracy": path_aggregators["stepwise_accuracy_sum"] / n_total,
                "avg_normalized_edit_distance": path_aggregators["normalized_edit_distance_sum"] / n_total,
            },

            # Category 5: Confidence Analysis
            "confidence_analysis": {
                "avg_score_correct_paths": path_aggregators["avg_score_correct_paths_sum"] / n_conf,
                "avg_score_wrong_paths": path_aggregators["avg_score_wrong_paths_sum"] / n_total,
                "avg_score_gap": path_aggregators["score_gap_sum"] / n_conf,
            },

            # Category 6: Beam Diversity
            "diversity": {
                "avg_unique_path_count": path_aggregators["unique_path_count_sum"] / n_total,
                "avg_unique_answer_count": path_aggregators["unique_answer_count_sum"] / n_total,
                "avg_path_diversity_rate": path_aggregators["path_diversity_rate_sum"] / n_total,
                "avg_answer_diversity_rate": path_aggregators["answer_diversity_rate_sum"] / n_total,
            },

            # Category 7: Length Analysis
            "length_analysis": {
                "avg_predicted_length": path_aggregators["avg_predicted_length_sum"] / n_total,
                "avg_gold_length": path_aggregators["avg_gold_length_sum"] / n_total,
                "avg_length_error": path_aggregators["length_error_sum"] / n_total,
            },
        },

        # Sample Counts
        "num_samples": total_samples,
        "num_samples_with_answers": samples_with_answers,
    }
    
    return {
        "config": {
            "threshold_delta": threshold_delta,
            "beam_size": beam_size,
            "source_inference_file": metadata.get("checkpoint_dir", "unknown"),
        },
        "metrics": final_metrics,
        "detailed_results": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics from saved inference results"
    )
    
    parser.add_argument(
        "--inference_file",
        type=str,
        required=True,
        help="Path to inference results JSON from run_inference.py",
    )
    parser.add_argument(
        "--threshold_delta",
        type=float,
        default=2.0,
        help="Confidence threshold delta for F1 calculation",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file for metrics results (JSON)",
    )
    
    args = parser.parse_args()
    
    # Load inference results
    print(f"Loading inference results from {args.inference_file}...")
    with open(args.inference_file, "r", encoding="utf-8") as f:
        inference_data = json.load(f)
    
    print(f"Loaded {len(inference_data['samples'])} samples")
    
    # Compute metrics
    print(f"\n{'='*60}")
    results = compute_metrics_from_inference(
        inference_data=inference_data,
        threshold_delta=args.threshold_delta,
        verbose=True,
    )
    print(f"{'='*60}\n")
    
    # Print metrics
    metrics = results["metrics"]
    answer_m = metrics["answer_metrics"]
    path_m = metrics["path_metrics"]

    print("=" * 80)
    print(" " * 25 + "EVALUATION RESULTS")
    print("=" * 80)

    # ========== ANSWER METRICS ==========
    print(f"\n{'='*80}")
    print(" " * 30 + "ANSWER METRICS")
    print(f"{'='*80}")

    print(f"\n[KGQA Standard Metrics]")
    print(f"  Hits@1 (Answer):       {answer_m['hits@1']:.4f} ({answer_m['hits@1']*100:.2f}%)")
    print(f"  Top1 Path Correct:     {answer_m['top1_path_correct']:.4f} ({answer_m['top1_path_correct']*100:.2f}%)")
    print(f"  F1:                    {answer_m['f1']:.4f}")
    print(f"  Precision:             {answer_m['precision']:.4f}")
    print(f"  Recall:                {answer_m['recall']:.4f}")

    print(f"\n[Additional Answer Metrics]")
    print(f"  Hits@3:         {answer_m['hits@3']:.4f} ({answer_m['hits@3']*100:.2f}%)")
    print(f"  Hits@5:         {answer_m['hits@5']:.4f} ({answer_m['hits@5']*100:.2f}%)")
    print(f"  Hits@10:        {answer_m['hits@10']:.4f} ({answer_m['hits@10']*100:.2f}%)")
    print(f"  MRR:            {answer_m['mrr']:.4f}")
    print(f"  Gold in beam:   {answer_m['gold_in_beam']:.4f} ({answer_m['gold_in_beam']*100:.2f}%)")
    print(f"  Any in beam:    {answer_m['answer_any_beam']:.4f} ({answer_m['answer_any_beam']*100:.2f}%)")

    # ========== PATH METRICS ==========
    print(f"\n{'='*80}")
    print(" " * 31 + "PATH METRICS")
    print(f"{'='*80}")

    # Core Exact Matching
    core = path_m["core_exact_matching"]
    print(f"\n[1. Core Exact Matching] (必须有)")
    print(f"  Any Path Match in Beam:                {core['any_path_match_in_beam']:.4f} ({core['any_path_match_in_beam']*100:.2f}%)")
    print(f"  Path Precision (All Predictions):      {core['path_precision_all']:.4f}")
    print(f"  Path Precision (Correct Answers):      {core['path_precision_for_correct_answers']:.4f}")
    print(f"  Path Recall (Gold Coverage):           {core['path_recall']:.4f}")

    # Quadrant Analysis
    quad = path_m["quadrant_analysis"]
    print(f"\n[2. Four-Quadrant Analysis]")
    print(f"  Both Correct (Answer✓ Path✓):         {quad['both_correct_count']:5d}  ({quad['both_correct_rate']*100:5.2f}%)")
    print(f"  Correct Answer, Wrong Path (Answer✓):  {quad['correct_answer_wrong_path_count']:5d}")
    print(f"  Wrong Answer, Correct Path (Path✓):    {quad['wrong_answer_correct_path_count']:5d}")
    print(f"  Both Wrong (Answer✗ Path✗):           {quad['both_wrong_count']:5d}")

    # Top-K Matching
    topk = path_m["topk_matching"]
    print(f"\n[3. Top-K Path Matching]")
    print(f"  Path Match @ 1:                        {topk['path_match@1']:.4f} ({topk['path_match@1']*100:.2f}%)")
    print(f"  Path Match @ 3:                        {topk['path_match@3']:.4f} ({topk['path_match@3']*100:.2f}%)")
    print(f"  Path Match @ 5:                        {topk['path_match@5']:.4f} ({topk['path_match@5']*100:.2f}%)")
    print(f"  Best Correct Path MRR:                 {topk['best_correct_path_mrr']:.4f}")

    # Partial Matching
    partial = path_m["partial_matching"]
    print(f"\n[4. Partial Path Matching] (Soft Matching)")
    print(f"  Avg Prefix Match Rate:                 {partial['avg_prefix_match_rate']:.4f}")
    print(f"  Avg Suffix Match Rate:                 {partial['avg_suffix_match_rate']:.4f}")
    print(f"  Avg Step-wise Accuracy:                {partial['avg_stepwise_accuracy']:.4f}")
    print(f"  Avg Normalized Edit Distance:          {partial['avg_normalized_edit_distance']:.4f}")

    # Confidence Analysis
    conf = path_m["confidence_analysis"]
    print(f"\n[5. Confidence & Score Analysis]")
    print(f"  Avg Score (Correct Paths):             {conf['avg_score_correct_paths']:.4f}")
    print(f"  Avg Score (Wrong Paths):               {conf['avg_score_wrong_paths']:.4f}")
    print(f"  Score Gap (Correct - Wrong):           {conf['avg_score_gap']:.4f}")

    # Diversity
    div = path_m["diversity"]
    print(f"\n[6. Beam Diversity]")
    print(f"  Avg Unique Path Count:                 {div['avg_unique_path_count']:.2f}")
    print(f"  Avg Unique Answer Count:               {div['avg_unique_answer_count']:.2f}")
    print(f"  Avg Path Diversity Rate:               {div['avg_path_diversity_rate']:.4f}")
    print(f"  Avg Answer Diversity Rate:             {div['avg_answer_diversity_rate']:.4f}")

    # Length Analysis
    length = path_m["length_analysis"]
    print(f"\n[7. Length Analysis]")
    print(f"  Avg Predicted Path Length:             {length['avg_predicted_length']:.2f}")
    print(f"  Avg Gold Path Length:                  {length['avg_gold_length']:.2f}")
    print(f"  Avg Length Error:                      {length['avg_length_error']:.2f}")

    # Sample Counts
    print(f"\n{'='*80}")
    print(" " * 30 + "SAMPLE COUNTS")
    print(f"{'='*80}")
    print(f"  Total samples:                         {metrics['num_samples']}")
    print(f"  Samples with answers:                  {metrics['num_samples_with_answers']}")

    print("=" * 80)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
