"""
Loss Functions for SCN Training

Implements InfoNCE contrastive loss with hard negative weighting.

Following SCN methodology:
    L = -log(exp(sim(v_nav, c⁺)/τ) / Z)

Where Z is the partition function including:
- Positive candidate c⁺
- Easy negatives (random neighbors)
- Hard negatives (spurious paths - same answer, different relation) with weight α
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss with Hard Negative Weighting and Numerical Stability.

    This loss encourages the navigation vector v_nav to be close to the
    correct candidate c⁺ while being far from incorrect candidates.

    Hard negatives (spurious paths) receive additional weight α > 1 to
    emphasize learning to distinguish semantically similar but incorrect paths.

    Loss formula:
        L = -log(exp(s⁺/τ) / [exp(s⁺/τ) + Σ exp(s⁻/τ) + α·Σ exp(s^sp/τ)])

    Where:
        s⁺ = sim(v_nav, c⁺): positive similarity
        s⁻ = sim(v_nav, c⁻): easy negative similarity
        s^sp = sim(v_nav, c^sp): hard (spurious) negative similarity
        τ: temperature
        α: hard negative weight (α > 1)

    Numerical Stability:
        Uses log-space computation to avoid exp() overflow when temperature
        is low and similarities are high.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,
    ):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature τ for softmax scaling (default: 0.07)
            hard_negative_weight: Weight α for hard negatives (default: 2.0)
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if hard_negative_weight < 1.0:
            raise ValueError(f"hard_negative_weight must be >= 1.0, got {hard_negative_weight}")

        self.temperature = temperature
        self.alpha = hard_negative_weight

    def forward(
        self,
        v_nav: torch.Tensor,
        c_positive: torch.Tensor,
        c_easy_negatives: Optional[torch.Tensor] = None,
        c_hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss with numerical stability.

        All input vectors should be L2-normalized.

        Args:
            v_nav: Navigation vector, shape [dim]
            c_positive: Positive candidate vector, shape [dim]
            c_easy_negatives: Easy negative candidates, shape [N_easy, dim] or None
            c_hard_negatives: Hard negative candidates, shape [N_hard, dim] or None

        Returns:
            Scalar loss value
        """
        # Compute raw cosine similarities (L2-normalized vectors)
        pos_sim_raw = torch.dot(v_nav, c_positive)  # scalar in [-1, 1]

        # Collect all logits for numerically stable computation
        # We'll use: loss = logsumexp(all_logits) - pos_logit
        all_logits = [pos_sim_raw / self.temperature]

        # Add easy negatives
        if c_easy_negatives is not None and c_easy_negatives.numel() > 0:
            easy_sims_raw = torch.matmul(c_easy_negatives, v_nav)  # [N_easy]
            easy_logits = easy_sims_raw / self.temperature
            all_logits.append(easy_logits)

        # Add hard negatives with weight α
        # For weighted negatives in log-space: log(α * exp(x)) = log(α) + x
        if c_hard_negatives is not None and c_hard_negatives.numel() > 0:
            hard_sims_raw = torch.matmul(c_hard_negatives, v_nav)  # [N_hard]
            # Add log(α) to hard negative logits to implement weighting in log-space
            hard_logits = hard_sims_raw / self.temperature + torch.log(torch.tensor(self.alpha, device=v_nav.device))
            all_logits.append(hard_logits)

        # Concatenate all logits
        pos_logit = all_logits[0]  # scalar
        if len(all_logits) == 1:
            # Only positive, loss = 0 (degenerate case)
            all_logits_tensor = pos_logit.unsqueeze(0)
        else:
            # Flatten pos_logit to 1D for concatenation
            all_logits_flat = [pos_logit.unsqueeze(0)]
            for logit in all_logits[1:]:
                if logit.dim() == 0:
                    all_logits_flat.append(logit.unsqueeze(0))
                else:
                    all_logits_flat.append(logit)
            all_logits_tensor = torch.cat(all_logits_flat, dim=0)

        # Numerically stable InfoNCE:
        # loss = -log(exp(pos_logit) / sum(exp(all_logits)))
        #      = -pos_logit + logsumexp(all_logits)
        log_denominator = torch.logsumexp(all_logits_tensor, dim=0)
        loss = log_denominator - pos_logit

        return loss


class MultiPositiveInfoNCELoss(nn.Module):
    """
    Multi-Positive InfoNCE Contrastive Loss with Numerical Stability.

    Extends standard InfoNCE to handle multiple positive samples.
    This is critical for multi-answer questions where several paths
    lead to correct answers.

    Loss formula:
        L = -log((Σ exp(s⁺_j/τ)) / (Σ exp(s⁺_j/τ) + Σ exp(s⁻_i/τ)))

    Where:
        s⁺_j = sim(v_nav, c⁺_j): positive similarities (multiple)
        s⁻_i = sim(v_nav, c⁻_i): negative similarities
        τ: temperature

    Numerical Stability:
        When temperature is low (e.g., 0.07-0.1) and similarities approach 1.0,
        exp(1.0/0.07) ≈ 1.6M can cause overflow. We use the standard numerical
        trick of subtracting the max value before exp():

        logsumexp(x) = max(x) + log(Σ exp(x - max(x)))

        PyTorch's torch.logsumexp already implements this, but we need to ensure
        the raw logits don't overflow before reaching logsumexp.

    Key difference from standard InfoNCE:
        - Standard: only ONE positive in numerator
        - Multi-Positive: SUM of ALL positives in numerator

    This prevents contradictory training signals when multiple
    neighbors lead to correct answers.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,
    ):
        """
        Initialize Multi-Positive InfoNCE loss.

        Args:
            temperature: Temperature τ for softmax scaling
                        Recommended: 0.1-0.2 for stability, 0.05-0.1 for sharper gradients
            hard_negative_weight: Weight α for hard negatives (default: 2.0)
        """
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if hard_negative_weight < 1.0:
            raise ValueError(f"hard_negative_weight must be >= 1.0, got {hard_negative_weight}")

        self.temperature = temperature
        self.alpha = hard_negative_weight

    def forward(
        self,
        v_nav: torch.Tensor,
        c_positives: torch.Tensor,
        c_negatives: Optional[torch.Tensor] = None,
        c_hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Multi-Positive InfoNCE loss with numerical stability.

        All input vectors should be L2-normalized.

        Args:
            v_nav: Navigation vector, shape [dim]
            c_positives: Positive candidate vectors, shape [N_pos, dim]
                        Must have at least 1 positive
            c_negatives: Easy negative candidate vectors, shape [N_neg, dim] or None
            c_hard_negatives: Hard negative candidate vectors (weighted by α),
                             shape [N_hard, dim] or None

        Returns:
            Scalar loss value
        """
        if c_positives.dim() == 1:
            # Single positive, add batch dimension
            c_positives = c_positives.unsqueeze(0)

        if c_positives.size(0) == 0:
            raise ValueError("c_positives must have at least one positive sample")

        # Compute raw cosine similarities (vectors are L2-normalized, so dot product = cosine)
        # These should be in [-1, 1] range
        pos_sims_raw = torch.matmul(c_positives, v_nav)  # [N_pos]

        # Collect all logits for numerically stable computation
        # Positive logits
        pos_logits = pos_sims_raw / self.temperature  # [N_pos]
        all_logits_list = [pos_logits]

        # Easy negative logits
        if c_negatives is not None and c_negatives.numel() > 0:
            if c_negatives.dim() == 1:
                c_negatives = c_negatives.unsqueeze(0)
            neg_sims_raw = torch.matmul(c_negatives, v_nav)  # [N_neg]
            neg_logits = neg_sims_raw / self.temperature
            all_logits_list.append(neg_logits)

        # Hard negative logits with weight α
        # For weighted negatives in log-space: log(α * exp(x)) = log(α) + x
        if c_hard_negatives is not None and c_hard_negatives.numel() > 0:
            if c_hard_negatives.dim() == 1:
                c_hard_negatives = c_hard_negatives.unsqueeze(0)
            hard_sims_raw = torch.matmul(c_hard_negatives, v_nav)  # [N_hard]
            # Add log(α) to hard negative logits to implement weighting in log-space
            hard_logits = hard_sims_raw / self.temperature + torch.log(
                torch.tensor(self.alpha, device=v_nav.device)
            )
            all_logits_list.append(hard_logits)

        # Concatenate all logits
        all_logits = torch.cat(all_logits_list, dim=0)

        # Use PyTorch's numerically stable logsumexp
        # logsumexp internally does: max(x) + log(sum(exp(x - max(x))))
        log_numerator = torch.logsumexp(pos_logits, dim=0)
        log_denominator = torch.logsumexp(all_logits, dim=0)

        # Final loss: -log(numerator / denominator) = log_denominator - log_numerator
        loss = log_denominator - log_numerator

        return loss


class StepwiseInfoNCELoss(nn.Module):
    """
    Stepwise InfoNCE loss for multi-hop reasoning.

    Computes InfoNCE loss for each step in the reasoning path,
    then aggregates (sum or mean).

    This is the main loss used during training with teacher forcing.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_weight: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize stepwise loss.

        Args:
            temperature: Temperature for InfoNCE
            hard_negative_weight: Weight for hard negatives
            reduction: How to aggregate step losses ("mean" or "sum")
        """
        super().__init__()

        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")

        self.infonce = InfoNCELoss(temperature, hard_negative_weight)
        self.reduction = reduction

    def forward(
        self,
        step_data: list,
    ) -> torch.Tensor:
        """
        Compute aggregated loss over all steps.

        Args:
            step_data: List of dicts, each containing:
                - "v_nav": Navigation vector for this step
                - "c_positive": Positive candidate vector
                - "c_easy_negatives": Optional easy negatives
                - "c_hard_negatives": Optional hard negatives

        Returns:
            Aggregated loss value
        """
        if len(step_data) == 0:
            raise ValueError("step_data cannot be empty")

        step_losses = []

        for step in step_data:
            loss = self.infonce(
                v_nav=step["v_nav"],
                c_positive=step["c_positive"],
                c_easy_negatives=step.get("c_easy_negatives"),
                c_hard_negatives=step.get("c_hard_negatives"),
            )
            step_losses.append(loss)

        # Stack and reduce
        losses = torch.stack(step_losses)

        if self.reduction == "mean":
            return losses.mean()
        else:  # sum
            return losses.sum()


class ActionLoss(nn.Module):
    """
    Standard cross-entropy loss for action selection.

    This is an alternative to InfoNCE when you want to use
    standard classification loss over all candidates.

    Loss = -log(p(a*)) where a* is the gold action
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize action loss.

        Args:
            temperature: Temperature for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        v_nav: torch.Tensor,
        candidate_matrix: torch.Tensor,
        gold_index: int,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for action selection.

        Args:
            v_nav: Navigation vector, shape [dim]
            candidate_matrix: All candidates including STOP, shape [N, dim]
            gold_index: Index of the correct action (0 = STOP)

        Returns:
            Scalar loss value
        """
        # Compute logits via cosine similarity
        logits = torch.matmul(candidate_matrix, v_nav) / self.temperature  # [N]

        # Cross-entropy loss
        target = torch.tensor(gold_index, device=v_nav.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

        return loss


def compute_step_loss(
    v_nav: torch.Tensor,
    gold_triple: tuple,
    neighbors: list,
    candidate_builder,
    temperature: float = 0.07,
    hard_negative_weight: float = 2.0,
    hard_negative_triples: Optional[list] = None,
) -> torch.Tensor:
    """
    Compute InfoNCE loss for a single reasoning step.

    This is a convenience function that handles:
    1. Building the positive candidate vector from gold_triple
    2. Classifying neighbors into easy negatives
    3. Using provided hard negatives (spurious paths)

    Args:
        v_nav: Navigation vector, shape [dim]
        gold_triple: The correct (h, r, t) triple for this step
        neighbors: List of all neighbor triples
        candidate_builder: CandidateBuilder instance for creating vectors
        temperature: InfoNCE temperature
        hard_negative_weight: Weight for hard negatives
        hard_negative_triples: Optional list of hard negative triples

    Returns:
        Scalar loss value
    """
    # Build positive candidate
    c_positive = candidate_builder.get_candidate_for_triple(gold_triple)

    # Build easy negatives (all neighbors except gold)
    easy_neg_list = []
    for triple in neighbors:
        if triple != gold_triple:
            c = candidate_builder.get_candidate_for_triple(triple)
            easy_neg_list.append(c)

    c_easy_negatives = None
    if easy_neg_list:
        c_easy_negatives = torch.stack(easy_neg_list, dim=0)

    # Build hard negatives if provided
    c_hard_negatives = None
    if hard_negative_triples:
        hard_neg_list = []
        for triple in hard_negative_triples:
            c = candidate_builder.get_candidate_for_triple(triple)
            hard_neg_list.append(c)
        c_hard_negatives = torch.stack(hard_neg_list, dim=0)

    # Compute loss
    loss_fn = InfoNCELoss(temperature, hard_negative_weight)
    return loss_fn(v_nav, c_positive, c_easy_negatives, c_hard_negatives)
