"""
Evaluation metrics for RAG systems.

Includes standard QA metrics (EM, F1), cost metrics, hallucination rate,
abstention metrics, and statistical significance testing.
"""

import logging
import re
import string
from collections import Counter

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Standard QA Metrics
# ============================================================

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)  # Remove articles
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())  # Normalize whitespace
    return text


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    """Exact match score (1.0 if prediction matches any ground truth)."""
    pred_normalized = normalize_answer(prediction)
    return float(any(
        pred_normalized == normalize_answer(gt) for gt in ground_truths
    ))


def f1_score(prediction: str, ground_truths: list[str]) -> float:
    """Token-level F1 score (max over all ground truths)."""
    pred_tokens = normalize_answer(prediction).split()

    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


# ============================================================
# Cost Metrics
# ============================================================

def cost_reduction(routed_cost: float, baseline_cost: float) -> float:
    """Percentage cost reduction compared to always-strong baseline."""
    if baseline_cost == 0:
        return 0.0
    return (1 - routed_cost / baseline_cost) * 100


# ============================================================
# Hallucination and Abstention Metrics
# ============================================================

def is_abstention(answer: str) -> bool:
    """Check if the answer is an abstention."""
    abstention_phrases = [
        "i don't know", "i do not know", "cannot answer",
        "not enough information", "insufficient context",
        "unable to answer", "no answer", "n/a",
    ]
    answer_lower = answer.lower().strip()
    return any(phrase in answer_lower for phrase in abstention_phrases)


def hallucination_rate(predictions: list[str], ground_truths: list[list[str]]) -> float:
    """
    Rate of incorrect answers (excluding abstentions).
    
    Hallucination = wrong answer that isn't an abstention.
    """
    total_non_abstain = 0
    hallucinations = 0

    for pred, gts in zip(predictions, ground_truths):
        if is_abstention(pred):
            continue
        total_non_abstain += 1
        if exact_match(pred, gts) == 0.0 and f1_score(pred, gts) < 0.3:
            hallucinations += 1

    if total_non_abstain == 0:
        return 0.0
    return hallucinations / total_non_abstain * 100


def abstention_precision(predictions: list[str], ground_truths: list[list[str]],
                           contexts_sufficient: list[bool]) -> float:
    """
    When the system abstains, how often was the context truly insufficient?
    
    High precision = system abstains only when it should.
    """
    true_positives = 0
    total_abstentions = 0

    for pred, gt, sufficient in zip(predictions, ground_truths, contexts_sufficient):
        if is_abstention(pred):
            total_abstentions += 1
            if not sufficient:
                true_positives += 1

    if total_abstentions == 0:
        return 0.0
    return true_positives / total_abstentions * 100


def abstention_recall(predictions: list[str], ground_truths: list[list[str]],
                        contexts_sufficient: list[bool]) -> float:
    """
    Of all instances where context was insufficient, how often did we abstain?
    """
    true_positives = 0
    total_insufficient = sum(1 for s in contexts_sufficient if not s)

    if total_insufficient == 0:
        return 0.0

    for pred, sufficient in zip(predictions, contexts_sufficient):
        if not sufficient and is_abstention(pred):
            true_positives += 1

    return true_positives / total_insufficient * 100


# ============================================================
# Statistical Significance
# ============================================================

def bootstrap_confidence_interval(
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Returns: (mean, lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores)
    n = len(scores)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return float(np.mean(scores)), float(lower), float(upper)


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> float:
    """
    Paired bootstrap test for statistical significance.
    
    Tests whether system A is significantly better than system B.
    Returns p-value.
    """
    rng = np.random.RandomState(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)

    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    count = 0

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample_diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        if sample_diff <= 0:
            count += 1

    return count / n_bootstrap


# ============================================================
# Aggregate Evaluation
# ============================================================

def evaluate_predictions(
    predictions: list[str],
    ground_truths: list[list[str]],
    costs: list[float],
    models_used: list[str],
    contexts_sufficient: list[bool] = None,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Compute all evaluation metrics for a set of predictions.
    
    Returns a comprehensive results dictionary.
    """
    n = len(predictions)

    # Per-instance scores
    em_scores = [exact_match(p, gt) for p, gt in zip(predictions, ground_truths)]
    f1_scores = [f1_score(p, gt) for p, gt in zip(predictions, ground_truths)]

    # Aggregate QA metrics with confidence intervals
    em_mean, em_low, em_high = bootstrap_confidence_interval(em_scores, n_bootstrap)
    f1_mean, f1_low, f1_high = bootstrap_confidence_interval(f1_scores, n_bootstrap)

    # Cost metrics
    total_cost = sum(costs)
    avg_cost = total_cost / n if n > 0 else 0

    # Routing distribution
    model_counts = {}
    for m in models_used:
        model_counts[m] = model_counts.get(m, 0) + 1

    # Abstention rate
    abstention_count = sum(1 for p in predictions if is_abstention(p))
    abstention_rate = abstention_count / n * 100 if n > 0 else 0

    # Hallucination rate
    hall_rate = hallucination_rate(predictions, ground_truths)

    results = {
        "n_instances": n,
        "exact_match": {"mean": em_mean, "ci_low": em_low, "ci_high": em_high},
        "f1": {"mean": f1_mean, "ci_low": f1_low, "ci_high": f1_high},
        "cost": {"total_usd": round(total_cost, 4), "avg_per_query_usd": round(avg_cost, 6)},
        "hallucination_rate": round(hall_rate, 2),
        "abstention_rate": round(abstention_rate, 2),
        "routing_distribution": {
            k: {"count": v, "pct": round(v / n * 100, 1)}
            for k, v in model_counts.items()
        },
    }

    # Sufficiency-aware metrics (if available)
    if contexts_sufficient is not None:
        results["abstention_precision"] = round(
            abstention_precision(predictions, ground_truths, contexts_sufficient), 2
        )
        results["abstention_recall"] = round(
            abstention_recall(predictions, ground_truths, contexts_sufficient), 2
        )

    return results
