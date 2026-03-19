"""
Cost-Aware Router for RAG pipelines.

Routes queries to strong model, weak model, or abstention based on
context sufficiency scores from the scorer.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    STRONG = "strong"
    WEAK = "weak"
    ABSTAIN = "abstain"


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    decision: RoutingDecision
    sufficiency_score: float
    confidence: float
    reasoning: str  # Human-readable explanation


class CostAwareRouter:
    """
    Routes queries based on context sufficiency scores.
    
    Routing logic:
      score >= high_threshold  → weak model (context is rich, cheap model suffices)
      low_threshold <= score < high_threshold → strong model (needs better reasoning)
      score < low_threshold → abstain (context is insufficient, don't hallucinate)
    """

    def __init__(
        self,
        high_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        # Cost tracking
        self._routing_stats = {
            "total": 0,
            "strong": 0,
            "weak": 0,
            "abstain": 0,
        }

    def route(self, sufficiency_score: float, confidence: float = 1.0) -> RoutingResult:
        """
        Make a routing decision based on the sufficiency score.
        
        Args:
            sufficiency_score: Float in [0, 1] from the sufficiency scorer.
            confidence: Scorer's confidence in its prediction.
            
        Returns:
            RoutingResult with decision and metadata.
        """
        self._routing_stats["total"] += 1

        if sufficiency_score >= self.high_threshold:
            decision = RoutingDecision.WEAK
            reasoning = (
                f"Sufficiency score {sufficiency_score:.3f} >= {self.high_threshold} "
                f"(high threshold). Context is sufficient for weak model."
            )
            self._routing_stats["weak"] += 1

        elif sufficiency_score >= self.low_threshold:
            decision = RoutingDecision.STRONG
            reasoning = (
                f"Sufficiency score {sufficiency_score:.3f} in "
                f"[{self.low_threshold}, {self.high_threshold}). "
                f"Context needs strong model reasoning."
            )
            self._routing_stats["strong"] += 1

        else:
            decision = RoutingDecision.ABSTAIN
            reasoning = (
                f"Sufficiency score {sufficiency_score:.3f} < {self.low_threshold} "
                f"(low threshold). Context is insufficient — abstaining."
            )
            self._routing_stats["abstain"] += 1

        return RoutingResult(
            decision=decision,
            sufficiency_score=sufficiency_score,
            confidence=confidence,
            reasoning=reasoning,
        )

    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = self._routing_stats["total"]
        if total == 0:
            return self._routing_stats

        return {
            **self._routing_stats,
            "weak_pct": self._routing_stats["weak"] / total * 100,
            "strong_pct": self._routing_stats["strong"] / total * 100,
            "abstain_pct": self._routing_stats["abstain"] / total * 100,
        }

    def reset_stats(self):
        """Reset routing statistics."""
        self._routing_stats = {"total": 0, "strong": 0, "weak": 0, "abstain": 0}

    def estimate_cost(self, num_queries: int, avg_input_tokens: int = 500,
                       avg_output_tokens: int = 100,
                       strong_input_cost: float = 0.0025,
                       strong_output_cost: float = 0.01,
                       weak_input_cost: float = 0.00015,
                       weak_output_cost: float = 0.0006) -> dict:
        """
        Estimate costs based on current routing distribution.
        
        Costs are per 1K tokens.
        """
        stats = self.get_stats()
        if stats["total"] == 0:
            return {"error": "No routing decisions made yet"}

        weak_ratio = stats["weak"] / stats["total"]
        strong_ratio = stats["strong"] / stats["total"]
        # Abstained queries have zero generation cost

        input_k = avg_input_tokens / 1000
        output_k = avg_output_tokens / 1000

        # Cost with routing
        routed_cost = num_queries * (
            weak_ratio * (input_k * weak_input_cost + output_k * weak_output_cost) +
            strong_ratio * (input_k * strong_input_cost + output_k * strong_output_cost)
        )

        # Cost without routing (all strong)
        baseline_cost = num_queries * (
            input_k * strong_input_cost + output_k * strong_output_cost
        )

        # Cost with all weak
        all_weak_cost = num_queries * (
            input_k * weak_input_cost + output_k * weak_output_cost
        )

        return {
            "routed_cost": routed_cost,
            "all_strong_cost": baseline_cost,
            "all_weak_cost": all_weak_cost,
            "cost_reduction_vs_strong": (1 - routed_cost / baseline_cost) * 100 if baseline_cost > 0 else 0,
            "routing_distribution": {
                "weak_pct": weak_ratio * 100,
                "strong_pct": strong_ratio * 100,
                "abstain_pct": (1 - weak_ratio - strong_ratio) * 100,
            }
        }


# === Baseline Routers for Comparison ===

class AlwaysStrongRouter:
    """Baseline: always route to the strong model."""
    def route(self, sufficiency_score: float, confidence: float = 1.0) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision.STRONG,
            sufficiency_score=sufficiency_score,
            confidence=confidence,
            reasoning="Always-strong baseline.",
        )


class AlwaysWeakRouter:
    """Baseline: always route to the weak model."""
    def route(self, sufficiency_score: float, confidence: float = 1.0) -> RoutingResult:
        return RoutingResult(
            decision=RoutingDecision.WEAK,
            sufficiency_score=sufficiency_score,
            confidence=confidence,
            reasoning="Always-weak baseline.",
        )


class RandomRouter:
    """Baseline: randomly route 50/50 between strong and weak."""
    def __init__(self, strong_ratio: float = 0.5, seed: int = 42):
        import random
        self.rng = random.Random(seed)
        self.strong_ratio = strong_ratio

    def route(self, sufficiency_score: float, confidence: float = 1.0) -> RoutingResult:
        if self.rng.random() < self.strong_ratio:
            decision = RoutingDecision.STRONG
        else:
            decision = RoutingDecision.WEAK
        return RoutingResult(
            decision=decision,
            sufficiency_score=sufficiency_score,
            confidence=confidence,
            reasoning=f"Random baseline (strong_ratio={self.strong_ratio}).",
        )


class QueryOnlyRouter:
    """
    Baseline: route based on query complexity only (no retrieval signals).
    
    Mimics RouteLLM-style routing using simple heuristics:
    - Longer queries → more likely to need strong model
    - Questions with comparison/reasoning keywords → strong model
    """
    COMPLEX_KEYWORDS = {
        "compare", "contrast", "difference", "why", "how", "explain",
        "analyze", "evaluate", "which is better", "pros and cons",
        "relationship between", "cause", "effect", "impact",
    }

    def __init__(self, complexity_threshold: float = 0.5, seed: int = 42):
        self.complexity_threshold = complexity_threshold

    def _estimate_complexity(self, query: str) -> float:
        """Simple heuristic for query complexity."""
        query_lower = query.lower()
        score = 0.0

        # Length signal
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.15

        # Keyword signal
        for kw in self.COMPLEX_KEYWORDS:
            if kw in query_lower:
                score += 0.2
                break

        # Multi-hop signal (multiple entities or questions)
        if query_lower.count("?") > 1:
            score += 0.2
        if " and " in query_lower and "?" in query_lower:
            score += 0.15

        return min(score, 1.0)

    def route(self, sufficiency_score: float = 0.0, confidence: float = 1.0,
              query: str = "") -> RoutingResult:
        complexity = self._estimate_complexity(query)
        if complexity >= self.complexity_threshold:
            decision = RoutingDecision.STRONG
        else:
            decision = RoutingDecision.WEAK
        return RoutingResult(
            decision=decision,
            sufficiency_score=complexity,
            confidence=confidence,
            reasoning=f"Query-only baseline (complexity={complexity:.3f}).",
        )
