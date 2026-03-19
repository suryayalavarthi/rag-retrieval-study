"""
End-to-end Cost-Aware RAG Pipeline.

Ties together retrieval, sufficiency scoring, routing, and generation
into a single pipeline that can be evaluated against baselines.
"""

import logging
import time
from typing import Optional

from .data_utils import QAInstance
from .generator import (
    AbstainGenerator, BaseGenerator, CostTracker, OpenAIGenerator, LocalGenerator
)
from .metrics import evaluate_predictions
from .retriever import DenseRetriever
from .router import (
    AlwaysStrongRouter, AlwaysWeakRouter, CostAwareRouter,
    QueryOnlyRouter, RandomRouter, RoutingDecision
)
from .sufficiency_scorer import SufficiencyScorer

logger = logging.getLogger(__name__)


class CostAwareRAGPipeline:
    """
    Full RAG pipeline with cost-aware routing.
    
    Flow: Query → Retrieve → Score Sufficiency → Route → Generate
    """

    def __init__(
        self,
        retriever: DenseRetriever,
        scorer: SufficiencyScorer,
        router: CostAwareRouter,
        strong_generator: BaseGenerator,
        weak_generator: BaseGenerator,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.scorer = scorer
        self.router = router
        self.strong_generator = strong_generator
        self.weak_generator = weak_generator
        self.abstain_generator = AbstainGenerator()
        self.top_k = top_k
        self.cost_tracker = CostTracker()

    def process_single(self, instance: QAInstance) -> QAInstance:
        """Process a single QA instance through the full pipeline."""

        # Step 1: Retrieve
        passages, scores = self.retriever.retrieve(instance.question, self.top_k)
        instance.retrieved_passages = passages
        instance.retrieval_scores = scores
        context = "\n\n".join(passages)

        # Step 2: Score sufficiency
        start_time = time.time()
        sufficiency_score = self.scorer.get_sufficiency_score(instance.question, context)
        scorer_latency = (time.time() - start_time) * 1000
        instance.sufficiency_score = sufficiency_score

        # Step 3: Route
        routing_result = self.router.route(sufficiency_score)
        instance.routing_decision = routing_result.decision.value

        # Step 4: Generate
        if routing_result.decision == RoutingDecision.WEAK:
            gen_result = self.weak_generator.generate(instance.question, context)
        elif routing_result.decision == RoutingDecision.STRONG:
            gen_result = self.strong_generator.generate(instance.question, context)
        else:  # ABSTAIN
            gen_result = self.abstain_generator.generate(instance.question, context)

        instance.generated_answer = gen_result.answer
        instance.model_used = gen_result.model
        self.cost_tracker.record(gen_result)

        return instance

    def process_batch(self, instances: list[QAInstance],
                       show_progress: bool = True) -> list[QAInstance]:
        """Process a batch of QA instances."""
        from tqdm import tqdm

        iterator = tqdm(instances, desc="Processing") if show_progress else instances
        results = []
        for inst in iterator:
            result = self.process_single(inst)
            results.append(result)
        return results

    def evaluate(self, instances: list[QAInstance], n_bootstrap: int = 1000) -> dict:
        """Evaluate processed instances."""
        predictions = [inst.generated_answer or "" for inst in instances]
        ground_truths = [inst.answers for inst in instances]
        costs = []  # Would need to track per-instance costs
        models_used = [inst.model_used or "unknown" for inst in instances]

        # Estimate per-instance cost from cost tracker
        for inst in instances:
            costs.append(0.0)  # Simplified; real tracking done in cost_tracker

        results = evaluate_predictions(
            predictions=predictions,
            ground_truths=ground_truths,
            costs=costs,
            models_used=models_used,
            n_bootstrap=n_bootstrap,
        )

        results["cost_tracker"] = self.cost_tracker.summary()
        results["routing_stats"] = self.router.get_stats() if hasattr(self.router, 'get_stats') else {}
        return results


def run_baseline_experiment(
    name: str,
    instances: list[QAInstance],
    retriever: DenseRetriever,
    generator: BaseGenerator,
    top_k: int = 5,
    show_progress: bool = True,
) -> tuple[list[QAInstance], dict]:
    """
    Run a baseline experiment (no routing, single model).
    """
    from tqdm import tqdm

    cost_tracker = CostTracker()
    results = []

    iterator = tqdm(instances, desc=f"Baseline: {name}") if show_progress else instances

    for inst in iterator:
        # Retrieve
        passages, scores = retriever.retrieve(inst.question, top_k)
        inst.retrieved_passages = passages
        inst.retrieval_scores = scores
        context = "\n\n".join(passages)

        # Generate
        gen_result = generator.generate(inst.question, context)
        inst.generated_answer = gen_result.answer
        inst.model_used = gen_result.model
        cost_tracker.record(gen_result)

        results.append(inst)

    # Evaluate
    predictions = [inst.generated_answer or "" for inst in results]
    ground_truths = [inst.answers for inst in results]
    models_used = [inst.model_used or "unknown" for inst in results]

    eval_results = evaluate_predictions(
        predictions=predictions,
        ground_truths=ground_truths,
        costs=[0.0] * len(results),
        models_used=models_used,
    )
    eval_results["cost_tracker"] = cost_tracker.summary()

    return results, eval_results
