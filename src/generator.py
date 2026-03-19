"""
LLM Generator for RAG answer generation.

Supports OpenAI (GPT-4o, GPT-4o-mini) and local models via HuggingFace.
Tracks token usage and costs.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """Answer the following question based ONLY on the provided context. 
If the context does not contain enough information to answer the question, respond with "I don't know."
Be concise — answer in one or two sentences.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class GenerationResult:
    """Result from a single generation call."""
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float


@dataclass
class CostTracker:
    """Tracks cumulative costs across multiple generation calls."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    num_calls: int = 0
    total_latency_ms: float = 0.0
    _per_model: dict = field(default_factory=dict)

    def record(self, result: GenerationResult):
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_cost += result.cost
        self.num_calls += 1
        self.total_latency_ms += result.latency_ms

        if result.model not in self._per_model:
            self._per_model[result.model] = {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "cost": 0.0
            }
        self._per_model[result.model]["calls"] += 1
        self._per_model[result.model]["input_tokens"] += result.input_tokens
        self._per_model[result.model]["output_tokens"] += result.output_tokens
        self._per_model[result.model]["cost"] += result.cost

    def summary(self) -> dict:
        return {
            "total_calls": self.num_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_latency_ms": round(self.total_latency_ms / max(self.num_calls, 1), 1),
            "per_model": self._per_model,
        }


class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(self, question: str, context: str) -> GenerationResult:
        pass

    def generate_batch(self, questions: list[str], contexts: list[str]) -> list[GenerationResult]:
        """Default batch generation (sequential). Override for parallel."""
        return [self.generate(q, c) for q, c in zip(questions, contexts)]


class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI API (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 256,
        temperature: float = 0.0,
        cost_per_1k_input: float = 0.00015,
        cost_per_1k_output: float = 0.0006,
        api_key: str = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cost_per_1k_input = cost_per_1k_input
        self.cost_per_1k_output = cost_per_1k_output
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def generate(self, question: str, context: str) -> GenerationResult:
        """Generate an answer using OpenAI API."""
        client = self._get_client()
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        start = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            latency = (time.time() - start) * 1000

            answer = response.choices[0].message.content.strip()
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (
                input_tokens / 1000 * self.cost_per_1k_input +
                output_tokens / 1000 * self.cost_per_1k_output
            )

            return GenerationResult(
                answer=answer,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return GenerationResult(
                answer="[ERROR]",
                model=self.model,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=(time.time() - start) * 1000,
            )


class LocalGenerator(BaseGenerator):
    """Generator using a local HuggingFace model (e.g., Llama-3-8B)."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens: int = 256,
        temperature: float = 0.0,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            logger.info(f"Loading local model: {self.model_name}")
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=self.device if self.device != "cpu" else -1,
                torch_dtype="auto",
            )
            logger.info("Local model loaded")

    def generate(self, question: str, context: str) -> GenerationResult:
        """Generate an answer using a local model."""
        self._load_pipeline()
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        start = time.time()
        try:
            output = self._pipeline(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),  # pipeline requires > 0
                do_sample=self.temperature > 0,
                return_full_text=False,
            )
            latency = (time.time() - start) * 1000

            answer = output[0]["generated_text"].strip()
            # Approximate token counts
            input_tokens = len(prompt.split()) * 1.3  # rough estimate
            output_tokens = len(answer.split()) * 1.3

            return GenerationResult(
                answer=answer,
                model=self.model_name,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost=0.0,  # Local model = free
                latency_ms=latency,
            )
        except Exception as e:
            logger.error(f"Local generation error: {e}")
            return GenerationResult(
                answer="[ERROR]",
                model=self.model_name,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                latency_ms=(time.time() - start) * 1000,
            )


class AbstainGenerator:
    """Pseudo-generator that returns abstention."""

    def generate(self, question: str, context: str) -> GenerationResult:
        return GenerationResult(
            answer="I don't know.",
            model="abstain",
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency_ms=0.0,
        )
