"""
Data utilities for loading and preprocessing QA datasets.

Supports Natural Questions, HotpotQA, and MuSiQue with a unified interface.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QAInstance:
    """A single QA instance with all necessary fields."""
    id: str
    question: str
    answers: list[str]  # List of acceptable answer strings
    # Optional fields populated during pipeline execution
    retrieved_passages: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    sufficiency_score: Optional[float] = None
    routing_decision: Optional[str] = None  # "strong", "weak", or "abstain"
    generated_answer: Optional[str] = None
    model_used: Optional[str] = None
    is_correct: Optional[bool] = None
    # For sufficiency label generation
    strong_answer: Optional[str] = None
    weak_answer: Optional[str] = None
    strong_correct: Optional[bool] = None
    weak_correct: Optional[bool] = None
    sufficiency_label: Optional[int] = None  # 0=insufficient, 1=moderate, 2=sufficient
    # Supporting passages (for datasets that have them)
    gold_passages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DatasetLoader:
    """Unified loader for QA datasets."""

    @staticmethod
    def load_natural_questions(data_dir: str, split: str = "validation",
                                max_samples: Optional[int] = None) -> list[QAInstance]:
        """
        Load Natural Questions dataset.
        
        Uses the simplified version from HuggingFace datasets.
        Each instance has a question and short_answers.
        """
        try:
            from datasets import load_dataset
            ds = load_dataset("google-research-datasets/nq_open", split=split)
        except Exception as e:
            logger.warning(f"Could not load NQ from HuggingFace: {e}")
            logger.info("Attempting to load from local cache...")
            cache_path = Path(data_dir) / "nq_open" / f"{split}.jsonl"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"NQ dataset not found. Run scripts/01_download_data.py first."
                )
            ds = []
            with open(cache_path) as f:
                for line in f:
                    ds.append(json.loads(line))

        instances = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = item["question"]
            # NQ open has a list of answer strings
            answers = item["answer"] if isinstance(item["answer"], list) else [item["answer"]]

            instances.append(QAInstance(
                id=f"nq_{i}",
                question=question,
                answers=answers,
            ))

        logger.info(f"Loaded {len(instances)} Natural Questions instances")
        return instances

    @staticmethod
    def load_hotpotqa(data_dir: str, split: str = "validation",
                       max_samples: Optional[int] = None) -> list[QAInstance]:
        """
        Load HotpotQA dataset (distractor setting).
        
        Multi-hop QA where multiple passages are needed to answer.
        """
        try:
            from datasets import load_dataset
            ds = load_dataset("hotpot_qa", "distractor", split=split)
        except Exception as e:
            logger.warning(f"Could not load HotpotQA from HuggingFace: {e}")
            cache_path = Path(data_dir) / "hotpotqa" / f"{split}.jsonl"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"HotpotQA not found. Run scripts/01_download_data.py first."
                )
            ds = []
            with open(cache_path) as f:
                for line in f:
                    ds.append(json.loads(line))

        instances = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = item["question"]
            answer = item["answer"]

            # Extract gold supporting passages
            gold_passages = []
            if "context" in item:
                titles = item["context"]["title"]
                sentences = item["context"]["sentences"]
                supporting_facts_titles = set(item.get("supporting_facts", {}).get("title", []))
                for title, sents in zip(titles, sentences):
                    if title in supporting_facts_titles:
                        gold_passages.append(" ".join(sents))

            instances.append(QAInstance(
                id=f"hotpot_{item.get('id', i)}",
                question=question,
                answers=[answer],
                gold_passages=gold_passages,
            ))

        logger.info(f"Loaded {len(instances)} HotpotQA instances")
        return instances

    @staticmethod
    def load_musique(data_dir: str, split: str = "validation",
                      max_samples: Optional[int] = None) -> list[QAInstance]:
        """
        Load MuSiQue dataset (multi-hop, harder than HotpotQA).
        """
        try:
            from datasets import load_dataset
            ds = load_dataset("dwivedodaya/musique", split=split)
        except Exception as e:
            logger.warning(f"Could not load MuSiQue from HuggingFace: {e}")
            cache_path = Path(data_dir) / "musique" / f"{split}.jsonl"
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"MuSiQue not found. Run scripts/01_download_data.py first."
                )
            ds = []
            with open(cache_path) as f:
                for line in f:
                    ds.append(json.loads(line))

        instances = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            question = item["question"]
            answer = item["answer"]

            # Extract gold passages from paragraphs
            gold_passages = []
            if "paragraphs" in item:
                for para in item["paragraphs"]:
                    if para.get("is_supporting", False):
                        gold_passages.append(para.get("paragraph_text", ""))

            instances.append(QAInstance(
                id=f"musique_{item.get('id', i)}",
                question=question,
                answers=[answer] if isinstance(answer, str) else answer,
                gold_passages=gold_passages,
            ))

        logger.info(f"Loaded {len(instances)} MuSiQue instances")
        return instances

    @classmethod
    def load_dataset(cls, name: str, data_dir: str = "./data",
                      split: str = "validation",
                      max_samples: Optional[int] = None) -> list[QAInstance]:
        """Load a dataset by name."""
        loaders = {
            "natural_questions": cls.load_natural_questions,
            "hotpotqa": cls.load_hotpotqa,
            "musique": cls.load_musique,
        }
        if name not in loaders:
            raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders.keys())}")
        return loaders[name](data_dir=data_dir, split=split, max_samples=max_samples)


def save_instances(instances: list[QAInstance], path: str):
    """Save QA instances to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst.to_dict()) + "\n")
    logger.info(f"Saved {len(instances)} instances to {path}")


def load_instances(path: str) -> list[QAInstance]:
    """Load QA instances from JSONL file."""
    instances = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            instances.append(QAInstance(**data))
    logger.info(f"Loaded {len(instances)} instances from {path}")
    return instances
