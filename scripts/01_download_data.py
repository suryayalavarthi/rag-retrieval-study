#!/usr/bin/env python3
"""
Download datasets for Cost-Aware RAG experiments.

Downloads: Natural Questions (Open), HotpotQA, MuSiQue
Saves to ./data/ directory in JSONL format for reproducibility.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_natural_questions(data_dir: Path):
    """Download NQ Open dataset."""
    from datasets import load_dataset

    logger.info("Downloading Natural Questions (Open)...")
    ds = load_dataset("google-research-datasets/nq_open")

    for split in ["train", "validation"]:
        if split in ds:
            output_path = data_dir / "nq_open" / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for item in ds[split]:
                    f.write(json.dumps(dict(item)) + "\n")
            logger.info(f"  {split}: {len(ds[split])} examples → {output_path}")


def download_hotpotqa(data_dir: Path):
    """Download HotpotQA (distractor setting)."""
    from datasets import load_dataset

    logger.info("Downloading HotpotQA (distractor)...")
    ds = load_dataset("hotpot_qa", "distractor")

    for split in ["train", "validation"]:
        if split in ds:
            output_path = data_dir / "hotpotqa" / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for item in ds[split]:
                    f.write(json.dumps(dict(item)) + "\n")
            logger.info(f"  {split}: {len(ds[split])} examples → {output_path}")


def download_musique(data_dir: Path):
    """Download MuSiQue dataset."""
    from datasets import load_dataset

    logger.info("Downloading MuSiQue...")
    try:
        ds = load_dataset("dwivedodaya/musique")
        for split in ds:
            output_path = data_dir / "musique" / f"{split}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for item in ds[split]:
                    f.write(json.dumps(dict(item)) + "\n")
            logger.info(f"  {split}: {len(ds[split])} examples → {output_path}")
    except Exception as e:
        logger.warning(f"Could not download MuSiQue from HuggingFace: {e}")
        logger.info("Try downloading manually from: https://github.com/StonyBrookNLP/musique")


def main():
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading datasets for Cost-Aware RAG experiments")
    logger.info("=" * 60)

    download_natural_questions(data_dir)
    download_hotpotqa(data_dir)
    download_musique(data_dir)

    logger.info("\n" + "=" * 60)
    logger.info("All downloads complete!")
    logger.info(f"Data directory: {data_dir.resolve()}")

    # Print summary
    total_files = list(data_dir.rglob("*.jsonl"))
    for f in sorted(total_files):
        line_count = sum(1 for _ in open(f))
        logger.info(f"  {f.relative_to(data_dir)}: {line_count:,} examples")


if __name__ == "__main__":
    main()
