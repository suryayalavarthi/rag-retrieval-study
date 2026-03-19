# RUN ON: LAPTOP
# Reason: Just downloads and formats datasets, no GPU needed, fast.

"""
scripts/01_download_data.py

Downloads NQ, HotpotQA, and MuSiQue validation splits from HuggingFace
and saves them in the format required by 03_generate_labels.py.

Output format (one JSON line per query):
{
  "query_id": "nq_0",
  "question": "who wrote hamlet",
  "gold_answers": ["William Shakespeare"],
  "dataset": "nq"
}

Outputs:
  data/nq_validation.jsonl
  data/hotpotqa_validation.jsonl
  data/musique_validation.jsonl
"""

import json
from pathlib import Path
from datasets import load_dataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_nq():
    out = DATA_DIR / "nq_validation.jsonl"
    if out.exists():
        print("Skipping NQ — already exists")
        return
    print("Downloading NQ (google-research-datasets/nq_open)...")
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    with open(out, "w") as f:
        for i, item in enumerate(ds):
            record = {
                "query_id": f"nq_{i}",
                "question": item["question"],
                "gold_answers": item["answer"] if isinstance(item["answer"], list) else [item["answer"]],
                "dataset": "nq",
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {i+1} queries → {out}")


def save_hotpotqa():
    out = DATA_DIR / "hotpotqa_validation.jsonl"
    if out.exists():
        print("Skipping HotpotQA — already exists")
        return
    print("Downloading HotpotQA (hotpot_qa, distractor)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    with open(out, "w") as f:
        for i, item in enumerate(ds):
            record = {
                "query_id": f"hotpotqa_{i}",
                "question": item["question"],
                "gold_answers": [item["answer"]],
                "dataset": "hotpotqa",
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {i+1} queries → {out}")


def save_musique():
    out = DATA_DIR / "musique_validation.jsonl"
    if out.exists():
        print("Skipping MuSiQue — already exists")
        return
    print("Downloading MuSiQue...")
    try:
        ds = load_dataset("allenai/musique", split="validation")
    except Exception:
        print("  allenai/musique failed, trying fallback...")
        ds = load_dataset("musique", split="validation")
    with open(out, "w") as f:
        for i, item in enumerate(ds):
            ans = item["answer"]
            gold_answers = ans if isinstance(ans, list) else [ans]
            record = {
                "query_id": f"musique_{i}",
                "question": item["question"],
                "gold_answers": gold_answers,
                "dataset": "musique",
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {i+1} queries → {out}")


if __name__ == "__main__":
    print("=" * 50)
    print("01_download_data.py")
    print("=" * 50)
    save_nq()
    save_hotpotqa()
    save_musique()
    print("\nDone. Files saved to data/")
