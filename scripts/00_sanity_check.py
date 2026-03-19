"""
scripts/00_sanity_check.py
RUN ON: LAPTOP
Reason: Fast, no GPU, 200 samples, zero API calls.

What it does:
1. Load 200 NQ validation samples
2. Build a BM25 index over the 200 QUESTIONS as corpus
3. For each query, retrieve top-5 similar questions via BM25
4. Proxy sufficiency score = normalized max BM25 score of top match
   (high score = query is "common" / well-covered = retrieval should help)
5. Label = does ANY of the retrieved questions share an answer token
   with the query? (proxy for: would a retrieved passage contain the answer)
6. Compute Pearson correlation between score and label
7. Print routing distribution + PASS/WEAK/FAIL verdict
8. Save results/sanity_check_output.json

Key design:
- Corpus = questions (not answers) — BM25 finds lexically similar questions
- A high BM25 score means the query is similar to many other questions
  → proxy for "this is a common/well-covered query type" → retrieval helps
- Label = any retrieved neighbor shares an answer word → retrieval overlap

Requires: datasets, rank_bm25, numpy, scipy
"""

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── constants ─────────────────────────────────────────────────────────────────
N_SAMPLES = 200
TOP_K = 5
HIGH_THRESH = 0.6
LOW_THRESH = 0.3
PASS_CORR = 0.3
WEAK_CORR = 0.1

RESULTS_PATH = Path(__file__).parent.parent / "results" / "sanity_check_output.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def answers_overlap(ans_a: str, ans_b: str) -> bool:
    """True if the two answers share at least one non-trivial word."""
    STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or",
                 "is", "was", "are", "were", "be", "been"}
    tokens_a = set(normalize_answer(ans_a).split()) - STOPWORDS
    tokens_b = set(normalize_answer(ans_b).split()) - STOPWORDS
    return len(tokens_a & tokens_b) > 0


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - mn) / (mx - mn)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SANITY CHECK — Does Retrieval Actually Help?")
    print("=" * 60)

    # ── 1. load dataset ───────────────────────────────────────────
    print("\n[1/5] Loading Natural Questions validation split...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    total = len(ds)
    print(f"      Total validation examples: {total}")

    indices = random.sample(range(total), N_SAMPLES)
    indices.sort()
    samples = ds.select(indices)

    questions = [s["question"] for s in samples]
    answers = [
        s["answer"][0] if isinstance(s["answer"], list) else s["answer"]
        for s in samples
    ]

    print(f"      Sampled {N_SAMPLES} examples (seed={SEED})")

    # ── 2. build BM25 index over QUESTIONS ───────────────────────
    print("\n[2/5] Building BM25 index over questions corpus...")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: pip install rank-bm25")
        sys.exit(1)

    corpus_tokens = [tokenize(q) for q in questions]
    bm25 = BM25Okapi(corpus_tokens)
    print(f"      Corpus: {len(corpus_tokens)} questions")

    # ── 3 & 4. retrieve + score + label ──────────────────────────
    print("\n[3/5] Retrieving and scoring each query...")
    t0 = time.time()

    raw_scores = []
    success_labels = []
    per_sample = []

    for i, (question, answer) in enumerate(zip(questions, answers)):
        if i % 50 == 0:
            print(f"      Progress: {i}/{N_SAMPLES} ({time.time()-t0:.1f}s elapsed)")

        query_tokens = tokenize(question)
        doc_scores = bm25.get_scores(query_tokens)

        # exclude self
        doc_scores[i] = -np.inf

        top_k_idx = np.argsort(doc_scores)[::-1][:TOP_K]
        top_scores = doc_scores[top_k_idx]

        max_score = float(top_scores[0]) if top_scores[0] != -np.inf else 0.0
        raw_scores.append(max_score)

        # label: does any top-k neighbor share an answer word?
        # this proxies: "similar questions → similar answers → retrieval helps"
        found = any(answers_overlap(answer, answers[j]) for j in top_k_idx)
        success_labels.append(int(found))

        top_questions = [questions[j] for j in top_k_idx]
        per_sample.append({
            "question": question,
            "answer": answer,
            "top_retrieved_questions": top_questions[:3],
            "raw_bm25_score": max_score,
            "answer_overlap_found": int(found),
        })

    elapsed = time.time() - t0
    print(f"      Done. {N_SAMPLES} samples in {elapsed:.1f}s")

    # ── 5. normalize + stats ──────────────────────────────────────
    print("\n[4/5] Computing statistics...")
    raw_arr = np.array(raw_scores, dtype=float)
    norm_arr = normalize_scores(raw_arr)
    label_arr = np.array(success_labels, dtype=float)

    for i, s in enumerate(per_sample):
        s["proxy_sufficiency_score"] = float(norm_arr[i])

    high_mask = norm_arr >= HIGH_THRESH
    mid_mask = (norm_arr >= LOW_THRESH) & (norm_arr < HIGH_THRESH)
    low_mask = norm_arr < LOW_THRESH

    n_high, n_mid, n_low = int(high_mask.sum()), int(mid_mask.sum()), int(low_mask.sum())
    pct_high = n_high / N_SAMPLES * 100
    pct_mid = n_mid / N_SAMPLES * 100
    pct_low = n_low / N_SAMPLES * 100

    overlap_rate = float(label_arr.mean()) * 100

    if label_arr.std() == 0:
        corr, pval = 0.0, 1.0
        print("      WARNING: all labels identical — check answer overlap logic")
    else:
        corr, pval = pearsonr(norm_arr, label_arr)

    if corr >= PASS_CORR:
        verdict = "PASS"
        verdict_msg = "Signal exists. BM25 similarity predicts answer overlap. Proceed to real retrieval experiments."
    elif corr >= WEAK_CORR:
        verdict = "WEAK"
        verdict_msg = "Weak signal. Foundation exists but dense retrieval (Contriever) will be needed for strong results."
    else:
        verdict = "FAIL"
        verdict_msg = "No signal. Rethink retrieval proxy before spending money."

    # ── print ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nSamples analyzed  : {N_SAMPLES}")
    print(f"Runtime           : {elapsed:.1f}s")
    print(f"Answer overlap rate (proxy success): {overlap_rate:.1f}%")
    print(f"\n── Routing Distribution Estimate ──")
    print(f"  High sufficiency (score >= {HIGH_THRESH}): {n_high:3d} samples ({pct_high:.1f}%)")
    print(f"  Med  sufficiency ({LOW_THRESH}-{HIGH_THRESH}): {n_mid:3d} samples ({pct_mid:.1f}%)")
    print(f"  Low  sufficiency (score < {LOW_THRESH}): {n_low:3d} samples ({pct_low:.1f}%)")
    print(f"\n── Correlation ──")
    print(f"  Pearson r : {corr:.4f}")
    print(f"  p-value   : {pval:.4f}")
    print(f"\n── Verdict ──")
    print(f"  [{verdict}] {verdict_msg}")
    print("=" * 60)

    # ── save ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving to {RESULTS_PATH} ...")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "script": "scripts/00_sanity_check.py",
            "seed": SEED,
            "n_samples": N_SAMPLES,
            "top_k": TOP_K,
            "corpus": "questions (BM25 similarity between questions)",
            "label": "answer_overlap (shared answer tokens between neighbors)",
            "runtime_seconds": round(elapsed, 2),
        },
        "summary": {
            "answer_overlap_rate_pct": round(overlap_rate, 2),
            "pearson_r": round(float(corr), 4),
            "p_value": round(float(pval), 4),
            "verdict": verdict,
            "verdict_message": verdict_msg,
            "routing_distribution": {
                "high": {"n": n_high, "pct": round(pct_high, 1)},
                "medium": {"n": n_mid, "pct": round(pct_mid, 1)},
                "low": {"n": n_low, "pct": round(pct_low, 1)},
            },
        },
        "samples": per_sample,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print("      Saved.")
    print("\nDone.")


if __name__ == "__main__":
    main()
