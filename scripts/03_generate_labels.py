# RUN ON: KAGGLE
# Reason: Thousands of OpenAI API calls in a long-running loop requires stable environment.

"""
scripts/03_generate_labels.py

Generate HELPED/NEUTRAL/HURT labels for every query in NQ, HotpotQA, MuSiQue.
Four conditions per query:
  1. GPT-4o-mini + retrieved passage
  2. GPT-4o-mini + random irrelevant passage
  3. GPT-4o + retrieved passage
  4. GPT-4o + random irrelevant passage

Labels are based on delta_F1 = F1(mini+retrieved) - F1(mini+random).
  HELPED:  delta_F1 > 0.20
  NEUTRAL: -0.20 <= delta_F1 <= 0.20
  HURT:    delta_F1 < -0.20

Outputs:
  results/labels.jsonl       — one record per query
  results/cache/             — per-call cache (never deleted)
  results/run_log.txt        — append-only log
  results/cost_tracker.json  — updated every 100 queries
  results/failures.jsonl     — retrieval and API failures

Estimated runtime: varies by dataset size and concurrency.
"""

import os
import re
import json
import time
import string
import random
import asyncio
import hashlib
import argparse
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from openai import AsyncOpenAI, RateLimitError
from rank_bm25 import BM25Okapi

# ── Environment ───────────────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("KAGGLE_WORKING_DIR", "."))
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "cache"
LABELS_FILE = RESULTS_DIR / "labels.jsonl"
FAILURES_FILE = RESULTS_DIR / "failures.jsonl"
RUN_LOG_FILE = RESULTS_DIR / "run_log.txt"
COST_TRACKER_FILE = RESULTS_DIR / "cost_tracker.json"
INDEX_FILE = RESULTS_DIR / "faiss_index" / "index.faiss"
PASSAGES_FILE = RESULTS_DIR / "passages.jsonl"
DATA_DIR = BASE_DIR / "data"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Add it as a Kaggle Secret.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_WEAK = "gpt-4o-mini"
MODEL_STRONG = "gpt-4o"
CONTRIEVER_MODEL = "facebook/contriever-msmarco"

# Cost per 1K tokens (approximate, update if OpenAI changes pricing)
COST_PER_1K = {
    "gpt-4o-mini": {"prompt": 0.000150, "completion": 0.000600},
    "gpt-4o":      {"prompt": 0.005000, "completion": 0.015000},
}

# ── Max tokens per dataset ────────────────────────────────────────────────────
MAX_TOKENS_BY_DATASET = {
    "nq": 50,
    "hotpotqa": 100,
    "musique": 100,
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

def append_run_log(msg: str):
    with open(RUN_LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")

# ── Answer normalization (SQuAD protocol) ─────────────────────────────────────
def normalize_answer(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in {"a", "an", "the"}]
    return " ".join(tokens)

# ── F1 and EM ─────────────────────────────────────────────────────────────────
def _token_f1(pred_tokens, gold_tokens):
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_f1(prediction: str, gold_answers: list) -> float:
    pred_norm = normalize_answer(prediction)
    pred_tokens = pred_norm.split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        best = max(best, _token_f1(pred_tokens, gold_tokens))
    return best

def compute_em(prediction: str, gold_answers: list) -> int:
    pred_norm = normalize_answer(prediction)
    return int(any(pred_norm == normalize_answer(g) for g in gold_answers))

# ── Random passage sampling ───────────────────────────────────────────────────
def sample_random_passage(gold_answers: list, passages: list, rng: random.Random) -> dict:
    gold_tokens = set()
    for g in gold_answers:
        gold_tokens.update(normalize_answer(g).split())

    for _ in range(10):
        candidate = rng.choice(passages)
        passage_tokens = set(normalize_answer(candidate["text"]).split())
        if not gold_tokens & passage_tokens:
            return candidate

    # Fallback: BM25 lowest score
    corpus = [p["text"].split() for p in passages]
    bm25 = BM25Okapi(corpus)
    query_tokens = " ".join(gold_tokens).split()
    scores = bm25.get_scores(query_tokens)
    fallback_idx = int(np.argmin(scores))
    return passages[fallback_idx]

# ── Retrieval ─────────────────────────────────────────────────────────────────
_retriever_tokenizer = None
_retriever_model = None
_device = None

def _load_retriever():
    global _retriever_tokenizer, _retriever_model, _device
    if _retriever_model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _retriever_tokenizer = AutoTokenizer.from_pretrained(CONTRIEVER_MODEL)
        _retriever_model = AutoModel.from_pretrained(CONTRIEVER_MODEL).to(_device)
        _retriever_model.eval()

def _mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def _encode(texts: list) -> np.ndarray:
    _load_retriever()
    inputs = _retriever_tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
    ).to(_device)
    with torch.no_grad():
        out = _retriever_model(**inputs)
    return _mean_pool(out.last_hidden_state, inputs["attention_mask"]).cpu().float().numpy()

def retrieve_passages(question: str, index, passages: list, k: int = 5) -> list:
    try:
        q_emb = _encode([question])
        _, ids = index.search(q_emb, k)
        return [passages[i] for i in ids[0] if 0 <= i < len(passages)]
    except Exception as e:
        log.warning(f"Retrieval failed: {e}")
        return []

def log_failure(query_id: str, dataset: str, failure_type: str, detail: str):
    record = {
        "query_id": query_id,
        "dataset": dataset,
        "failure_type": failure_type,
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
    }
    with open(FAILURES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Cache ─────────────────────────────────────────────────────────────────────
def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"

def _read_cache(cache_key: str):
    p = _cache_path(cache_key)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def _write_cache(cache_key: str, data: dict):
    with open(_cache_path(cache_key), "w") as f:
        json.dump(data, f)

# ── API call ──────────────────────────────────────────────────────────────────
PROMPT_SYSTEM = (
    "You are a question answering system. "
    "Answer the question using only the provided passage. "
    "Be concise. Output only the answer, nothing else. "
    "If the passage does not contain the answer, output exactly: I don't know"
)

async def call_model_async(
    model: str,
    passage: str,
    question: str,
    max_tokens: int,
    cache_key: str,
) -> dict:
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    user_msg = f"Passage: {passage}\n\nQuestion: {question}\n\nAnswer:"

    base_delay = 5
    max_delay = 60
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                seed=SEED,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            result = {
                "answer": answer,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                },
            }
            _write_cache(cache_key, result)
            return result
        except RateLimitError:
            delay = min(base_delay * (2 ** attempt), max_delay)
            log.warning(f"RateLimitError on {model}, attempt {attempt+1}/5, sleeping {delay}s")
            await asyncio.sleep(delay)
        except Exception as e:
            log.warning(f"API error on {model}, attempt {attempt+1}/5: {e}")
            await asyncio.sleep(base_delay)

    return {"answer": "", "tokens": {"prompt": 0, "completion": 0}, "failed": True}

# ── Label assignment ──────────────────────────────────────────────────────────
def assign_label(delta_f1: float) -> str:
    if delta_f1 > 0.20:
        return "HELPED"
    elif delta_f1 < -0.20:
        return "HURT"
    return "NEUTRAL"

def compute_gap_closure(ceiling_f1, mini_random_f1, mini_retrieved_f1):
    gap_without = ceiling_f1 - mini_random_f1
    gap_with = ceiling_f1 - mini_retrieved_f1
    if gap_without == 0:
        return None, gap_without, gap_with
    return (gap_without - gap_with) / gap_without, gap_without, gap_with

# ── Cost tracking ─────────────────────────────────────────────────────────────
_cost_state = {
    "tokens": {MODEL_WEAK: {"prompt": 0, "completion": 0},
               MODEL_STRONG: {"prompt": 0, "completion": 0}},
    "queries_completed": 0,
    "queries_remaining": 0,
}

def update_cost_tracker(tokens_mini: dict, tokens_strong: dict, remaining: int):
    _cost_state["tokens"][MODEL_WEAK]["prompt"] += tokens_mini.get("prompt", 0)
    _cost_state["tokens"][MODEL_WEAK]["completion"] += tokens_mini.get("completion", 0)
    _cost_state["tokens"][MODEL_STRONG]["prompt"] += tokens_strong.get("prompt", 0)
    _cost_state["tokens"][MODEL_STRONG]["completion"] += tokens_strong.get("completion", 0)
    _cost_state["queries_completed"] += 1
    _cost_state["queries_remaining"] = remaining

def compute_cost(model: str, tokens: dict) -> float:
    rates = COST_PER_1K.get(model, {"prompt": 0, "completion": 0})
    return (tokens["prompt"] * rates["prompt"] + tokens["completion"] * rates["completion"]) / 1000

def write_cost_tracker(total_queries: int):
    mini_tok = _cost_state["tokens"][MODEL_WEAK]
    strong_tok = _cost_state["tokens"][MODEL_STRONG]
    mini_cost = compute_cost(MODEL_WEAK, mini_tok)
    strong_cost = compute_cost(MODEL_STRONG, strong_tok)
    completed = _cost_state["queries_completed"]
    remaining = _cost_state["queries_remaining"]
    avg_cost = (mini_cost + strong_cost) / max(completed, 1)
    data = {
        "tokens_by_model": {
            MODEL_WEAK: mini_tok,
            MODEL_STRONG: strong_tok,
        },
        "cost_by_model": {
            MODEL_WEAK: round(mini_cost, 4),
            MODEL_STRONG: round(strong_cost, 4),
        },
        "total_cost_so_far": round(mini_cost + strong_cost, 4),
        "queries_completed": completed,
        "queries_remaining": remaining,
        "estimated_cost_remaining": round(avg_cost * remaining, 4),
        "updated_at": datetime.now().isoformat(),
    }
    with open(COST_TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── Dry-run cost estimate ─────────────────────────────────────────────────────
def estimate_cost(queries: list) -> None:
    avg_passage_tokens = 120
    avg_question_tokens = 15
    avg_answer_tokens_mini = 15
    avg_answer_tokens_strong = 20
    system_tokens = 60
    per_query_prompt = system_tokens + avg_passage_tokens + avg_question_tokens
    n = len(queries)
    mini_prompt = n * 2 * per_query_prompt
    mini_compl = n * 2 * avg_answer_tokens_mini
    strong_prompt = n * 2 * per_query_prompt
    strong_compl = n * 2 * avg_answer_tokens_strong
    mini_cost = compute_cost(MODEL_WEAK, {"prompt": mini_prompt, "completion": mini_compl})
    strong_cost = compute_cost(MODEL_STRONG, {"prompt": strong_prompt, "completion": strong_compl})
    print(f"\n=== DRY RUN COST ESTIMATE ===")
    print(f"Queries         : {n:,}")
    print(f"GPT-4o-mini est : ${mini_cost:.2f}")
    print(f"GPT-4o est      : ${strong_cost:.2f}")
    print(f"Total est       : ${mini_cost + strong_cost:.2f}")
    print("(Actual cost will vary based on passage and answer lengths)")

# ── Per-query processing ──────────────────────────────────────────────────────
async def process_query(
    item: dict,
    index,
    passages: list,
    rng: random.Random,
    semaphore: asyncio.Semaphore,
    dry_run: bool,
) -> dict | None:
    query_id = item["query_id"]
    dataset = item["dataset"]
    question = item["question"]
    gold_answers = item["gold_answers"]
    max_tokens = MAX_TOKENS_BY_DATASET.get(dataset, 100)

    async with semaphore:
        # Retrieve passages
        retrieved = retrieve_passages(question, index, passages, k=5)
        if not retrieved:
            log_failure(query_id, dataset, "retrieval", "FAISS returned empty list")
            return None

        retrieved_passage = " ".join(p["text"] for p in retrieved[:1])  # top-1 for generation
        retrieved_ids = [p.get("id", str(i)) for i, p in enumerate(retrieved)]

        # Sample random passage
        rand_passage_obj = sample_random_passage(gold_answers, passages, rng)
        random_passage = rand_passage_obj["text"]
        random_passage_id = rand_passage_obj.get("id", "unknown")

        if dry_run:
            return None

        # Cache keys
        def ckey(model, condition):
            raw = f"{model}|{condition}|{query_id}"
            return hashlib.md5(raw.encode()).hexdigest()

        # Four conditions (sequential within query)
        mini_ret = await call_model_async(
            MODEL_WEAK, retrieved_passage, question, max_tokens,
            ckey(MODEL_WEAK, "retrieved")
        )
        mini_rand = await call_model_async(
            MODEL_WEAK, random_passage, question, max_tokens,
            ckey(MODEL_WEAK, "random")
        )
        strong_ret = await call_model_async(
            MODEL_STRONG, retrieved_passage, question, max_tokens,
            ckey(MODEL_STRONG, "retrieved")
        )
        strong_rand = await call_model_async(
            MODEL_STRONG, random_passage, question, max_tokens,
            ckey(MODEL_STRONG, "random")
        )

        # Check for API failures
        api_failures = []
        for label, result in [
            ("mini_retrieved", mini_ret),
            ("mini_random", mini_rand),
            ("gpt4o_retrieved", strong_ret),
            ("gpt4o_random", strong_rand),
        ]:
            if result.get("failed"):
                api_failures.append(label)

        if api_failures:
            log_failure(query_id, dataset, "api",
                        f"Failed conditions: {api_failures}")
            return None

        # Scoring
        mini_ret_f1 = compute_f1(mini_ret["answer"], gold_answers)
        mini_ret_em = compute_em(mini_ret["answer"], gold_answers)
        mini_rand_f1 = compute_f1(mini_rand["answer"], gold_answers)
        mini_rand_em = compute_em(mini_rand["answer"], gold_answers)
        strong_ret_f1 = compute_f1(strong_ret["answer"], gold_answers)
        strong_ret_em = compute_em(strong_ret["answer"], gold_answers)
        strong_rand_f1 = compute_f1(strong_rand["answer"], gold_answers)
        strong_rand_em = compute_em(strong_rand["answer"], gold_answers)

        delta_f1 = mini_ret_f1 - mini_rand_f1
        label = assign_label(delta_f1)
        ceiling_f1 = strong_ret_f1
        gap_closure, gap_without, gap_with = compute_gap_closure(
            ceiling_f1, mini_rand_f1, mini_ret_f1
        )

        # Token aggregation for cost tracking (both conditions per model)
        mini_tokens_total = {
            "prompt": mini_ret["tokens"]["prompt"] + mini_rand["tokens"]["prompt"],
            "completion": mini_ret["tokens"]["completion"] + mini_rand["tokens"]["completion"],
        }
        strong_tokens_total = {
            "prompt": strong_ret["tokens"]["prompt"] + strong_rand["tokens"]["prompt"],
            "completion": strong_ret["tokens"]["completion"] + strong_rand["tokens"]["completion"],
        }

        record = {
            "query_id": query_id,
            "dataset": dataset,
            "question": question,
            "gold_answers": gold_answers,
            "retrieved_passage_ids": retrieved_ids,
            "random_passage_id": random_passage_id,
            "mini_retrieved_answer": mini_ret["answer"],
            "mini_retrieved_f1": round(mini_ret_f1, 4),
            "mini_retrieved_em": mini_ret_em,
            "mini_random_answer": mini_rand["answer"],
            "mini_random_f1": round(mini_rand_f1, 4),
            "mini_random_em": mini_rand_em,
            "gpt4o_retrieved_answer": strong_ret["answer"],
            "gpt4o_retrieved_f1": round(strong_ret_f1, 4),
            "gpt4o_retrieved_em": strong_ret_em,
            "gpt4o_random_answer": strong_rand["answer"],
            "gpt4o_random_f1": round(strong_rand_f1, 4),
            "gpt4o_random_em": strong_rand_em,
            "delta_f1": round(delta_f1, 4),
            "label": label,
            "ceiling_f1": round(ceiling_f1, 4),
            "gap_without": round(gap_without, 4),
            "gap_with": round(gap_with, 4),
            "gap_closure": round(gap_closure, 4) if gap_closure is not None else None,
            "retrieval_failed": False,
            "api_failures": api_failures,
            "mini_retrieved_tokens": mini_ret["tokens"],
            "gpt4o_retrieved_tokens": strong_ret["tokens"],
        }

        return record, mini_tokens_total, strong_tokens_total

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["nq", "hotpotqa", "musique"],
                        choices=["nq", "hotpotqa", "musique"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N queries (pilot mode)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without making API calls")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already completed queries (default: on)")
    args = parser.parse_args()

    print("=" * 60)
    print("03_generate_labels.py")
    print("=" * 60)
    append_run_log(f"Session started. datasets={args.datasets} limit={args.limit} dry_run={args.dry_run}")

    # Load FAISS index and passages
    print(f"\n[1/4] Loading FAISS index from {INDEX_FILE}...")
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_FILE}. Run 02_build_index.py first.")
    index = faiss.read_index(str(INDEX_FILE))
    print(f"  Index loaded: {index.ntotal:,} vectors")

    print(f"[2/4] Loading passages from {PASSAGES_FILE}...")
    passages = []
    with open(PASSAGES_FILE) as f:
        for line in f:
            passages.append(json.loads(line))
    print(f"  Passages loaded: {len(passages):,}")

    # Load all queries
    print(f"[3/4] Loading queries...")
    all_queries = []
    for ds in args.datasets:
        path = DATA_DIR / f"{ds}_validation.jsonl"
        if not path.exists():
            log.warning(f"Data file not found: {path} — skipping {ds}")
            continue
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                item["dataset"] = ds
                all_queries.append(item)
    print(f"  Total queries loaded: {len(all_queries):,}")

    if args.limit:
        all_queries = all_queries[: args.limit]
        print(f"  Limit applied: {len(all_queries):,} queries")

    if args.dry_run:
        estimate_cost(all_queries)
        return

    # Resume: find already completed query IDs
    completed_ids = set()
    if args.resume and LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["query_id"])
                except Exception:
                    pass
    # Also check cache directory
    for cache_file in CACHE_DIR.glob("*.json"):
        pass  # cache presence checked per-call; don't pre-load all

    pending = [q for q in all_queries if q["query_id"] not in completed_ids]
    skipped = len(all_queries) - len(pending)
    print(f"[4/4] Resume: {skipped} already done, {len(pending)} remaining")
    append_run_log(f"Skipped {skipped} completed queries. Processing {len(pending)}.")

    if not pending:
        print("All queries already complete.")
        return

    rng = random.Random(SEED)
    semaphore = asyncio.Semaphore(20)
    total = len(pending)

    print(f"\nProcessing {total:,} queries with concurrency=20 ...")
    print("Cost tracker updated every 100 queries.\n")

    label_counts = {"HELPED": 0, "NEUTRAL": 0, "HURT": 0}
    processed = 0

    async def handle_query(item):
        nonlocal processed
        result = await process_query(item, index, passages, rng, semaphore, dry_run=False)
        if result is None:
            return
        record, mini_tok, strong_tok = result
        with open(LABELS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        label_counts[record["label"]] += 1
        update_cost_tracker(mini_tok, strong_tok, total - processed)
        processed += 1
        if processed % 100 == 0:
            write_cost_tracker(total)
            append_run_log(f"Progress: {processed}/{total} queries done.")

    tasks = [handle_query(item) for item in pending]
    for coro in tqdm(asyncio.as_completed(tasks), total=total, desc="Labeling"):
        await coro

    write_cost_tracker(total)
    append_run_log(f"Session complete. {processed} queries labeled.")

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Labeled : {processed:,} queries")
    print(f"  HELPED  : {label_counts['HELPED']:,}")
    print(f"  NEUTRAL : {label_counts['NEUTRAL']:,}")
    print(f"  HURT    : {label_counts['HURT']:,}")
    print(f"  Labels  : {LABELS_FILE}")
    print(f"  Costs   : {COST_TRACKER_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
