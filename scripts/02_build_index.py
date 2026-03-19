# RUN ON: KAGGLE
# Reason: Encoding 100k Wikipedia passages with Contriever requires GPU and significant memory.

"""
scripts/02_build_index.py

Builds a FAISS retrieval index using Contriever (facebook/contriever-msmarco)
embeddings over 100,000 Wikipedia passages.

Outputs:
  BASE_DIR/results/faiss_index/index.faiss  — FAISS flat L2 index
  BASE_DIR/results/passages.jsonl           — passage texts (one JSON per line)

Run on Kaggle T4. Estimated runtime: 60-90 minutes on Kaggle T4.
"""

import os
import json
import time
import random
import numpy as np
import faiss
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ── Environment ──────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("KAGGLE_WORKING_DIR", ".")
RESULTS_DIR = Path(BASE_DIR) / "results"
INDEX_DIR = RESULTS_DIR / "faiss_index"
PASSAGES_FILE = RESULTS_DIR / "passages.jsonl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/contriever-msmarco"
NUM_PASSAGES = 100_000
BATCH_SIZE = 64
SEED = 42
PROGRESS_EVERY = 10_000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("02_build_index.py — Build Contriever + FAISS Index")
print("=" * 60)
print(f"Estimated runtime: 60-90 minutes on Kaggle T4")
print(f"Model         : {MODEL_NAME}")
print(f"Passages      : {NUM_PASSAGES:,}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Output dir    : {RESULTS_DIR}")
print("=" * 60)

# ── Step 1: Load Wikipedia passages ──────────────────────────────────────────
print("\n[1/4] Loading Wikipedia passages...")
t0 = time.time()

wiki = load_dataset(
    "wikipedia",
    "20220301.en",
    split="train",
    trust_remote_code=True,
)

# Sample NUM_PASSAGES articles deterministically
rng = random.Random(SEED)
indices = rng.sample(range(len(wiki)), min(NUM_PASSAGES, len(wiki)))
sampled = wiki.select(indices)

# Split each article into the first ~100-word passage chunk
passages = []
for item in sampled:
    text = item["text"].strip()
    # Take first 100 words as the passage
    words = text.split()
    passage = " ".join(words[:100])
    if passage:
        passages.append({
            "id": item["id"],
            "title": item["title"],
            "text": passage,
        })
    if len(passages) >= NUM_PASSAGES:
        break

print(f"  Loaded {len(passages):,} passages in {time.time()-t0:.1f}s")

# ── Step 2: Save passages.jsonl ───────────────────────────────────────────────
print(f"\n[2/4] Saving passages to {PASSAGES_FILE}...")
with open(PASSAGES_FILE, "w", encoding="utf-8") as f:
    for p in passages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"  Saved {len(passages):,} passages.")

# ── Step 3: Encode passages with Contriever ───────────────────────────────────
print(f"\n[3/4] Encoding passages with {MODEL_NAME}...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def mean_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).float()
    return (token_embeddings * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)


all_embeddings = []
t_encode = time.time()

for start in range(0, len(passages), BATCH_SIZE):
    batch_texts = [p["text"] for p in passages[start: start + BATCH_SIZE]]
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    all_embeddings.append(embeddings.cpu().float().numpy())

    done = min(start + BATCH_SIZE, len(passages))
    if done % PROGRESS_EVERY == 0 or done == len(passages):
        elapsed = time.time() - t_encode
        print(f"  Progress: {done:,}/{len(passages):,} passages encoded ({elapsed:.1f}s elapsed)")

embeddings_matrix = np.vstack(all_embeddings)
print(f"  Encoding complete. Shape: {embeddings_matrix.shape}  Time: {time.time()-t_encode:.1f}s")

# ── Step 4: Build and save FAISS index ───────────────────────────────────────
print(f"\n[4/4] Building FAISS flat L2 index...")
t_faiss = time.time()

dim = embeddings_matrix.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_matrix)

index_path = str(INDEX_DIR / "index.faiss")
faiss.write_index(index, index_path)

index_size_mb = os.path.getsize(index_path) / (1024 ** 2)
print(f"  Index built in {time.time()-t_faiss:.1f}s")
print(f"  Total vectors : {index.ntotal:,}")
print(f"  Index size    : {index_size_mb:.1f} MB")
print(f"  Saved to      : {index_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.time() - t0
print("\n" + "=" * 60)
print("DONE")
print(f"  Total passages indexed : {len(passages):,}")
print(f"  Embedding dimension    : {dim}")
print(f"  Total runtime          : {total_time/60:.1f} minutes")
print(f"  Index saved to         : {index_path}")
print(f"  Passages saved to      : {PASSAGES_FILE}")
print("=" * 60)
