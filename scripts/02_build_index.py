# RUN ON: KAGGLE
# Reason: Encoding 21M DPR Wikipedia passages requires GPU and significant memory.

"""
scripts/02_build_index.py

Builds a FAISS retrieval index using sentence-transformers/all-MiniLM-L6-v2
embeddings over the full DPR Wikipedia passage corpus (~21M passages).

Outputs:
  BASE_DIR/results/faiss_index/index.faiss  — IVFPQ compressed FAISS index
  BASE_DIR/results/passages_meta.jsonl      — passage id + title only (no full text)

Note: passages.jsonl (full text, ~13GB) is NOT saved to avoid Kaggle disk limit.
Full passage text is retrieved at query time via passage ID.

Run on Kaggle P100. Estimated runtime: 6-8 hours on Kaggle P100.
"""

import os
import json
import time
import random
import numpy as np
import faiss
import torch
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────
BASE_DIR = os.environ.get("KAGGLE_WORKING_DIR", ".")
RESULTS_DIR = Path(BASE_DIR) / "results"
INDEX_DIR = RESULTS_DIR / "faiss_index"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 1024
SEED = 42
PROGRESS_EVERY = 500_000
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("WARNING: This script takes 4-6 hours on Kaggle P100.")
print("Do not close the Kaggle tab.")
print("Kaggle session expires 12 hours from open.")
print("Make sure you have time to download outputs.")
print("=" * 60)
print("02_build_index.py — Build MiniLM + FAISS Index")
print("=" * 60)
print(f"Estimated runtime: 2-4 hours on Kaggle P100")
print(f"Model         : {MODEL_NAME}")
print(f"Corpus        : DPR Wikipedia (~21M passages)")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Output dir    : {RESULTS_DIR}")
print("=" * 60)

# ── Step 1: Download and load DPR Wikipedia passages ─────────────────────────
PASSAGES_TSV = RESULTS_DIR / "psgs_w100.tsv"
PASSAGES_GZ = RESULTS_DIR / "psgs_w100.tsv.gz"
PASSAGES_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"

print("\n[1/4] Downloading DPR Wikipedia passages...")
print("  Source: Facebook AI public files")
print("  Size: ~3.5GB compressed, ~13GB uncompressed")
print("  This takes 10-20 mins depending on connection")
t0 = time.time()

if not PASSAGES_TSV.exists():
    print("  Downloading with wget...")
    ret = os.system(
        f"wget -q --show-progress -O {PASSAGES_GZ} {PASSAGES_URL}"
    )
    if ret != 0:
        raise RuntimeError("wget download failed")
    print("  Download complete.")

    print("  Extracting...")
    ret = os.system(f"gunzip -f {PASSAGES_GZ}")
    if ret != 0:
        raise RuntimeError("gunzip failed")
    print("  Extraction complete.")

    # Verify
    size_gb = PASSAGES_TSV.stat().st_size / 1e9
    print(f"  TSV size: {size_gb:.1f}GB")
else:
    print("  TSV already exists, skipping download.")

print("  Loading passages from TSV...")
passages = []
with open(str(PASSAGES_TSV), 'r', encoding='utf-8') as f:
    next(f)  # skip header: id\ttext\ttitle
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            passages.append({
                "id": parts[0],
                "text": parts[1],
                "title": parts[2],
            })
        if (i + 1) % 1_000_000 == 0:
            print(f"  Loaded {i+1:,} passages...")

print(f"  Loaded {len(passages):,} passages in {time.time()-t0:.1f}s")

if TEST_MODE:
    passages = passages[:100_000]
    print(f"  TEST MODE: limited to {len(passages):,} passages")

# Delete TSV immediately to free 13GB before encoding
if PASSAGES_TSV.exists():
    os.remove(str(PASSAGES_TSV))
    print(f"  TSV deleted to free disk space for encoding.")
    # Verify free space
    import shutil
    free_gb = shutil.disk_usage("/kaggle/working").free / 1e9
    print(f"  Free disk space: {free_gb:.1f}GB")

# ── Step 2: Save passage metadata (id + title only, no full text) ─────────────
# Full passages.jsonl (~13GB) is NOT saved here — disk space on Kaggle is limited.
# Full text is retrieved at query time from the TSV via passage ID in 03_generate_labels.py.
META_FILE = RESULTS_DIR / "passages_meta.jsonl"
print(f"\n[2/4] Saving passage metadata to {META_FILE}...")
with open(str(META_FILE), "w") as f:
    for p in passages:
        f.write(json.dumps({"id": p["id"], "title": p["title"]}) + "\n")
print(f"  Saved {len(passages):,} passage metadata entries")

# ── Step 3: Encode passages with MiniLM ──────────────────────────────────────
print(f"\n[3/4] Encoding passages with {MODEL_NAME}...")
os.system("pip install -q sentence-transformers")

CHECKPOINT_FILE = RESULTS_DIR / "embeddings_checkpoint.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

if CHECKPOINT_FILE.exists():
    print(f"  Loading checkpoint from {CHECKPOINT_FILE}")
    embeddings_matrix = np.load(str(CHECKPOINT_FILE))
    print(f"  Checkpoint loaded: {embeddings_matrix.shape}")
else:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.half()  # FP16 for speed

    texts = [p["text"] for p in passages]
    all_embeddings = []
    t_encode = time.time()

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start+BATCH_SIZE]
        emb = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        all_embeddings.append(emb)

        done = min(start+BATCH_SIZE, len(texts))
        if done % PROGRESS_EVERY == 0 or done == len(texts):
            elapsed = time.time() - t_encode
            rate = done / elapsed
            remaining = (len(texts) - done) / rate / 3600
            print(f"  Progress: {done:,}/{len(texts):,} "
                  f"({elapsed:.1f}s, {rate:.0f} pass/sec, "
                  f"~{remaining:.1f}h remaining)")

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"  Done. Shape: {embeddings_matrix.shape} "
          f"Time: {time.time()-t_encode:.1f}s")

    # Save checkpoint
    np.save(str(CHECKPOINT_FILE), embeddings_matrix)
    print(f"  Checkpoint saved to {CHECKPOINT_FILE}")

# ── Step 4: Build and save FAISS index ───────────────────────────────────────
print(f"\n[4/4] Building IVFPQ FAISS index...")
t_faiss = time.time()

dim = embeddings_matrix.shape[1]
nlist = 4096
m = 48

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)

n_train = min(500_000, len(embeddings_matrix))
train_sample = embeddings_matrix[:n_train].copy()
faiss.normalize_L2(train_sample)
print(f"  Training IVF quantizer on {n_train:,} vectors...")
index.train(train_sample)
print(f"  Training complete.")

faiss.normalize_L2(embeddings_matrix)
index.add(embeddings_matrix)
index.nprobe = 64

index_path = str(INDEX_DIR / "index.faiss")
faiss.write_index(index, index_path)

index_size_mb = os.path.getsize(index_path) / (1024 ** 2)
print(f"  Index built in {time.time()-t_faiss:.1f}s")
print(f"  Total vectors : {index.ntotal:,}")
print(f"  Index size    : {index_size_mb:.1f} MB")
print(f"  Saved to      : {index_path}")

# ── Cleanup: delete TSV and GZ to free disk space ────────────────────────────
if PASSAGES_TSV.exists():
    os.remove(str(PASSAGES_TSV))
    print("  Deleted TSV to free disk space.")
if PASSAGES_GZ.exists():
    os.remove(str(PASSAGES_GZ))
    print("  Deleted GZ to free disk space.")

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.time() - t0
print("\n" + "=" * 60)
print("DONE")
print(f"  Total passages indexed : {len(passages):,} (DPR Wikipedia)")
print(f"  Embedding dimension    : {dim}")
print(f"  Total runtime          : {total_time/60:.1f} minutes")
print(f"  Index saved to         : {index_path}")
print(f"  Metadata saved to      : {META_FILE}")
print("=" * 60)
