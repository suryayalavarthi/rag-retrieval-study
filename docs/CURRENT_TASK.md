# Current Task

> Update this file BEFORE every Claude Code session. Paste it at the top of your first message.

---

## Current Task
### What I am building today
scripts/02_build_index.py

### What it does
Builds a FAISS retrieval index using Contriever
embeddings over a Wikipedia passage corpus.
This index is used in all three dataset experiments
to retrieve passages for each query.

### RUN ON: KAGGLE
Reason: Encoding Wikipedia passages with Contriever
requires GPU and significant memory.

### What done means
- Script downloads or loads Wikipedia passages
- Encodes passages using Contriever
- Builds FAISS flat index
- Saves index to results/faiss_index/
- Saves passage texts to results/passages.jsonl
- Prints: total passages indexed, encoding time,
  index size on disk
- Script completes without errors on Kaggle T4

### Corpus size
Start with 100,000 Wikipedia passages for
initial experiments.
Full corpus can be expanded later if needed.
This keeps Kaggle runtime under 2 hours.

### Constraints
- RUN ON: KAGGLE
- Use facebook/contriever-msmarco model
- Use faiss-cpu (not faiss-gpu — more stable)
- Save all outputs to BASE_DIR/results/
  using environment-aware paths
- Add os.environ.get() for all keys
- Print progress every 10,000 passages
- Seed: 42
- Do not touch any existing source files
- After creating script, update SESSION_LOG.md

### Definition of done
Script exists at scripts/02_build_index.py
Contains RUN ON: KAGGLE comment at top
Saves index and passages to results/
Completes in under 2 hours on Kaggle T4
