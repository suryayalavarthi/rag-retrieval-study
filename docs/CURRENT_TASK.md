# Current Task

> Update this file BEFORE every Claude Code session. Paste it at the top of your first message.

---

## Current Task
### What to build today
Run scripts/02_build_index.py locally on M1 Pro.

### RUN ON: LOCAL M1 PRO
Reason: 32GB RAM handles streaming easily. No Kaggle needed. Run overnight.

### What done means
- 21M DPR passages indexed with MiniLM (sentence-transformers/all-MiniLM-L6-v2)
- FAISS IVFPQ index saved to results/faiss_index/
- passages_meta.jsonl saved to results/
- Runtime: estimated 10-15 hours overnight

### Before running
pip install transformers faiss-cpu torch tqdm sentence-transformers rank_bm25

### Command to run
python scripts/02_build_index.py

### Do not start until
All dependencies installed and confirmed.
