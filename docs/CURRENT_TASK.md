# Current Task

> Update this file BEFORE every Claude Code session. Paste it at the top of your first message.

---

## Current Task
### What I am building today
WAITING — Schedule a dedicated Kaggle session.

### Next session task
Run scripts/02_build_index.py with full DPR corpus
and IVFPQ compressed index on Kaggle.

### RUN ON: KAGGLE
Reason: 21M passages, ~13GB download,
6-8 hours encoding, GPU required.

### Prerequisites before starting
1. Fresh Kaggle session with full GPU quota
2. P100 GPU selected
3. Internet ON
4. At least 8 hours available
5. Do not start unless you can monitor it

### What done means
- 21M DPR passages indexed with Contriever
- IVFPQ compressed index saved (~10GB)
- passages.jsonl saved separately
- Both files downloaded before session ends
- Index size should be under 15GB

### Storage note
passages.jsonl is too large for Kaggle working dir.
Save it to Kaggle Datasets instead.
Only index.faiss goes to /kaggle/working/results/
