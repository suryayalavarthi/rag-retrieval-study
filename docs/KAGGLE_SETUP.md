# Kaggle Setup Guide

> Reference this file at the start of every Kaggle session.

---

## How to start every Kaggle session

Paste this at the top of every Kaggle notebook:

```python
!git clone https://github.com/YOUR_USERNAME/cost-aware-rag.git
%cd cost-aware-rag
!pip install -r requirements.txt
```

Replace `YOUR_USERNAME` with your actual GitHub username before running.

---

## How to save results before Kaggle disconnects

All results must be saved to `/kaggle/working/` during the session.
Download them manually or push to GitHub before the session ends.

**Set a timer for 25 hours** as a reminder to save — Kaggle sessions expire at 12 hours (interactive) or 9 hours (GPU).

Push results to GitHub at end of every session:
```bash
cp -r results/ /kaggle/working/results/
# Then download from Kaggle output panel, or:
!git add results/ && git commit -m "add results" && git push
```

---

## Which scripts run where

| Script | Where | Reason |
|--------|-------|--------|
| 00_sanity_check.py | LAPTOP | Fast, no GPU, 200 samples only |
| 02_build_index.py | KAGGLE | Encodes full corpus, needs memory |
| 03_generate_labels.py | KAGGLE | Thousands of API calls in loop |
| 04_train_scorer.py | KAGGLE | DeBERTa fine-tuning needs GPU |
| 05_run_experiments.py | KAGGLE | Full dataset, multiple models |
| 06_generate_figures.py | LAPTOP | Pure Python, fast, no GPU |
| Any new debug scripts | LAPTOP | Always start small |
| Any new training scripts | KAGGLE | Always needs GPU |

---

## Environment variables needed in Kaggle

Add these as **Kaggle Secrets** — never hardcode in any script or notebook.

| Secret name | What it is |
|-------------|-----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o / GPT-4o-mini |
| `WANDB_API_KEY` | Weights & Biases tracking key |

### How to add Kaggle Secrets
1. Open your notebook on Kaggle
2. Click **Settings** (top right) → **Add-ons** → **Secrets**
3. Click **Add secret**
4. Enter the name exactly as above, paste the value

### How to read secrets in code (always use this pattern)
```python
import os
from kaggle_secrets import UserSecretsClient

# In Kaggle notebooks:
secrets = UserSecretsClient()
os.environ["OPENAI_API_KEY"] = secrets.get_secret("OPENAI_API_KEY")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")

# In scripts run via !python ..., use os.environ.get():
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set — add it as a Kaggle Secret")
```

---

## Kaggle GPU tips for this project

- Use **T4 x2** for `04_train_scorer.py` (DeBERTa fine-tuning)
- Use **T4 x1** for `02_build_index.py` and `05_run_experiments.py`
- GPU quota resets weekly — use it for training, not data loading
- If GPU quota is exhausted, `03_generate_labels.py` can run on CPU (API-bound, not compute-bound)

---

## Critical warnings

- **Never hardcode API keys** in any notebook or script
- **Always use `os.environ.get()`** — never `os.environ["KEY"]` directly (crashes if missing)
- **Save to `/kaggle/working/`** — files outside this path are lost on disconnect
- **Commit to GitHub** at end of every Kaggle session — Kaggle output is not permanent
- **Do not run `00_sanity_check.py` on Kaggle** — run it on laptop, it only needs 200 samples
