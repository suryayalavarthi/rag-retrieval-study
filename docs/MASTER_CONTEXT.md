# Project: Does Retrieval Actually Help? — Empirical RAG Measurement Study

## One Line Summary
Systematic measurement study of when retrieval helps, hurts, and wastes money — releasing a labeled dataset and leaderboard that every future RAG routing paper will cite.

## Publication Goal
**Primary:** IEEE Access (journal, rolling deadline, Scopus indexed)
**Backup:** IEEE BigData 2026 workshop track
**arXiv:** Preprint first to establish priority (~July 2026)
**Citation goal:** 100+ citations from dataset and leaderboard reuse

## Paper Type
Measurement and analysis study.
NOT a systems paper. NOT a performance claim paper. NOT a cost reduction paper.
We measure. We label. We release. Others build on top.

## Primary Contribution
Systematic empirical measurement across NQ, HotpotQA, MuSiQue:
- What % of queries benefit from retrieval
- What % are hurt by retrieval
- What % are neutral
- How this varies by model size (GPT-4o vs GPT-4o-mini)
- What the dollar cost of wrong routing decisions is

## Secondary Contributions
1. **Labeled dataset on HuggingFace** — every query labeled: did retrieval help/hurt/neutral, strong model result, weak model result, cost per query
2. **Leaderboard on HuggingFace** — for future RAG routing systems to benchmark against our labeled data

## What We Are NOT Claiming
- We are NOT claiming our system beats baselines
- We are NOT claiming 40-70% cost reduction
- We are NOT claiming DeBERTa routing is better than anything
- DeBERTa scorer is ONE measurement tool, not the hero of the paper

## Core Research Questions
1. What % of queries benefit from retrieval vs are hurt by it?
2. Does model size interact with retrieval benefit (GPT-4o vs GPT-4o-mini)?
3. What is the dollar cost of wrong routing decisions?
4. Can a lightweight scorer (DeBERTa) predict retrieval benefit before generation?

## Tech Stack
- Retriever: Contriever + FAISS IVFPQ (measurement instrument)
- Scorer: DeBERTa-v3-base (measurement tool, not the contribution)
- Models: GPT-4o (strong), GPT-4o-mini (weak)
- Datasets: Natural Questions, HotpotQA, MuSiQue
- Tracking: wandb
- Language: Python 3.10+
- Format: IEEE Access LaTeX template

## Datasets
| Dataset | Split | Size |
|---------|-------|------|
| Natural Questions | validation | ~3,500 |
| HotpotQA | validation | ~7,400 |
| MuSiQue | validation | ~2,400 |

## Living Documents
| File | Purpose | When to read |
|------|---------|--------------|
| `docs/MASTER_CONTEXT.md` | Full project state, decisions, status | Every session start |
| `docs/CURRENT_TASK.md` | Exactly what to build today | Every session start |
| `docs/RESEARCH_DECISIONS.md` | All decisions + reasoning | Before making any design choice |
| `docs/PROMPTS_LIBRARY.md` | Saved prompts that work | When writing prompts |
| `docs/SESSION_LOG.md` | End-of-session logs | Every session start (last entry) |
| `docs/LABELING_PROTOCOL.md` | Exact labeling rules | Before touching 03_generate_labels.py |
| `docs/ADVISOR_CONTEXT.md` | Paste into advisor chat at session start | Every advisor strategy session |
| `docs/KAGGLE_SETUP.md` | Kaggle session setup and secrets | Before any Kaggle session |

## Codebase Status (as of March 2026)
### EXISTS:
- src/retriever.py — Contriever + FAISS retrieval
- src/sufficiency_scorer.py — DeBERTa classifier + continuous score
- src/router.py — threshold router + baseline routers
- src/generator.py — OpenAI + local HF generator, cost tracking
- src/pipeline.py — end-to-end orchestration
- src/metrics.py — EM/F1 + bootstrap CIs
- src/data_utils.py — NQ, HotpotQA, MuSiQue loaders
- configs/base_config.yaml — config schema (not wired yet)
- scripts/01_download_data.py — dataset download
- scripts/00_sanity_check.py — BM25 proxy check (ran, FAIL = proxy too weak, not hypothesis)

### MISSING (must build):
- scripts/03_generate_labels.py (KAGGLE)
- scripts/04_train_scorer.py (KAGGLE)
- scripts/05_run_experiments.py (KAGGLE)
- scripts/06_generate_figures.py (LAPTOP)

### KNOWN BUGS (fix only when task says to):
- pipeline.py feeds costs=[0.0,...] to metrics — cost metrics meaningless
- QueryOnlyRouter receives no query string from pipeline.py
- Dataset name mismatch: config uses "hotpot_qa"/"dwivedodaya/musique", code expects "hotpotqa"/"musique"
- configs/base_config.yaml not wired to any code

## Compute Strategy
| Script | Where | Reason |
|--------|-------|--------|
| 00_sanity_check.py | LAPTOP | Fast, no GPU, 200 samples |
| 02_build_index.py | KAGGLE | Full corpus, needs memory |
| 03_generate_labels.py | KAGGLE | Thousands of API calls |
| 04_train_scorer.py | KAGGLE | DeBERTa GPU training |
| 05_run_experiments.py | KAGGLE | Full dataset, multiple models |
| 06_generate_figures.py | LAPTOP | Pure Python, fast |

## Key Decisions Made
- Paper type: measurement study, NOT systems paper — framing is "we measure", not "we beat"
- FreshQA dropped → MuSiQue (larger, harder, reproducible)
- Target IEEE Access (rolling deadline) + backup IEEE BigData 2026
- Dataset + leaderboard release = citation magnet strategy
- DeBERTa is a measurement tool, not the contribution
- Train scorer on NQ, evaluate generalization on HotpotQA + MuSiQue
- Use IEEE Access LaTeX template from day one

## Open Questions
1. Hydra vs argparse for config wiring?
2. Second verifier (Claude API): include or drop given framing shift?
3. HuggingFace dataset card format — draft early or after experiments?

## Budget
| Item | Cost |
|------|------|
| Label generation (GPT-4o + GPT-4o-mini, ~13K queries) | $40-55 |
| Full experiments (all baselines × 3 datasets) | $45-60 |
| Debugging and reruns buffer | $20-25 |
| **Total project budget** | **$140** |
| **Already spent** | **$0** |
| **Remaining** | **$140** |

## GitHub
Repo: https://github.com/suryayalavarthi/rag-retrieval-study
Visibility: Private until paper submission
Do not commit: results/, data/, *.faiss, *.bak, API keys

## Current Status
> **Update this section at the end of every session.**
- [ ] 02_build_index.py — COMPLETE (test run 100k passages). Full 21M run pending.
  - 100,000 Wikipedia passages indexed (TEST MODE)
  - MiniLM-L6-v2 embeddings 384 dimensions
  - 694 pass/sec on Kaggle P100
  - Runtime: 7.6 minutes (TEST MODE)
  - Full 21M run: ~8.4 hours encoding + 45 min FAISS build
- [ ] Labels generated (03_generate_labels.py)
- [ ] Scorer trained (04_train_scorer.py)
- [ ] Main experiments run (05_run_experiments.py)
- [ ] Figures generated (06_generate_figures.py)
- [ ] Dataset uploaded to HuggingFace
- [ ] Leaderboard created on HuggingFace
- [ ] Paper drafted
- [ ] arXiv submitted
- [ ] IEEE Access submitted
