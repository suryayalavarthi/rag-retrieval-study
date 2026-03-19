# CLAUDE.md — Auto-loaded by Claude Code every session

## On Session Start — Do This First
Read these three files in order:
1. `docs/MASTER_CONTEXT.md` — full project state
2. `docs/CURRENT_TASK.md` — today's task
3. `docs/SESSION_LOG.md` — last session summary

Then greet the user with:
> "I have read your project context. Today's task is **[X from CURRENT_TASK.md]**. Shall I proceed or do you want to update the task first?"

---

## Project Summary
**Paper:** "Does Retrieval Actually Help? A Large-Scale Empirical Study of When RAG Succeeds, Fails, and Wastes Money"
**Type:** Measurement and analysis study — NOT a systems paper, NOT a performance claim paper
**Goal:** IEEE Access (Scopus indexed, rolling deadline) — 100+ citation target via dataset + leaderboard release
**arXiv:** Preprint ~July 2026 to establish priority
**Stack:** Contriever + FAISS → DeBERTa (measurement tool) → GPT-4o / GPT-4o-mini
**Datasets:** Natural Questions, HotpotQA, MuSiQue

## What This Paper Claims (and Does NOT Claim)
**CLAIMS:** We measure what % of queries benefit/hurt/neutral from retrieval, at what cost, across 3 datasets and 2 model sizes.
**DOES NOT CLAIM:** Our system beats anything. DeBERTa routing is superior. 40-70% cost reduction.
**The DeBERTa scorer is a measurement tool. The dataset and leaderboard are the contribution.**

## Living Documents
| File | Purpose | When to read |
|------|---------|--------------|
| `docs/MASTER_CONTEXT.md` | Full project state, decisions, status | Every session start |
| `docs/CURRENT_TASK.md` | Exactly what to build today | Every session start |
| `docs/RESEARCH_DECISIONS.md` | All decisions + reasoning | Before making any design choice |
| `docs/PROMPTS_LIBRARY.md` | Saved prompts that work | When writing prompts |
| `docs/SESSION_LOG.md` | End-of-session logs | Every session start (last entry) |
| `docs/LABELING_PROTOCOL.md` | Exact labeling rules | Before touching 03_generate_labels.py |

---

## Rules — Follow Every Session

### Before writing any code
- Read `docs/CURRENT_TASK.md` — work only on what is listed there
- If the task is vague or ambiguous, ask for clarification before writing any code
- Never add features, refactor, or "improve" code that is not part of the current task
- Never frame outputs as "our method beats X" — this is a measurement paper

### While working
- Before editing any existing file, create a backup at `{filename}.bak`
- Never delete `.bak` files unless the user explicitly says to
- Never change working code unless the task explicitly requires it
- If something breaks unexpectedly, stop and report — do not attempt silent fixes
- Confirm with the user before deleting any file or directory

### After completing any task
1. Append to `docs/SESSION_LOG.md`:
```
## Session — [YYYY-MM-DD HH:MM]
### What was built
### Files changed
### What works
### What is broken or incomplete
### Next session should start with
```
2. Update the status checkboxes in `docs/MASTER_CONTEXT.md`
3. Ask the user: "Task complete. Please update `docs/CURRENT_TASK.md` for the next session."

### One task per session
- Do not start a second task without user confirmation
- If a task is too large for one session, split it and ask which part to do first

---

## Compute Strategy — Where to Run What

Every time you create or discuss a script, state clearly:
  **RUN ON: LAPTOP** or **RUN ON: KAGGLE**

### Decision Guide
Run on LAPTOP if ALL true: under 5 min, no GPU, under 500 rows, debug/sanity check.
Run on KAGGLE if ANY true: over 5 min, needs GPU, full dataset, many API calls, FAISS index.

### Script Assignments
| Script | Where | Reason |
|--------|-------|--------|
| 00_sanity_check.py | LAPTOP | Fast, no GPU, 200 samples |
| 02_build_index.py | KAGGLE | Full corpus, needs memory |
| 03_generate_labels.py | KAGGLE | Thousands of API calls |
| 04_train_scorer.py | KAGGLE | DeBERTa GPU training |
| 05_run_experiments.py | KAGGLE | Full dataset, multiple models |
| 06_generate_figures.py | LAPTOP | Pure Python, fast |

### Kaggle script rules
- Add `# RUN ON: KAGGLE` and `# Reason: [one sentence]` at top
- Use `BASE_DIR = os.environ.get("KAGGLE_WORKING_DIR", ".")`
- Save all outputs to `BASE_DIR/results/`
- Use `os.environ.get()` for all API keys — never hardcode
- Print `"Estimated runtime: X minutes on Kaggle T4"` at start

### Laptop script rules
- Add `# RUN ON: LAPTOP` at top
- Keep samples under 200
- Never call OpenAI API more than 10 times
- If over 5 minutes, stop and tell user to move to Kaggle

---

## Known Bugs (fix only when task says to)
- `pipeline.py` feeds `costs=[0.0,...]` to metrics — cost metrics are meaningless
- `QueryOnlyRouter` in `pipeline.py` receives no query string
- Config name mismatch: `base_config.yaml` uses `"hotpot_qa"` / `"dwivedodaya/musique"`, code expects `"hotpotqa"` / `"musique"`
- `configs/base_config.yaml` exists but nothing reads it — not wired

## Missing Scripts (build only when task says to)
- `scripts/02_build_index.py` — RUN ON: KAGGLE
- `scripts/03_generate_labels.py` — RUN ON: KAGGLE
- `scripts/04_train_scorer.py` — RUN ON: KAGGLE
- `scripts/05_run_experiments.py` — RUN ON: KAGGLE
- `scripts/06_generate_figures.py` — RUN ON: LAPTOP
