# Session Log

> Last 5 minutes of every session: fill this in. Keep last 5 sessions.

---

## Session — 2026-03-19
### What was done
- Created full living document system (MASTER_CONTEXT, RESEARCH_DECISIONS, CURRENT_TASK, PROMPTS_LIBRARY, SESSION_LOG)
- Confirmed publication goal: IEEE/Scopus (not preprint-only)
- Created zip of reference files for AI project upload
- Wrote AI project instructions focused on IEEE rigor
- Identified all known bugs and missing scripts in codebase

### What works
- Core src/ modules all exist and are logically sound
- Architecture matches paper narrative cleanly

### What is broken / missing
- No experiment runner (scripts 02-06 missing)
- cost tracking hardcoded to 0.0 in pipeline.py
- QueryOnlyRouter gets no query string
- Dataset name mismatch in config vs code
- Config not wired to any code

### Next session must start with
Fix bugs listed above + build scripts/05_run_experiments.py as minimal runner.
Paste MASTER_CONTEXT.md + CURRENT_TASK.md at top of Claude Code session.

---

## Session — 2026-03-19 (Session 2)
### What was built
- `scripts/00_sanity_check.py` — free sanity check, zero API calls
- `results/` directory created

### Files changed
- `scripts/00_sanity_check.py` — created
- `results/` — directory created
- `docs/MASTER_CONTEXT.md` — paper pivoted to measurement study framing
- `docs/CURRENT_TASK.md` — updated to sanity check task
- `CLAUDE.md` — paper title updated to match pivot

### What works
- Sanity check script is complete and ready to run
- Loads 200 NQ validation samples via HuggingFace datasets
- BM25 retrieval with rank_bm25 (no API calls)
- Computes Pearson correlation between proxy sufficiency score and success label
- Prints routing distribution + PASS/WEAK/FAIL verdict
- Saves full results to results/sanity_check_output.json

### What is broken or incomplete
- Script not yet run — needs `pip install datasets rank-bm25 scipy scikit-learn`
- All prior src/ bugs still exist (fix only when task says to)

### Next session should start with
Run `python scripts/00_sanity_check.py` and review the verdict.
If PASS → proceed to scripts/02_build_index.py + scripts/03_generate_labels.py
If WEAK/FAIL → return to AI advisor chat to rethink before spending money.

---

## Session — 2026-03-19 (Session 3)
### What was built
- `docs/KAGGLE_SETUP.md` — full Kaggle reference guide
- Compute strategy added to `CLAUDE.md` and `docs/MASTER_CONTEXT.md`

### Files changed
- `CLAUDE.md` — added "Compute Strategy — Where to Run What" section
- `docs/MASTER_CONTEXT.md` — added compute strategy table
- `docs/KAGGLE_SETUP.md` — created

### What works
- Every future script will be labeled RUN ON: LAPTOP or RUN ON: KAGGLE
- Claude Code will never leave the user guessing where to run something
- Kaggle secrets, GPU tips, and save/push workflow all documented

### What is broken or incomplete
- GitHub repo not yet created — needed before any Kaggle sessions
- `00_sanity_check.py` not yet run

### Next session should start with
1. Create GitHub repo and push project (non-negotiable before Kaggle)
2. Run `python scripts/00_sanity_check.py` on laptop
3. Report verdict here and update CURRENT_TASK.md accordingly

---

## Session — 2026-03-19 (Session 4)
### What was built
- Permanent strategic pivot applied across all three core files

### Files changed
- `docs/MASTER_CONTEXT.md` — full rewrite: measurement study framing, updated contributions, removed all "beats baselines" language
- `CLAUDE.md` — updated paper type, added "what this paper claims / does not claim" section
- `docs/RESEARCH_DECISIONS.md` — added Decision 3 (measurement study), Decision 4 (citation strategy), Decision 5 (scorer as tool), Decision 7 (argparse leaning), Decision 8 (sanity check result)

### What works
- All three files now reflect the correct paper type: measurement study, not systems paper
- DeBERTa scorer correctly framed as measurement tool, not contribution
- Dataset + leaderboard release correctly identified as citation strategy
- IEEE Access (rolling deadline) confirmed as primary venue

### What is broken or incomplete
- CURRENT_TASK.md not yet updated for next step
- GitHub repo still not created
- `00_sanity_check.py` ran: FAIL (r=0.02) — proxy too weak, not hypothesis failure

### Next session should start with
Build `scripts/02_build_index.py` on Kaggle — Contriever + FAISS index over Wikipedia passages.
This is the real retrieval signal. After this, re-run sufficiency check with actual dense retrieval.

---

## Session — 2026-03-19 (Session 5)
### What was built
- Budget aligned across both files to single agreed number

### Files changed
- `docs/MASTER_CONTEXT.md` — budget updated: $140 total, $0 spent, $140 remaining
- `docs/ADVISOR_CONTEXT.md` — budget updated: same breakdown

### What works
- Both files now show identical budget: $140 total / $0 spent / $140 remaining
- Breakdown: labels $40-55, experiments $45-60, buffer $20-25

### What is broken or incomplete
- Nothing new broken

### Next session should start with
Build `scripts/02_build_index.py` on Kaggle (RUN ON: KAGGLE)

---

## Session — 2026-03-19 (Session 8)
### What was built
- Fixed contradiction in `docs/LABELING_PROTOCOL.md`

### Files changed
- `docs/LABELING_PROTOCOL.md` — "What Happens When Edge Cases Occur" section: retrieval failure now excluded from all label buckets, consistent with "Retrieval Failure Handling" section

### What works
- Protocol is now internally consistent: retrieval failures are never labeled HURT

### What is broken or incomplete
- Nothing

### Next session should start with
Build `scripts/02_build_index.py` on Kaggle (RUN ON: KAGGLE)

---

## Session — 2026-03-19 (Session 7)
### What was built
- 5 new sections added to `docs/LABELING_PROTOCOL.md`

### Files changed
- `docs/LABELING_PROTOCOL.md` — added: Answer Normalization Policy, Multi-Answer Handling, Threshold Robustness Check, Random Passage Specification, Retrieval Failure Handling, Human Validation Sample

### What works
- Labeling protocol is now complete enough to write 03_generate_labels.py
- Answer normalization follows standard SQuAD protocol
- Random passage spec includes irrelevance check + seed + fallback
- Retrieval failures tracked separately (not counted as HURT)
- Human validation target: 150 queries, >85% agreement

### What is broken or incomplete
- Nothing new broken

### Next session should start with
Build `scripts/02_build_index.py` on Kaggle (RUN ON: KAGGLE)

---

## Session — 2026-03-19 (Session 10)
### What was built
- `.gitignore` — updated with *.bak, secrets.txt entries
- `README.md` — updated paper title and description to match measurement study pivot
- `docs/MASTER_CONTEXT.md` — added GitHub section

### Files changed
- `.gitignore` — added *.bak and secrets.txt
- `README.md` — title and overview updated to correct paper framing
- `docs/MASTER_CONTEXT.md` — GitHub section added

### What works
- README now reflects the correct paper title and measurement study framing
- .gitignore covers all sensitive files: results/, data/, *.faiss, *.bak, API keys
- MASTER_CONTEXT.md has GitHub repo placeholder ready to fill in

### What is broken or incomplete
- GitHub repo URL is still YOUR_USERNAME placeholder — needs real repo created

### Next session should start with
Create GitHub repo and replace YOUR_USERNAME placeholder in MASTER_CONTEXT.md.
Then run scripts/02_build_index.py on Kaggle.

---

## Session — 2026-03-19 (Session 9)
### What was built
- `scripts/02_build_index.py` — Contriever + FAISS index builder

### Files changed
- `scripts/02_build_index.py` — created

### What works
- Loads 100,000 Wikipedia passages (first 100-word chunk per article)
- Encodes with `facebook/contriever-msmarco` using mean pooling
- Builds FAISS flat L2 index
- Saves index to `BASE_DIR/results/faiss_index/index.faiss`
- Saves passages to `BASE_DIR/results/passages.jsonl`
- Prints progress every 10,000 passages
- All paths environment-aware (`KAGGLE_WORKING_DIR`)
- Seed 42 throughout
- Estimated runtime: 60-90 minutes on Kaggle T4

### What is broken or incomplete
- Script not yet run — must run on Kaggle (needs GPU)
- Requires: `pip install transformers datasets faiss-cpu torch`

### Next session should start with
Upload `scripts/02_build_index.py` to Kaggle and run it.
Then update CURRENT_TASK.md to `scripts/03_generate_labels.py`.

---

## Session — 2026-03-19 (Session 6)
### What was built
- `docs/LABELING_PROTOCOL.md` added to living documents table in both files

### Files changed
- `CLAUDE.md` — added LABELING_PROTOCOL.md row to living documents table
- `docs/MASTER_CONTEXT.md` — added full living documents table (was missing), including LABELING_PROTOCOL.md

### What works
- Both files now list all living documents including LABELING_PROTOCOL.md
- Claude Code will know to read LABELING_PROTOCOL.md before touching 03_generate_labels.py

### What is broken or incomplete
- `docs/LABELING_PROTOCOL.md` does not exist yet — needs to be created before label generation work begins

### Next session should start with
Build `scripts/02_build_index.py` on Kaggle (RUN ON: KAGGLE)
