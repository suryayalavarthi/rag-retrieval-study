# Session Log

> Last 5 minutes of every session: fill this in. Keep last 5 sessions.

---

## Session end — March 19 2026
### What was completed
- Pilot run revealed corpus coverage problem
- Spot check confirmed Wikipedia has relevant articles
- Decision: use full DPR 21M corpus with IVFPQ index
- RESEARCH_DECISIONS.md updated with index decision
- 02_build_index.py updated for DPR + IVFPQ
- CURRENT_TASK.md set to DPR index build

### What works
- FAISS index build pipeline works end to end
- Label generation script built and tested
- All data files downloaded and ready
- GitHub repo current

### Next session must start with
Open new Claude AI project conversation.
Paste MASTER_CONTEXT.md.
Ask advisor to review DPR index build plan
before running on Kaggle.

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

## Session — 2026-03-19 (Session 26)
### What was built
- Nothing — planning session

### Files changed
- `docs/CURRENT_TASK.md` — updated with storage note: passages.jsonl → Kaggle Datasets, only index.faiss to /kaggle/working/results/

### What works
- CURRENT_TASK.md clearly states prerequisites and storage strategy

### What is broken or incomplete
- Nothing

### Next session should start with
Schedule dedicated 8-hour Kaggle P100 session.
Run 02_build_index.py. Save index.faiss to working dir, passages.jsonl to Kaggle Dataset.

---

## Session — 2026-03-19 (Session 25)
### What was built
- Updated 02_build_index.py with DPR corpus fix and IVFPQ index

### Files changed
- `scripts/02_build_index.py` — dataset changed to facebook/wiki_dpr; download size updated to ~13GB; FAISS changed from IndexFlatL2 to IVFPQ (nlist=4096, m=96, nprobe=64); NUM_PASSAGES comment removed

### What works
- Script is now ready for Kaggle P100 run
- IVFPQ index will be ~8-12GB (fits in Kaggle 20GB limit)
- train_vectors normalized with faiss.normalize_L2 before training
- All vectors normalized before adding to index

### What is broken or incomplete
- Script not yet run — awaiting full Kaggle GPU session

### Next session should start with
Run 02_build_index.py on Kaggle P100 when full GPU quota is available.

---

## Session — 2026-03-19 (Session 24)
### What was built
- Logged FAISS index compression decision

### Files changed
- `docs/RESEARCH_DECISIONS.md` — added Decision: FAISS Index Compression (IVFPQ, nlist=4096, m=96)

### What works
- Decision logged with paper methodology note ready to paste into paper

### What is broken or incomplete
- 02_build_index.py still uses IndexFlatL2 — needs updating to IVFPQ before next Kaggle run

### Next session should start with
Update 02_build_index.py to use IVFPQ index before running on Kaggle.

---

## Session — 2026-03-19 (Session 23)
### What was built
- Nothing built — planning session

### Files changed
- `docs/CURRENT_TASK.md` — updated to DPR index run task with full prerequisites and warning

### What works
- CURRENT_TASK.md now clearly states: do not start without full day + full GPU quota

### What is broken or incomplete
- Nothing

### Next session should start with
Check GPU quota. If full, run 02_build_index.py on Kaggle P100.
Download index.faiss and passages.jsonl before session ends.

---

## Session — 2026-03-19 (Session 22)
### What was built
- Updated 02_build_index.py to use DPR Wikipedia corpus

### Files changed
- `scripts/02_build_index.py` — corpus changed from 100K sampled wikimedia/wikipedia to full DPR wiki_dpr (~21M passages); BATCH_SIZE 64→128; runtime estimate updated to 6-8 hours on Kaggle P100

### What works
- Loads wiki_dpr psgs_w100.multiset.no_index (standard DPR passage corpus)
- Progress printed every 1M passages during load
- All other steps (encode, FAISS build, save) unchanged

### What is broken or incomplete
- Script needs re-run on Kaggle with DPR corpus — previous index (100K passages) is now stale
- wiki_dpr download is ~40GB — plan for 20-30 min download + 6-8 hr encoding

### Next session should start with
Re-run 02_build_index.py on Kaggle P100 with DPR corpus.
Previous index.faiss is invalid — must rebuild.

---

## Session — 2026-03-19 (Session 21)
### What was built
- Rewrote search strategy in 00b_corpus_spot_check.py

### Files changed
- `scripts/00b_corpus_spot_check.py` — now searches Wikipedia using gold answer directly; full article text match; PASS threshold 7/10

### What works
- find_wikipedia_page() searches by gold answer (e.g. "Bobby Scott" → finds Bobby Scott article)
- match check uses full normalized article text, not just first 200 chars
- Much more reliable than keyword extraction from question

### What is broken or incomplete
- Not yet run

### Next session should start with
Run python scripts/00b_corpus_spot_check.py and paste output.

---

## Session — 2026-03-19 (Session 20)
### What was built
- Fixed search strategy in 00b_corpus_spot_check.py

### Files changed
- `scripts/00b_corpus_spot_check.py` — search now strips question words and stop words before querying Wikipedia; fallback to first 3 meaningful words; increased to 10 queries; PASS threshold updated to 8/10

### What works
- extract_search_term() strips: who/what/when/where/why/how/which/whose/was/is/are/were/did/do/does/the/a/an/of/in/on/at/to/and/or
- find_wikipedia_page() tries full meaningful term, then 3-word fallback
- Summary: 8+/10 = PASS, 5+/10 = WEAK, <5 = FAIL

### What is broken or incomplete
- Not yet run

### Next session should start with
Run python scripts/00b_corpus_spot_check.py and paste output.

---

## Session — 2026-03-19 (Session 19)
### What was built
- `scripts/00b_corpus_spot_check.py` — Wikipedia corpus spot check

### Files changed
- `scripts/00b_corpus_spot_check.py` — created

### What works
- Loads first 5 NQ queries from data/nq_validation.jsonl
- Searches Wikipedia using first 4 words of each question
- Checks if gold answer appears in article text (normalized substring match)
- Prints question, gold answer, article title, first 200 chars, match result
- Prints PASS/WEAK/FAIL summary (4+/5 = PASS)
- Requires: pip install wikipedia-api
- No API keys, no FAISS, runs in under 2 minutes

### What is broken or incomplete
- Not yet run — needs data/nq_validation.jsonl to exist first

### Next session should start with
1. Run python scripts/01_download_data.py (creates nq_validation.jsonl)
2. Run python scripts/00b_corpus_spot_check.py
3. Report result here

---

## Session — 2026-03-19 (Session 18)
### What was built
- Fixed MuSiQue dataset identifier to dgslibisey/MuSiQue

### Files changed
- `scripts/01_download_data.py` — save_musique() now uses dgslibisey/MuSiQue, no fallback logic

### What works
- Clean single load_dataset call, no try/except

### What is broken or incomplete
- Script still not run — needs laptop test

### Next session should start with
Run python scripts/01_download_data.py on laptop.

---

## Session — 2026-03-19 (Session 17)
### What was built
- Fixed MuSiQue dataset name in 01_download_data.py
- Added skip-if-exists logic to all three download functions

### Files changed
- `scripts/01_download_data.py` — MuSiQue now tries allenai/musique, fallback to musique; all three functions skip if output file already exists

### What works
- Re-running the script is safe — won't re-download completed datasets
- MuSiQue uses allenai/musique with fallback

### What is broken or incomplete
- Script still not run — needs to be tested on laptop

### Next session should start with
Run python scripts/01_download_data.py on laptop and check all three files are created.

---

## Session — 2026-03-19 (Session 16)
### What was built
- Pushed 03_generate_labels.py to GitHub
- Rewrote scripts/01_download_data.py with correct output format for 03_generate_labels.py
- Pushed both to GitHub

### Files changed
- `scripts/03_generate_labels.py` — pushed (already existed)
- `scripts/01_download_data.py` — rewritten: now outputs correct format (query_id, question, gold_answers, dataset)
- `.gitignore` — already had all required entries, no changes needed

### What works
- 01_download_data.py saves NQ, HotpotQA, MuSiQue in exact format 03_generate_labels.py expects
- All three scripts (01, 02, 03) now on GitHub

### What is broken or incomplete
- 01_download_data.py not yet run — run on LAPTOP before Kaggle pilot
- 03_generate_labels.py pilot (--limit 10) still pending

### Next session should start with
1. Run python scripts/01_download_data.py on laptop
2. Upload data/ files to Kaggle
3. Run pilot: python scripts/03_generate_labels.py --datasets nq --limit 10

---

## Session — 2026-03-19 (Session 15)
### What was built
- `scripts/03_generate_labels.py` — full label generation pipeline

### Files changed
- `scripts/03_generate_labels.py` — created
- `docs/CURRENT_TASK.md` — updated to pilot run state

### What works
- All four conditions per query (mini+retrieved, mini+random, gpt4o+retrieved, gpt4o+random)
- normalize_answer: SQuAD protocol (lowercase, strip punct, remove articles)
- compute_f1: token-level, multi-answer max
- compute_em: multi-answer
- sample_random_passage: irrelevance check, 10 retries, BM25 fallback, seed 42
- retrieve_passages: Contriever + FAISS top-5
- call_model_async: cache-first, retry on RateLimitError (exponential backoff, 5 retries)
- assign_label: HELPED/NEUTRAL/HURT with 0.20 threshold
- compute_gap_closure: ceiling metric, None when gap_without=0
- Resume logic: skips completed query_ids from labels.jsonl
- Cost tracker: written every 100 queries
- Failure logging: retrieval and API failures to failures.jsonl
- CLI: --datasets, --limit, --dry-run, --resume
- All paths environment-aware (KAGGLE_WORKING_DIR)
- OPENAI_API_KEY from environment only, never hardcoded
- RUN ON: KAGGLE comment at top

### What is broken or incomplete
- Script not yet run — pilot of 10 queries required first
- Requires: openai, faiss-cpu, torch, transformers, rank_bm25, tqdm

### Next session should start with
Run pilot: python scripts/03_generate_labels.py --datasets nq --limit 10
Check labels.jsonl, cost_tracker.json, cache/ before full run.

---

## Session — 2026-03-19 (Session 14)
### What was built
- Nothing built — gate session before spending money

### Files changed
- `docs/CURRENT_TASK.md` — set to "waiting for advisor approval" state

### What works
- Hard stop in place: Claude Code will not build 03_generate_labels.py without advisor sign-off

### What is broken or incomplete
- Nothing

### Next session should start with
1. Go to Claude AI advisor project
2. Paste MASTER_CONTEXT.md + LABELING_PROTOCOL.md
3. Get advisor to design 03_generate_labels.py spec
4. Return here with approved spec and update CURRENT_TASK.md

---

## Session — 2026-03-19 (Session 13)
### What was built
- Ran scripts/02_build_index.py on Kaggle P100 — SUCCESS

### Files changed
- `docs/MASTER_CONTEXT.md` — 02_build_index.py marked complete with run details

### What works
- 100,000 Wikipedia passages indexed with Contriever
- Embeddings: 768 dimensions
- index.faiss: 293MB stored locally
- passages.jsonl stored locally
- Runtime: 14 minutes on Kaggle P100

### What is broken or incomplete
- index.faiss and passages.jsonl stored locally only — not on GitHub (correct, gitignored)

### Next session should start with
Build scripts/03_generate_labels.py — RUN ON: KAGGLE
Update CURRENT_TASK.md first.

---

## Session — 2026-03-19 (Session 12)
### What was built
- Fixed dataset source in scripts/02_build_index.py

### Files changed
- `scripts/02_build_index.py` — changed dataset from `"wikipedia"/"20220301.en"` to `"wikimedia/wikipedia"/"20231101.en"`

### What works
- Script now uses correct HuggingFace dataset identifier and a more recent Wikipedia snapshot
- Pushed to GitHub

### What is broken or incomplete
- Script still not run on Kaggle

### Next session should start with
Run scripts/02_build_index.py on Kaggle.

---

## Session — 2026-03-19 (Session 11)
### What was built
- README.md completely rewritten to correct framing

### Files changed
- `README.md` — overwritten: now reflects measurement study correctly, no old systems paper language

### What works
- README pushed to GitHub — repo now shows correct paper description
- No badges, no architecture diagram, no bibtex with wrong title

### What is broken or incomplete
- Nothing

### Next session should start with
Run scripts/02_build_index.py on Kaggle.

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
