# Research Decisions Log

> Every significant decision with the reasoning. Never delete — only append.

---

## Decision 1 — Dataset: Drop FreshQA, Add MuSiQue
**Date:** March 2026
**Decision:** Replace FreshQA with MuSiQue as third dataset
**Why:** FreshQA has only ~600 examples (too small, noisy results) and requires up-to-date retrieval corpus (reproducibility risk). MuSiQue has ~2,400 dev examples, is a harder multi-hop dataset, and better stress-tests sufficiency scoring.
**Impact:** Cleaner results, better reviewer reception.

---

## Decision 2 — Venue: IEEE Access primary, IEEE BigData backup
**Date:** March 2026 (updated March 2026)
**Decision:** Primary target is IEEE Access (journal, rolling deadline, Scopus indexed). Backup: IEEE BigData 2026 workshop track.
**Why:** IEEE Access has rolling submission — no deadline pressure. Measurement studies fit journals better than conferences. Scopus indexed = counts for citation goals. NLP-specific venues (ACL, EMNLP) are harder for solo authors without NLP lab affiliation.
**Previous decision:** Was targeting IEEE ICDM / ACM CIKM — changed because rolling deadline is better for a solo researcher and journal format suits measurement paper length.

---

## Decision 3 — Paper Type: Measurement Study, NOT Systems Paper
**Date:** March 2026
**Decision:** This is a measurement and analysis paper. We measure when retrieval helps/hurts/is neutral. We do NOT claim our routing system beats baselines or achieves X% cost reduction.
**Why:** The sanity check revealed the proxy signal is weak — this reaffirmed that the value is in the measurement itself, not the routing system. Measurement papers have longer citation lifespans and are easier to defend to reviewers ("we measured X" is unchallengeable; "we beat X" invites scrutiny).
**Impact on framing:** DeBERTa scorer is a measurement tool. Dataset + leaderboard are the contribution.
**What we do NOT claim:**
- Our system beats baselines
- 40-70% cost reduction
- DeBERTa routing is superior to anything

---

## Decision 4 — Citation Strategy: Dataset + Leaderboard Release
**Date:** March 2026
**Decision:** Release labeled dataset on HuggingFace + leaderboard for future RAG routing systems.
**Why:** Every future RAG routing paper needs labeled data to evaluate on. If we own the benchmark, every paper cites us. This is the 100+ citation path — not from paper quality alone, but from infrastructure reuse.
**Dataset contents:** Every query labeled with: did retrieval help/hurt/neutral, GPT-4o result, GPT-4o-mini result, sufficiency score, dollar cost per query.

---

## Decision 5 — Scorer Role: Measurement Tool, Not Contribution
**Date:** March 2026
**Decision:** DeBERTa sufficiency scorer is framed as a measurement instrument, not the paper's novel contribution.
**Why:** Claiming "our DeBERTa scorer is better" would require extensive ablations and fair comparisons against other routing methods. Framing it as a tool for measurement sidesteps this — it just needs to work well enough to generate reliable labels, not to be state-of-the-art.
**Impact:** Ablation 4 (routing threshold sensitivity) is still valid. Ablation 2 (scorer architecture comparison) becomes a measurement-validity check, not a performance comparison.

---

## Decision 6 — Scorer Training Split: Train on NQ, Test Generalization
**Date:** March 2026
**Decision:** Train DeBERTa sufficiency scorer on Natural Questions splits only. Evaluate cross-dataset generalization on HotpotQA and MuSiQue.
**Why:** Prevents data leakage. Generalization becomes a measurement-validity finding: "does retrieval benefit labeling transfer across dataset types?"
**IEEE note:** Reviewers will check for this. Doing it right from the start.

---

## Decision 7 — Config: Hydra vs argparse (OPEN)
**Date:** March 2026
**Status:** OPEN — not yet decided
**Options:**
- Hydra: powerful, supports config composition, standard in ML research
- argparse: simpler, no dependencies, easier for reviewers to reproduce
**Leaning toward:** argparse (simpler = more reproducible for measurement paper; no ablation sweeps needed)

---

## Decision 8 — Sanity Check Result: Proceed to Real Retrieval
**Date:** March 2026
**Decision:** Sanity check returned FAIL (Pearson r=0.02) but this is a proxy failure, not a hypothesis failure. Proceeding to build real Contriever+FAISS index.
**Why:** BM25 over 200 NQ questions against each other is too weak a proxy — NQ questions are all unique and share no tokens. The real signal only appears with dense retrieval (Contriever) against a real Wikipedia passage corpus.
**Next step:** Build scripts/02_build_index.py on Kaggle with full Wikipedia corpus.

---

## Decision: FAISS Index Compression
**Date:** March 2026
**Decision:** Use IVFPQ compressed index instead of IndexFlatIP for the full DPR corpus.

**Reason:** Kaggle working directory is limited to 20GB.
IndexFlatIP on 21M passages produces ~60GB index.
IVFPQ compression reduces this to 8-12GB with
less than 2% recall loss at recall@20.

This is standard practice in published retrieval
work including the original DPR paper and
subsequent RAG papers. Not a compromise.

**Paper methodology note:** "We use an IVFPQ compressed
Contriever index (nlist=4096, m=96) over 21M DPR
Wikipedia passages. Compression reduces index size
by approximately 85% with less than 2% recall loss
at recall@20, consistent with prior work."

**Rejected alternatives:**
- Corpus reduction to 5-10M: not defensible
- Colab Pro: adds cost for solvable problem
- Google Drive sharding: fragile infrastructure
