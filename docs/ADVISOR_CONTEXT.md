# ADVISOR_CONTEXT.md
# Paste this at the start of every Claude AI advisor strategy session.

---

## Who I Am
Solo researcher building a research paper for IEEE Access (Scopus indexed).
I use Claude Code for building and Claude AI (you) for strategy and review.
This file gives you full context so we never repeat ourselves.

---

## Paper Title
"Does Retrieval Actually Help? A Large-Scale Empirical Study of When RAG Succeeds, Fails, and Wastes Money"

## Paper Type
Measurement and analysis study.
NOT a systems paper. NOT a performance claim paper. NOT a cost reduction paper.
We measure. We label. We release. Others build on top.

---

## The Strategic Pivot (Why We Changed Direction)
Original idea: Build a DeBERTa router that beats baselines, claim 40-70% cost reduction.
Problem: Sanity check returned Pearson r=0.02 — the proxy signal was too weak.
Pivot: Instead of claiming "our router is best," we measure what nobody has measured:
- What % of RAG queries actually benefit from retrieval?
- What % are hurt? What % are neutral?
- What does this cost in dollars?
This is harder to reject ("we measured X" is unchallengeable) and has longer citation life.

---

## What We Claim
1. Systematic measurement across NQ, HotpotQA, MuSiQue of when retrieval helps/hurts/neutral
2. Measurement by model size: GPT-4o (strong) vs GPT-4o-mini (weak)
3. Dollar cost analysis of wrong routing decisions
4. Labeled dataset released on HuggingFace (every query: help/hurt/neutral + costs)
5. Leaderboard on HuggingFace for future RAG routing systems

## What We Do NOT Claim
- Our system beats baselines
- 40-70% cost reduction
- DeBERTa routing is superior to anything
- DeBERTa is the contribution (it is a measurement tool only)

---

## Three Core Research Questions
1. What % of queries benefit from retrieval vs are hurt by it, across NQ / HotpotQA / MuSiQue?
2. Does model size interact with retrieval benefit? (Does GPT-4o-mini suffer more from bad retrieval than GPT-4o?)
3. What is the dollar cost of wrong routing decisions at scale?

---

## Target Venue
**Primary:** IEEE Access — journal, rolling deadline, Scopus indexed
**Backup:** IEEE BigData 2026 workshop track
**arXiv:** Preprint ~July 2026 to establish priority
**Citation goal:** 100+ citations from dataset + leaderboard reuse by future RAG papers

---

## Tech Stack
- Retriever: Contriever + FAISS (measurement instrument)
- Scorer: DeBERTa-v3-base (measurement tool, not the contribution)
- Models: GPT-4o (strong), GPT-4o-mini (weak)
- Datasets: NQ (~3,500), HotpotQA (~7,400), MuSiQue (~2,400)
- Tracking: wandb
- Release: HuggingFace dataset + leaderboard

---

## Current Project Status
- [x] Core src/ modules exist (retriever, scorer, router, generator, pipeline, metrics, data_utils)
- [x] Strategic pivot decided and documented
- [x] Sanity check ran: FAIL (r=0.02) — proxy too weak, not hypothesis failure
- [ ] Contriever+FAISS index built over Wikipedia (next step — Kaggle)
- [ ] Labels generated (~13K queries via GPT-4o)
- [ ] DeBERTa scorer trained
- [ ] Main experiments run
- [ ] Figures and tables generated
- [ ] Dataset uploaded to HuggingFace
- [ ] Leaderboard created
- [ ] Paper drafted
- [ ] arXiv submitted
- [ ] IEEE Access submitted

---

## Budget
| Item | Estimated Cost |
|------|---------------|
| Label generation (GPT-4o + GPT-4o-mini, ~13K queries) | $40-55 |
| Full experiments (all baselines × 3 datasets) | $45-60 |
| Debugging and reruns buffer | $20-25 |
| **Total project budget** | **$140** |
| **Already spent** | **$0** |
| **Remaining** | **$140** |
No money spent yet. All work so far is zero-cost.

---

## Next Action Item
Build `scripts/02_build_index.py` — Contriever + FAISS index over Wikipedia passages.
**RUN ON: KAGGLE** (full corpus encoding needs GPU memory).
This enables real dense retrieval, replacing the failed BM25 proxy.
After this: re-run sufficiency check with actual retrieval signal.

---

## Key Decisions Already Made (do not re-litigate)
1. MuSiQue replaces FreshQA — final
2. IEEE Access is primary venue — final
3. Paper type = measurement study — final
4. DeBERTa = measurement tool, not contribution — final
5. Dataset + leaderboard = citation strategy — final
6. Train scorer on NQ, test generalization on HotpotQA + MuSiQue — final
7. Sanity check FAIL = proxy failure, not hypothesis failure — proceed

## Open Questions (bring these to advisor sessions)
1. argparse vs Hydra for config wiring?
2. Should second AI verifier (Claude API) be included or dropped given new framing?
3. HuggingFace dataset card — draft early or after experiments?
4. How to frame DeBERTa as "measurement tool" in the paper without underselling it?
