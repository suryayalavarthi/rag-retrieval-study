# Critical Audit: "Retrieve, Assess, Route" Research Plan

## VERDICT: Viable with significant corrections needed

The core idea is strong and novel. However, the original plan has several issues that need fixing before we start building.

---

## CRITICAL ISSUE 1: Timeline vs. Real Deadlines

**Problem:** The plan suggested ACL 2026 SRW as a target with "March-April 2026" deadlines. The actual deadline is **March 18, 2026** — that's **13 days from today**. A 24-week plan obviously can't hit that.

**Fix — Updated Venue Strategy:**

| Venue | Deadline | Feasibility |
|-------|----------|-------------|
| ~~ACL 2026 SRW~~ | March 18, 2026 | ❌ Impossible (13 days away) |
| EMNLP 2026 Main (via ARR May cycle) | ~May 2026 submission | ⚠️ Aggressive but possible for short paper |
| EMNLP 2026 Workshops | ~Aug-Sept 2026 | ✅ Best target (5-6 months out) |
| AAAI 2027 Workshop Track | ~Aug 2026 | ✅ Good backup |
| arXiv Preprint | Anytime | ✅ Publish as soon as ready to establish priority |

**Revised strategy:** Target an **arXiv preprint by July 2026** (establishes priority and is citable), then submit to **EMNLP 2026 workshops** (deadline ~Aug-Sept 2026).

---

## CRITICAL ISSUE 2: Novelty Verification

**Status: ✅ CONFIRMED NOVEL**

I searched for papers combining context sufficiency scoring with cost-aware model routing in RAG. No existing paper does this. The closest works are:
- "Sufficient Context" (ICLR 2025) — studies sufficiency but doesn't connect to cost/routing
- RouteLLM (ICLR 2025) — routes based on query features, ignores retrieval signals
- Adaptive-RAG — adapts retrieval strategy, not generator model selection
- Self-RAG — self-reflection tokens, but uses a single model (no routing)

**The gap is real.** No one has used retrieval quality signals to inform model routing decisions.

---

## CRITICAL ISSUE 3: Dataset Choice Concerns

**Problem:** FreshQA has only ~600 examples and requires up-to-date knowledge retrieval. This is tricky because:
- Small size means noisy results
- Temporal nature means retrieval corpus needs to be current
- Reproducibility is harder

**Fix:** Replace FreshQA with **MuSiQue** (multi-hop reasoning, ~2,400 dev examples). It's a harder multi-hop dataset that better tests sufficiency. Keep Natural Questions and HotpotQA.

| Dataset | Size (dev) | Type | Tests |
|---------|-----------|------|-------|
| Natural Questions | ~3,500 | Single-hop factual | Baseline retrieval |
| HotpotQA | ~7,400 | Multi-hop reasoning | Sufficiency with complex queries |
| MuSiQue | ~2,400 | Multi-hop (harder) | Extreme sufficiency challenge |

---

## CRITICAL ISSUE 4: Evaluation Rigor

**Problem:** The original plan's metrics are good but missing some things reviewers will ask for:

**Add these metrics:**
- **Pareto frontier analysis** — plot accuracy vs. cost for different routing thresholds, showing your method dominates baselines
- **Calibration analysis** — are the sufficiency scores well-calibrated? (e.g., when you predict 0.8 sufficiency, is the weak model correct ~80% of the time?)
- **Latency overhead** — exact milliseconds added by the sufficiency scorer
- **Statistical significance** — bootstrap confidence intervals on all main results

---

## CRITICAL ISSUE 5: Sufficiency Scorer Training

**Problem:** The plan says "fine-tune DeBERTa" but doesn't address potential data leakage. If you train the sufficiency scorer on the same dataset splits you evaluate on, reviewers will (rightly) reject.

**Fix:** 
- Use a **separate training split** or entirely different dataset for training the scorer
- Cross-dataset evaluation: train on NQ, test on HotpotQA and vice versa
- This also becomes a contribution: "How well does sufficiency scoring generalize?"

---

## ISSUE 6: Missing Ablation Studies

Reviewers love ablations. Plan these from the start:

1. **Sufficiency features:** What input does the scorer need? Query only? Query + passages? Query + passages + retrieval scores?
2. **Scorer architecture:** DeBERTa vs. simpler logistic regression on retrieval features vs. LLM-based scoring
3. **Number of retrieved passages:** How does k (number of passages) affect sufficiency estimation?
4. **Routing threshold sensitivity:** How do results change as you adjust the strong/weak/abstain thresholds?

---

## ISSUE 7: Budget Underestimate

**Problem:** $20-40 is optimistic. Here's a more realistic breakdown:

| Item | Realistic Cost |
|------|---------------|
| Training data generation (strong model on ~13K queries) | $30-50 |
| Evaluation runs (multiple baselines × 3 datasets) | $20-30 |
| Iteration and debugging | $15-25 |
| **Total** | **$65-105** |

**Mitigation:** Use Llama-3-8B (free on Colab) as the weak model instead of GPT-4o-mini for some experiments. Use UC compute for batch jobs. Cache all API responses aggressively.

---

## ISSUE 8: Writing Quality for Solo Author

Without a co-author to review, writing quality is a risk. 

**Mitigations:**
1. Read 3-4 well-written SRW papers for style reference
2. Use the ACL LaTeX template from day one
3. Get feedback from UC peers (even non-NLP CS students can catch clarity issues)
4. Budget 2 weeks minimum for revision

---

## REVISED TIMELINE (22 weeks → arXiv by early August 2026)

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-2 | Literature deep-dive | Annotated bibliography of 15 papers |
| 3-4 | Environment + RAG baseline | Working RAG pipeline on all 3 datasets |
| 5-6 | Training data generation | Labeled sufficiency dataset from strong/weak model comparison |
| 7-8 | Sufficiency scorer | Trained DeBERTa classifier with validation results |
| 9-10 | Router + integration | Full pipeline: retrieve → score → route → generate |
| 11-13 | Main experiments | Full results across 3 datasets, 5 baselines |
| 14-15 | Ablations + analysis | All ablation studies, Pareto curves, calibration plots |
| 16-19 | Writing | Complete paper draft in ACL format |
| 20-21 | Feedback + revision | Get peer feedback, revise |
| 22 | Submit | arXiv preprint + prepare for EMNLP workshop submission |

---

## BOTTOM LINE

The research idea is genuinely novel and publishable. The main risks are execution speed (you're solo) and ensuring statistical rigor. The fixes above address the concrete problems in the original plan. Let's build it.
