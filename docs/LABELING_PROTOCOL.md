# Labeling Protocol
## docs/LABELING_PROTOCOL.md
## Written: March 2026

---

## The Core Question This Paper Answers

When does retrieval help a cheap model (GPT-4o-mini) 
do what only an expensive model (GPT-4o) could do 
without it?

---

## What We Measure

For every query in NQ, HotpotQA, and MuSiQue we run 
four experiments:

1. GPT-4o-mini with retrieved passages
2. GPT-4o-mini with a random irrelevant passage
3. GPT-4o with retrieved passages (ceiling reference)
4. GPT-4o with a random irrelevant passage (ceiling baseline)

We use a random irrelevant passage instead of zero context 
because zero context is an unfair baseline — it tests 
whether having any context helps, not whether the specific 
retrieved content helps.

---

## How We Score Answers

Primary metric: F1 score
- Gives partial credit for partial matches
- "Barack Obama" counts as correct when answer 
  is "Barack Hussein Obama"

Secondary metric: Exact Match
- Reported alongside F1 but not used for labels
- Too strict for multi-hop answers

---

## Answer Normalization Policy
Before computing F1 and EM, normalize both predicted
answer and gold answer using this exact procedure:
1. Lowercase all text
2. Remove punctuation
3. Remove articles: "a", "an", "the"
4. Normalize whitespace
5. This normalization matches the standard SQuAD
   evaluation script used in prior QA literature

## Multi-Answer Handling
For queries with multiple valid gold answers
(common in NQ and HotpotQA):
- Compute F1 against each gold answer separately
- Take the maximum F1 as the final score
- This matches the standard NQ evaluation protocol
- Report number of multi-answer queries per dataset

---

## How We Assign Labels

Labels are based on GPT-4o-mini experience only.
GPT-4o is used only as a quality ceiling reference.

delta_F1 = F1(GPT-4o-mini with retrieval) 
         - F1(GPT-4o-mini with random passage)

HELPED:  delta_F1 > +0.20
NEUTRAL: delta_F1 between -0.20 and +0.20
HURT:    delta_F1 < -0.20

Threshold is 0.20 because differences below 0.20 
are noise not meaningful signal.

---

## Threshold Robustness Check
Before finalizing labels, run analysis at three
thresholds: 0.15, 0.20, 0.25
Report HELPED/NEUTRAL/HURT percentages at all three.
If headline findings shift dramatically across
thresholds, investigate before proceeding.
Final labels use 0.20 but paper reports all three
as robustness evidence.

## Random Passage Specification
Random passage is sampled as follows:
- Source corpus: Wikipedia passages from the
  same retrieval index used for real retrieval
- Irrelevance check: passage must not contain
  any token from the gold answer (after normalization)
- One random passage per query (not averaged)
- Random seed: 42 for all sampling
- If no irrelevant passage found after 10 attempts,
  use the passage with lowest BM25 score to query

## Retrieval Failure Handling
Track retrieval failures separately.
Do not include retrieval failures in HURT bucket.
Report failure count per dataset separately.
If failure rate exceeds 2%, investigate before
proceeding with full experiments.

## Human Validation Sample
After generating labels, manually verify 50 randomly
sampled queries per dataset (150 total).
Specifically verify borderline cases where
delta_F1 is between 0.15 and 0.25.
Report agreement rate between automatic labels
and human judgment in the paper.
Target agreement rate: above 85%.
If below 85%, review threshold choice.

---

## The Quality Gap Metric (Our Novel Contribution)

For each query we also compute:

ceiling = F1(GPT-4o with retrieval)
gap_without = ceiling - F1(GPT-4o-mini without retrieval)
gap_with = ceiling - F1(GPT-4o-mini with retrieval)

gap_closure = (gap_without - gap_with) / gap_without

This measures: how much of the GPT-4o quality ceiling 
does retrieval recover for GPT-4o-mini?

A gap_closure of 0.9 means retrieval brought GPT-4o-mini 
90% of the way to GPT-4o quality on that query.

This is our headline finding. We report average 
gap_closure across all three datasets and all 
three label categories.

---

## What Happens When Edge Cases Occur

If retrieved passage is empty or retrieval fails:
Exclude from all label buckets. Track as retrieval
failure. Report count separately per dataset.
Do not conflate infrastructure failure with
semantic harm.

If both models score 0.0 on a query:
Label as NEUTRAL. Neither model could answer it.

If gap_without is 0 (models already equal):
Exclude from gap_closure analysis. Report count separately.

---

## Why These Decisions Were Made

F1 over EM: partial matches should count as partial 
success. EM is too strict for real-world evaluation.

Random passage over zero context: tests whether 
retrieved content specifically helps, not just 
whether having any text helps.

Threshold 0.20: small differences below this level 
are measurement noise not real signal.

GPT-4o-mini as label authority: the paper is written 
for people deciding whether to use retrieval with 
cheap models. Their experience defines the labels.

GPT-4o as ceiling only: provides reference point 
for gap closure metric without contaminating labels.
