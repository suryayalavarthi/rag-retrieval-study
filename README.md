# RAG Retrieval Study

## Paper
"Does Retrieval Actually Help? A Large-Scale Empirical
Study of When RAG Succeeds, Fails, and Wastes Money"

## What This Is
A measurement study — not a systems paper.
We systematically measure when retrieval helps, hurts,
and is neutral across three QA datasets, releasing a
labeled dataset and public leaderboard for future work.

## Research Questions
1. What percentage of queries benefit from retrieval
   vs are hurt by it vs are neutral?
2. Does this vary by model size (GPT-4o vs GPT-4o-mini)?
3. How much of the GPT-4o quality ceiling does retrieval
   recover for GPT-4o-mini? (gap closure metric)
4. What is the dollar cost of wrong routing decisions?

## Datasets
- Natural Questions (single-hop, ~3,500 dev)
- HotpotQA (multi-hop, ~7,400 dev)
- MuSiQue (harder multi-hop, ~2,400 dev)

## Stack
- Retriever: Contriever + FAISS
- Models: GPT-4o (ceiling), GPT-4o-mini (target)
- Scorer: DeBERTa-v3-base (measurement instrument)

## Releases (coming July 2026)
- Labeled dataset on HuggingFace
- Public leaderboard for RAG routing benchmarking

## Setup
pip install -r requirements.txt

## Status
In progress. arXiv preprint target: July 2026.

## License
MIT
