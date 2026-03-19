# RAG Retrieval Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the code and experiments for the paper *"Does Retrieval Actually Help? A Large-Scale Empirical Study of When RAG Succeeds, Fails, and Wastes Money"*.

**Key Idea:** Empirical measurement study — we systematically measure when retrieval helps, hurts, and wastes money across NQ, HotpotQA, and MuSiQue, releasing a labeled dataset and leaderboard that future RAG routing papers can benchmark against.

## Architecture

```
Query → Retriever (Contriever + FAISS) → Retrieved Passages
                                              ↓
                                    Sufficiency Scorer (DeBERTa)
                                              ↓
                                     ┌────────┼────────┐
                                     ↓        ↓        ↓
                                  High      Medium     Low
                                Sufficiency Sufficiency Sufficiency
                                     ↓        ↓        ↓
                                Weak Model Strong Model  Abstain
                               (GPT-4o-mini) (GPT-4o)  ("I don't know")
```

## Project Structure

```
cost-aware-rag/
├── src/
│   ├── __init__.py
│   ├── retriever.py          # Dense retrieval with Contriever + FAISS
│   ├── sufficiency_scorer.py # Context sufficiency classification
│   ├── router.py             # Cost-aware routing logic
│   ├── generator.py          # LLM generation (strong/weak models)
│   ├── pipeline.py           # End-to-end RAG pipeline
│   ├── metrics.py            # Evaluation metrics (F1, EM, cost, hallucination)
│   └── data_utils.py         # Dataset loading and preprocessing
├── configs/
│   ├── base_config.yaml      # Default experiment configuration
│   └── ablation_configs/     # Configs for ablation studies
├── scripts/
│   ├── 01_download_data.py       # Download NQ, HotpotQA, MuSiQue
│   ├── 02_build_index.py         # Build FAISS retrieval index
│   ├── 03_generate_labels.py     # Generate sufficiency training labels
│   ├── 04_train_scorer.py        # Train DeBERTa sufficiency scorer
│   ├── 05_run_experiments.py     # Run main experiments
│   └── 06_generate_figures.py    # Generate paper figures and tables
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_analysis.ipynb
│   ├── 03_sufficiency_analysis.ipynb
│   └── 04_results_visualization.ipynb
├── results/                  # Experiment outputs (gitignored)
├── paper/                    # LaTeX source for the paper
├── requirements.txt
├── setup.py
└── README.md
```

## Datasets

| Dataset | Type | Size (dev) | Purpose |
|---------|------|-----------|---------|
| Natural Questions | Single-hop factual | ~3,500 | Baseline factoid retrieval |
| HotpotQA | Multi-hop reasoning | ~7,400 | Complex sufficiency testing |
| MuSiQue | Multi-hop (harder) | ~2,400 | Extreme sufficiency challenge |

## Quick Start

```bash
# Clone and install
git clone https://github.com/suryayalavarthi/rag-retrieval-study.git
cd cost-aware-rag
pip install -r requirements.txt

# Download datasets
python scripts/01_download_data.py

# Build retrieval index
python scripts/02_build_index.py

# Generate sufficiency training labels
python scripts/03_generate_labels.py

# Train sufficiency scorer
python scripts/04_train_scorer.py

# Run experiments
python scripts/05_run_experiments.py
```

## Baselines

- **Always-Strong**: All queries → GPT-4o (quality upper bound)
- **Always-Weak**: All queries → GPT-4o-mini (cost lower bound)
- **Random Router**: 50/50 routing
- **RouteLLM**: Query-only routing (no retrieval signals)
- **Self-RAG**: Self-reflective RAG with reflection tokens

## Citation

```bibtex
@article{surya2026retrieveassessroute,
  title={Retrieve, Assess, Route: Cost-Efficient RAG through Adaptive Context Sufficiency Scoring},
  author={Surya},
  year={2026}
}
```

## License

MIT License
