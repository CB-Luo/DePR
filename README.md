# DePR: Decoupled Context Propagation and Proximity-Indexed Retrieval for Multi-Step RAG

Official implementation of **DePR: Decoupled Context Propagation and Proximity-Indexed Retrieval for Multi-Step RAG**.

## Overview

**DePR** is a training-free multi-step retrieval-augmented generation (RAG) framework for complex multi-hop question answering over Web corpora. It addresses two fundamental bottlenecks in existing multi-step RAG pipelines—**irrelevant context interference** across execution steps and **fine-grained evidence loss** under whole-document topic dilution—through two complementary query-time mechanisms:

1. **Decoupled Context Propagation (DCP):** Enforces the local Markov property over a sub-question resolution directed acyclic graph (DAG) augmented with a universal sink $s$. Each intermediate sub-question conditions exclusively on its direct parent records $\mathrm{Pa}_{\mathcal{G}}(v)$ to prevent cross-branch interference, while the universal sink aggregates resolved records across all branches for final synthesis.
2. **Proximity-Indexed Retrieval (PIR):** Complements coarse whole-document hybrid retrieval (BM25 + dense embeddings via Reciprocal Rank Fusion) with an on-demand inverted phrase mention index $\mathcal{M}(e_v)$, scoring local proximity windows ($\pm 1$ sentence around known entity mentions) to recover fine-grained bridge facts buried in pages devoted to other topics.

## Repository Structure

```text
DePR/
├── src/
│   ├── config.py               # Hyperparameter configurations (K=3, RRF c=60, window size ±1)
│   ├── decomposition.py        # Sub-question resolution DAG construction
│   ├── executor.py             # Topological execution and Decoupled Context Propagation (DCP)
│   ├── retrieval.py            # Coarse hybrid retrieval (BM25 + dense RRF)
│   ├── proximity_retrieval.py  # Proximity-Indexed Retrieval (PIR) over mention windows
│   ├── synthesis.py            # Step reading and universal sink answer synthesis
│   └── prompts/                # Prompt templates for decomposition, step reading, and synthesis
├── scripts/
│   ├── run_depr.py             # Main inference entry point across evaluation sets
│   └── evaluate.py             # Standardized evaluation (EM, F1, Str, LLM Judge)
├── requirements.txt            # Python package dependencies
└── README.md
```

## Environment Setup

```bash
conda create -n depr python=3.11 -y
conda activate depr
pip install -r requirements.txt
```

Configure your API key via environment variables before running inference:

```bash
export OPENAI_API_KEY="your_api_key_here"
export PYTHONHASHSEED=0
```

## Datasets

We evaluate **DePR** on three general multi-hop QA benchmarks and two targeted structural diagnostic sets:
- **General Multi-Hop Benchmarks:** `2WikiMultiHopQA` ($n=1,000$), `HotpotQA` ($n=1,000$), and `MuSiQue` ($n=1,000$).
- **Targeted Structural Diagnostic Sets:** `Bridge-400` ($n=400$, parallel branch convergence) and `4-Hop-405` ($n=405$, deep 4-step serial and converging tracking).

## Quick Start

```bash
PYTHONHASHSEED=0 python scripts/run_depr.py \
    --dataset musique \
    --model gpt-4o-mini \
    --top_k 3 \
    --rrf_c 60 \
    --proximity_window 1 \
    --output_dir outputs/musique
```

## Main Results

| Method | Paradigm | 2Wiki (EM / F1) | HotpotQA (EM / F1) | MuSiQue (EM / F1) | Bridge-400 (EM / F1) | 4-Hop-405 (EM / F1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| GPT-4o-mini | Vanilla RAG | 24.30 / 26.55 | 27.20 / 33.79 | 11.90 / 16.50 | 22.00 / 22.24 | 4.20 / 7.39 |
| HippoRAG 2 | Offline Graph-based | 44.40 / 52.85 | 44.50 / 57.60 | 22.90 / 35.58 | 55.75 / 58.31 | 13.83 / 22.27 |
| GFM-RAG | Offline Graph-based | 56.70 / 65.16 | 47.90 / 62.46 | 23.90 / 36.13 | 81.50 / 81.90 | 11.85 / 19.80 |
| KG²RAG | Offline Graph-based | 47.30 / 57.80 | 43.60 / 59.62 | 17.70 / 32.60 | 62.00 / 63.61 | 8.40 / 17.94 |
| LinearRAG | Offline Graph-based | 44.10 / 55.21 | 37.90 / 50.82 | 16.80 / 27.18 | 73.50 / 76.30 | 13.58 / 20.89 |
| LogicRAG | Multi-step (decomp.) | 55.40 / 65.74 | 46.50 / 59.17 | 26.80 / 38.77 | 72.25 / 72.96 | 13.83 / 22.92 |
| DualRAG | Multi-step (iterative) | 59.50 / 70.56 | 47.40 / 57.75 | 25.50 / 38.67 | 90.00 / 90.15 | 14.07 / 24.21 |
| IRCoT | Multi-step (iterative) | 38.80 / 52.66 | 46.50 / 60.12 | 25.80 / 39.45 | 73.25 / 77.65 | 16.05 / 24.62 |
| ChainRAG (syn.) | Multi-step (decomp.) | 41.60 / 46.21 | 34.10 / 45.06 | 15.70 / 26.88 | 75.50 / 75.98 | 9.14 / 19.11 |
| ChainRAG (ctx.) | Multi-step (decomp.) | 37.90 / 42.58 | 41.50 / 54.84 | 18.30 / 29.51 | 56.00 / 56.97 | 12.59 / 22.32 |
| **DePR (Ours)** | **Multi-step (decomp.)** | **67.50 / 75.89** | **55.70 / 68.97** | **40.50 / 52.23** | **91.88 / 92.00** | **24.44 / 30.60** |
