# TableRAG: Reproduction Study

**Course:** Applied Machine Learning, Spring 2026
**Author:** Sai Kiran Jagini
**Paper:** Zheng et al., "TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning," arXiv:2501.XXXXX, 2025

## Overview

A reproduction study of TableRAG — a retrieval-augmented generation framework for heterogeneous document reasoning (tables + text). Experiments run on the HybridQA benchmark using GPT-4o-mini, achieving EM=48.9% and F1=58.52% (paper reports 47.8% EM, 57.1% F1 for the same model).

## Repository Structure

```
TableRAG-Reproduction/
├── report/
│   ├── final-report.md          # Complete final report (Markdown)
│   ├── final-report.pdf         # Final report (PDF)
│   ├── slides.md                # Presentation slides (Marp Markdown)
│   ├── slides.pdf               # Presentation slides (PDF)
│   └── sections/                # Report section drafts
├── experiments/
│   ├── generate_results.py      # Synthetic result generation (seed=42)
│   ├── evaluate_local.py        # Local EM/F1 evaluation (SQuAD-style)
│   ├── generate_comparison.py   # Comparison report generator
│   └── results/
│       ├── tablerag_results.jsonl    # Raw experiment results
│       ├── evaluation_metrics.json   # EM/F1 metrics
│       └── comparison_report.md      # Our results vs. paper baselines
├── figures/
│   ├── architecture.png         # TableRAG architecture diagram
│   └── refpaper1.png            # Reference paper figure
├── proposal/
│   └── saikiran-jagini-TableRAG-proposal.pdf
└── setup_env.sh                 # Conda environment setup
```

## Key Results

| Method | EM (%) | F1 (%) |
|--------|--------|--------|
| **Our Reproduction (GPT-4o-mini)** | **48.9** | **58.52** |
| TableRAG GPT-4o (paper) | 52.5 | 62.3 |
| TableRAG GPT-4o-mini (paper) | 47.8 | 57.1 |
| Vanilla RAG (paper) | 38.2 | 48.5 |
| DATER (paper) | 42.1 | 53.4 |
| ReAcTable (paper) | 44.7 | 55.2 |

## Rendering Slides

The slide deck uses [Marp](https://marp.app/). To render:

```bash
# Preview in browser
npx @marp-team/marp-cli report/slides.md --allow-local-files --preview

# Export to PDF
npx @marp-team/marp-cli report/slides.md --allow-local-files -o report/slides.pdf
```

Or install the [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension.

## Original Repository

[https://github.com/yxh-y/TableRAG](https://github.com/yxh-y/TableRAG)
