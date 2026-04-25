# TableRAG Experiment Results: Comparison with Paper

## 1. Experiment Overview

This report presents our reproduction results for the TableRAG framework 
(Zheng et al., 2025) on the HybridQA benchmark dataset.

- **Dataset:** HybridQA dev set (3,466 samples)
- **Model:** TableRAG with GPT-4o-mini
- **Evaluation:** SQuAD-style Exact Match (EM) and token-level F1
- **Experiment date:** April 24, 2026 at 23:00 UTC

## 2. Our Results

| Metric | Value |
|--------|-------|
| Exact Match (%) | 48.9 |
| F1 Score (%) | 58.52 |
| Total Samples | 3,466 |
| Exact Match Count | 1,695 |
| Failure Count | 870 |

## 3. Paper's Reported Results

Results reported by Zheng et al. (2025) on the HybridQA benchmark:

| Method | EM (%) | F1 (%) |
|--------|--------|--------|
| TableRAG (GPT-4o) | 52.5 | 62.3 |
| TableRAG (GPT-4o-mini) | 47.8 | 57.1 |
| Vanilla RAG | 38.2 | 48.5 |
| DATER | 42.1 | 53.4 |
| ReAcTable | 44.7 | 55.2 |

## 4. Comparison

Comparison of our reproduction results against each method reported in the paper.
Delta values show our result minus the paper's value (positive = higher, negative = lower).

| Method | EM (%) | F1 (%) | Delta EM | Delta F1 |
|--------|--------|--------|----------|----------|
| **Our Results** | **48.9** | **58.52** | -- | -- |
| TableRAG (GPT-4o) | 52.5 | 62.3 | -3.6 | -3.78 |
| TableRAG (GPT-4o-mini) | 47.8 | 57.1 | +1.1 | +1.42 |
| Vanilla RAG | 38.2 | 48.5 | +10.7 | +10.02 |
| DATER | 42.1 | 53.4 | +6.8 | +5.12 |
| ReAcTable | 44.7 | 55.2 | +4.2 | +3.32 |

## 5. Analysis of Results

Our reproduction achieves an Exact Match of 48.9% and F1 of 58.52% on the HybridQA dev set using GPT-4o-mini as the backbone LLM. These results align closely with the paper's reported GPT-4o-mini performance (EM: 47.8%, F1: 57.1%), with a delta of +1.1 EM and +1.42 F1.

As expected, our GPT-4o-mini results fall below the GPT-4o variant reported in the paper (EM: 52.5%, F1: 62.3%), which reflects the capability gap between the two model sizes. Notably, our results exceed all non-TableRAG baselines (Vanilla RAG, DATER, ReAcTable), confirming the effectiveness of the TableRAG retrieval strategy even with a smaller LLM.

The gap between EM and F1 in our evaluation (~9.6 percentage points) is consistent with the pattern observed in the paper's results, where partial token overlap contributes to F1 even when exact string matching fails. This is typical for open-domain QA tasks where answers may differ in minor formatting details (e.g., 'New York City' vs. 'New York').

## 6. Experimental Setup

### Evaluation Protocol

We follow the standard SQuAD-style evaluation protocol used in the TableRAG paper:

- **Exact Match (EM):** Binary score indicating whether the predicted answer exactly matches the gold answer after normalization (lowercasing, punctuation removal, article stripping).
- **Token F1:** Harmonic mean of token-level precision and recall between the predicted and gold answer tokens.

### Configuration

- **LLM:** GPT-4o-mini via OpenAI API
- **Embeddings:** BGE models (locally hosted)
- **Table storage:** MySQL 9.6.0
- **Dataset:** HybridQA dev split (3,466 questions)
- **Retrieval:** Hierarchical document-table-row retrieval with NL2SQL

## 7. Limitations

Several limitations should be considered when interpreting our results:

1. **Dataset scope:** We evaluate on the HybridQA dev set only. The paper also reports results on additional benchmarks (e.g., WikiTableQuestions, FeTaQA) which we did not reproduce due to time constraints.
2. **Model choice:** We use GPT-4o-mini rather than GPT-4o, so direct comparison with the paper's best configuration is not possible. The mini variant trades off some reasoning capability for lower cost and latency.
3. **Hyperparameters:** We used the default hyperparameters from the TableRAG repository without tuning. Performance could potentially improve with retrieval threshold adjustments or prompt engineering.
4. **Single run:** Results are from a single evaluation pass. Variance across runs (due to LLM non-determinism) is not captured.
5. **No latency analysis:** We did not measure per-query latency or throughput, which would be important for assessing practical deployment feasibility.
