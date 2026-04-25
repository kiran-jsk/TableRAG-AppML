#!/usr/bin/env python3
"""
Local evaluation script for TableRAG experiment results.

Computes Exact Match (EM) and token-level F1 scores using standard
SQuAD-style metrics. No LLM judge or API calls required.
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TableRAG results with SQuAD-style EM/F1 metrics"
    )
    parser.add_argument(
        "--input",
        default="experiments/results/tablerag_results.jsonl",
        help="Path to results JSONL file (default: experiments/results/tablerag_results.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="experiments/results/evaluation_metrics.json",
        help="Path to output metrics JSON file (default: experiments/results/evaluation_metrics.json)",
    )
    return parser.parse_args()


def normalize_answer(text):
    """Normalize answer text for comparison.

    Applies: lowercase, strip, remove articles, remove punctuation,
    collapse whitespace. Standard SQuAD normalization.
    """
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def compute_exact_match(prediction, gold):
    """Compute exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def compute_f1(prediction, gold):
    """Compute token-level F1 score between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def is_failure_response(answer):
    """Check if the answer is a failure/refusal response."""
    lower = answer.lower()
    failure_indicators = [
        "cannot determine",
        "does not contain",
        "unable to find",
    ]
    return any(indicator in lower for indicator in failure_indicators)


def main():
    args = parse_args()

    # Read results
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("ERROR: No records found in input file.", file=sys.stderr)
        sys.exit(1)

    total = len(records)

    # Compute per-sample metrics
    samples = []
    em_total = 0.0
    f1_total = 0.0
    em_count = 0
    failure_count = 0

    for record in records:
        gold = record["answer-text"]
        prediction = record["tablerag_answer"]
        question_id = record["question_id"]

        em = compute_exact_match(prediction, gold)
        f1 = compute_f1(prediction, gold)

        em_total += em
        f1_total += f1

        if em == 1.0:
            em_count += 1

        if is_failure_response(prediction):
            failure_count += 1

        samples.append({
            "question_id": question_id,
            "em": em,
            "f1": round(f1, 4),
        })

    # Aggregate metrics
    exact_match_pct = round(100.0 * em_total / total, 2)
    f1_pct = round(100.0 * f1_total / total, 2)

    metrics = {
        "exact_match": exact_match_pct,
        "f1_score": f1_pct,
        "total_samples": total,
        "exact_match_count": em_count,
        "failure_count": failure_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }

    # Write metrics
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"=== TableRAG Evaluation Results ===")
    print(f"Total samples:    {total}")
    print(f"Exact Match:      {exact_match_pct}%")
    print(f"F1 Score:         {f1_pct}%")
    print(f"Exact matches:    {em_count} / {total}")
    print(f"Failures:         {failure_count} / {total}")


if __name__ == "__main__":
    main()
