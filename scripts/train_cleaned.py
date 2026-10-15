#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from phishguard.clean_data import prepare_clean_dataset, split_clean_dataset
from phishguard.evaluation import evaluate_predictions
from phishguard.models import build_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, train, and evaluate both email classifiers."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/Phishing_Email.csv")
    )
    parser.add_argument("--text-column", default="Email Text")
    parser.add_argument("--label-column", default="Email Type")
    parser.add_argument("--model-dir", type=Path, default=Path("models/cleaned"))
    parser.add_argument(
        "--results", type=Path, default=Path("results/cleaned/metrics.json")
    )
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=Path("results/metrics.json"),
        help="Existing baseline report used for comparison; it is read only.",
    )
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    return args


def json_safe_parameters(estimator) -> dict:
    return {
        key: value.__name__ if isinstance(value, type) else value
        for key, value in estimator.get_params(deep=False).items()
    }


def main() -> None:
    args = parse_args()
    source_data = pd.read_csv(args.data)
    cleaned_data, cleaning_counts = prepare_clean_dataset(
        source_data, args.text_column, args.label_column
    )
    x_train, x_test, y_train, y_test = split_clean_dataset(
        cleaned_data,
        args.test_size,
        args.random_state,
        args.text_column,
        args.label_column,
    )

    with args.baseline_results.open(encoding="utf-8") as baseline_file:
        baseline_results = json.load(baseline_file)
    baseline_svm_accuracy = baseline_results["models"]["svm"]["accuracy"]

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": str(args.data),
        "cleaning": cleaning_counts,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "split_method": "stratified after cleaning and normalized-text deduplication",
        "normalization": (
            "Unicode NFC, casefold, and collapse whitespace; punctuation, URLs, "
            "and other email content are preserved."
        ),
        "baseline_svm_accuracy": baseline_svm_accuracy,
        "models": {},
    }
    for name, model in build_models(args.random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        model_metrics = {
            "parameters": {
                "tfidf": json_safe_parameters(model.named_steps["tfidf"]),
                "classifier": json_safe_parameters(model.named_steps["classifier"]),
            },
            **metrics,
            "phishing_recall": metrics["phishing_class"]["recall"],
            "false_positives": metrics["confusion_matrix"][0][1],
            "false_negatives": metrics["confusion_matrix"][1][0],
        }
        if name == "svm":
            model_metrics["accuracy_delta_vs_original_baseline"] = (
                metrics["accuracy"] - baseline_svm_accuracy
            )
        results["models"][name] = model_metrics
        joblib.dump(model, args.model_dir / f"{name}.joblib")

    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
