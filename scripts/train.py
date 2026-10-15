#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from phishguard.data import load_dataset
from phishguard.evaluation import evaluate_predictions
from phishguard.models import build_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare TF-IDF SVM and Random Forest models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Phishing_Email.csv"),
        help="Path to the labeled email CSV (default: data/Phishing_Email.csv).",
    )
    parser.add_argument("--text-column", default="Email Text")
    parser.add_argument("--label-column", default="Email Type")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1.")
    return args


def main() -> None:
    args = parse_args()
    texts, labels = load_dataset(args.data, args.text_column, args.label_column)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "dataset": str(Path(args.data)),
        "dataset_samples": len(texts),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "models": {},
    }
    for name, model in build_models(args.random_state).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results["models"][name] = {
            "parameters": {
                "tfidf": {
                    key: value.__name__ if isinstance(value, type) else value
                    for key, value in model.named_steps["tfidf"]
                    .get_params(deep=False)
                    .items()
                },
                "classifier": {
                    key: value.__name__ if isinstance(value, type) else value
                    for key, value in model.named_steps["classifier"]
                    .get_params(deep=False)
                    .items()
                },
            },
            **evaluate_predictions(y_test, predictions),
        }
        joblib.dump(model, args.output_dir / f"{name}.joblib")

    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
