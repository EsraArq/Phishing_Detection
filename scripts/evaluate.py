#!/usr/bin/env python3
import argparse
import json

import joblib

from phishguard.data import load_dataset
from phishguard.evaluation import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PhishGuard model against labeled email data."
    )
    parser.add_argument("--model", required=True, help="Path to a trusted .joblib model.")
    parser.add_argument("--data", required=True, help="Path to the labeled email CSV.")
    parser.add_argument("--text-column", default="Email Text")
    parser.add_argument("--label-column", default="Email Type")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    texts, labels = load_dataset(args.data, args.text_column, args.label_column)
    model = joblib.load(args.model)
    predictions = model.predict(texts)
    print(json.dumps(evaluate_predictions(labels, predictions), indent=2))


if __name__ == "__main__":
    main()
