#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from phishguard.clean_data import prepare_clean_dataset
from phishguard.synthetic_challenge_evaluation import (
    DEFAULT_RESULTS_PATH,
    DEFAULT_V2_MODEL,
    DEFAULT_V3_MODEL,
    run_saved_model_challenge,
)
from phishguard.v3_grouped_experiment import (
    group_similar_messages,
    split_by_similarity_groups,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing grouped models on fictional challenge emails."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Phishing_Email.csv"),
        help="Dataset used only to reconstruct the grouped training partition.",
    )
    parser.add_argument("--v2-model", type=Path, default=DEFAULT_V2_MODEL)
    parser.add_argument("--v3-model", type=Path, default=DEFAULT_V3_MODEL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned_data, _ = prepare_clean_dataset(pd.read_csv(args.data))
    group_ids, _ = group_similar_messages(cleaned_data["Email Text"])
    train_indices, _, _ = split_by_similarity_groups(
        cleaned_data["Email Type"], group_ids
    )
    training_texts = cleaned_data["Email Text"].reset_index(drop=True).iloc[
        train_indices
    ]
    report = run_saved_model_challenge(
        args.v2_model,
        args.v3_model,
        training_texts=training_texts,
        results_path=args.results,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
