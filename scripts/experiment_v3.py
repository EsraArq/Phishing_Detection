#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from phishguard.v3_experiment import (
    V3_MODEL_DIRECTORY,
    V3_RESULTS_PATH,
    run_v3_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fresh Version 2 and masked Version 3 SVM models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/Phishing_Email.csv"),
        help="Path to the existing labeled email CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_data = pd.read_csv(args.data)
    result = run_v3_experiment(
        source_data,
        model_directory=V3_MODEL_DIRECTORY,
        results_path=V3_RESULTS_PATH,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
