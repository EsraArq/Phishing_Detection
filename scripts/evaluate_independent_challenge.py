#!/usr/bin/env python3
import json

from phishguard.independent_challenge_evaluation import (
    run_saved_independent_challenge,
)


def main() -> None:
    report = run_saved_independent_challenge()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
