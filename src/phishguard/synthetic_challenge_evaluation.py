"""Aggregate-only evaluation of existing models on fictional challenge emails."""

import json
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from phishguard.clean_data import normalize_email_text
from phishguard.data import LABELS
from phishguard.synthetic_challenge import (
    LEGITIMATE,
    PHISHING,
    SYNTHETIC_CHALLENGE_SET,
)

DEFAULT_V2_MODEL = Path("models/v3/grouped-experiment/v2_baseline_grouped.joblib")
DEFAULT_V3_MODEL = Path("models/v3/grouped-experiment/v3_masked_grouped.joblib")
DEFAULT_RESULTS_PATH = Path("results/v3/synthetic-challenge/metrics.json")


def _evaluate_partition(labels: list[str], predictions: list[str]) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "phishing_precision": float(
            precision_score(labels, predictions, pos_label=PHISHING, zero_division=0)
        ),
        "phishing_recall": float(
            recall_score(labels, predictions, pos_label=PHISHING, zero_division=0)
        ),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
    }


def evaluate_challenge_models(
    models: dict[str, object],
    challenge_set=SYNTHETIC_CHALLENGE_SET,
    training_texts=None,
    results_path: Path | None = None,
) -> dict:
    """Evaluate named saved models and save only metrics and scenario aggregates."""
    if not models:
        raise ValueError("At least one saved model is required.")
    if len(challenge_set) != 40:
        raise ValueError("Challenge evaluation requires exactly 40 messages.")

    labels = [item.label for item in challenge_set]
    texts = [item.text for item in challenge_set]
    scenarios: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(challenge_set):
        scenarios[item.scenario].append(index)

    duplicate_counts = {"exact": 0, "normalized": 0}
    if training_texts is not None:
        training_text_list = [str(text) for text in training_texts]
        training_exact = set(training_text_list)
        training_normalized = {
            normalize_email_text(text) for text in training_text_list
        }
        duplicate_counts = {
            "exact": sum(text in training_exact for text in texts),
            "normalized": sum(
                normalize_email_text(text) in training_normalized for text in texts
            ),
        }

    model_reports = {}
    predictions_by_model: dict[str, list[str]] = {}
    for model_name, model in models.items():
        predictions = list(model.predict(texts))
        if len(predictions) != len(labels):
            raise ValueError(
                f"Model {model_name!r} returned an unexpected number of predictions."
            )
        if not set(predictions).issubset(LABELS):
            raise ValueError(f"Model {model_name!r} returned an unsupported label.")
        predictions_by_model[model_name] = predictions
        scenario_reports = {}
        for scenario, indices in sorted(scenarios.items()):
            scenario_labels = [labels[index] for index in indices]
            scenario_predictions = [predictions[index] for index in indices]
            scenario_metrics = _evaluate_partition(
                scenario_labels, scenario_predictions
            )
            scenario_metrics["misclassified"] = sum(
                actual != predicted
                for actual, predicted in zip(scenario_labels, scenario_predictions)
            )
            scenario_reports[scenario] = scenario_metrics
        model_reports[model_name] = {
            "overall": _evaluate_partition(labels, predictions),
            "by_scenario": scenario_reports,
        }

    disagreement_by_scenario = {}
    if len(predictions_by_model) > 1:
        model_names = sorted(predictions_by_model)
        for scenario, indices in sorted(scenarios.items()):
            disagreement_by_scenario[scenario] = sum(
                len({predictions_by_model[name][index] for name in model_names}) > 1
                for index in indices
            )
        total_disagreements = sum(
            len({predictions[index] for predictions in predictions_by_model.values()})
            > 1
            for index in range(len(labels))
        )
        disagreement_summary = {
            "models_compared": model_names,
            "total_disagreements": total_disagreements,
            "by_scenario": disagreement_by_scenario,
        }
    else:
        disagreement_summary = {
            "models_compared": list(predictions_by_model),
            "total_disagreements": 0,
            "by_scenario": {},
        }

    report = {
        "evaluation": "fictional synthetic challenge set; not official research metrics",
        "challenge_size": len(challenge_set),
        "label_counts": {
            label: labels.count(label) for label in (LEGITIMATE, PHISHING)
        },
        "scenario_counts": {
            scenario: len(indices) for scenario, indices in sorted(scenarios.items())
        },
        "training_duplicate_check": {
            "method": "raw exact match and casefold/Unicode/whitespace-normalized exact match",
            "compared_against_training_examples": training_texts is not None,
            "matching_challenge_messages": duplicate_counts,
        },
        "models": model_reports,
        "prediction_disagreements": disagreement_summary,
        "limitations": [
            "The challenge set is small, hand-authored, and fully synthetic.",
            "Scenario-level scores are based on only a few examples and are not reliable population estimates.",
            "Exact duplicate checks do not detect paraphrases or all near-duplicates.",
            "Synthetic wording and reserved domains may differ substantially from real phishing and legitimate email.",
        ],
    }
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def run_saved_model_challenge(
    v2_model_path: Path = DEFAULT_V2_MODEL,
    v3_model_path: Path = DEFAULT_V3_MODEL,
    training_texts=None,
    results_path: Path = DEFAULT_RESULTS_PATH,
) -> dict:
    """Load existing grouped models and run the synthetic challenge evaluation."""
    models = {
        "v2_grouped": joblib.load(v2_model_path),
        "v3_grouped": joblib.load(v3_model_path),
    }
    return evaluate_challenge_models(
        models,
        training_texts=training_texts,
        results_path=results_path,
    )
