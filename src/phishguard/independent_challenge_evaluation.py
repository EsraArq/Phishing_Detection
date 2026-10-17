"""Evaluate the independent synthetic set with indicators and saved SVMs."""

import json
from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.metrics import confusion_matrix

from phishguard.data import LABELS
from phishguard.independent_synthetic_challenge import (
    INDEPENDENT_SYNTHETIC_CHALLENGE_SET,
)
from phishguard.risk_indicators import detect_security_risk_indicators
from phishguard.synthetic_challenge import LEGITIMATE, PHISHING
from phishguard.synthetic_challenge_evaluation import (
    DEFAULT_V2_MODEL,
    DEFAULT_V3_MODEL,
)

INDEPENDENT_RESULTS_PATH = Path(
    "results/v3/synthetic-challenge-independent/metrics.json"
)


def _binary_metrics(labels: list[str], predictions: list[str]) -> dict:
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    true_positive = int(matrix[1, 1])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_negative = int(matrix[0, 0])
    return {
        "samples": len(labels),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "phishing_detection_rate": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "false_positive_rate": (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else 0.0
        ),
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
    }


def evaluate_independent_challenge(
    models: dict[str, object],
    challenge_set=INDEPENDENT_SYNTHETIC_CHALLENGE_SET,
    results_path: Path | None = None,
) -> dict:
    """Run independent indicator/model metrics and persist aggregate-only output."""
    if len(challenge_set) != 60:
        raise ValueError("Independent challenge evaluation requires 60 messages.")
    if not models:
        raise ValueError("At least one saved model is required.")

    labels = [item.label for item in challenge_set]
    texts = [item.text for item in challenge_set]
    scenarios: dict[str, list[int]] = defaultdict(list)
    indicator_predictions = []
    indicator_categories_by_index = []
    for index, item in enumerate(challenge_set):
        scenarios[item.scenario].append(index)
        indicators = detect_security_risk_indicators(item.text)
        indicator_predictions.append(PHISHING if indicators else LEGITIMATE)
        indicator_categories_by_index.append(
            sorted({indicator["category"] for indicator in indicators})
        )

    model_predictions: dict[str, list[str]] = {}
    for name, model in models.items():
        predictions = list(model.predict(texts))
        if len(predictions) != len(labels) or not set(predictions).issubset(LABELS):
            raise ValueError(f"Model {name!r} returned invalid predictions.")
        model_predictions[name] = predictions

    indicator_report = _binary_metrics(labels, indicator_predictions)
    indicator_report["false_alarm_scenarios"] = {
        scenario: sum(
            indicator_predictions[index] == PHISHING
            for index in indices
            if labels[index] == LEGITIMATE
        )
        for scenario, indices in sorted(scenarios.items())
        if any(labels[index] == LEGITIMATE for index in indices)
        and any(
            labels[index] == LEGITIMATE
            and indicator_predictions[index] == PHISHING
            for index in indices
        )
    }
    indicator_report["indicator_categories"] = dict(
        sorted(
            (
                category,
                sum(category in categories for categories in indicator_categories_by_index),
            )
            for category in {
                category
                for categories in indicator_categories_by_index
                for category in categories
            }
        )
    )

    model_reports = {}
    indicators_catch_missed = {}
    for name, predictions in model_predictions.items():
        model_reports[name] = _binary_metrics(labels, predictions)
        missed_phishing = [
            index
            for index, label in enumerate(labels)
            if label == PHISHING and predictions[index] != PHISHING
        ]
        caught = [
            index
            for index in missed_phishing
            if indicator_predictions[index] == PHISHING
        ]
        indicators_catch_missed[name] = {
            "model_false_negatives": len(missed_phishing),
            "missed_phishing_with_indicator": len(caught),
            "fraction_of_model_misses_flagged": (
                len(caught) / len(missed_phishing) if missed_phishing else 0.0
            ),
            "indicator_categories_on_model_misses": dict(
                sorted(
                    (
                        category,
                        sum(
                            category in indicator_categories_by_index[index]
                            for index in caught
                        ),
                    )
                    for category in {
                        category
                        for index in caught
                        for category in indicator_categories_by_index[index]
                    }
                )
            ),
        }

    report = {
        "evaluation": "independent fictional synthetic challenge; not official research metrics",
        "challenge_size": len(challenge_set),
        "label_counts": {
            label: labels.count(label) for label in (LEGITIMATE, PHISHING)
        },
        "scenario_counts": {
            scenario: {
                label: sum(labels[index] == label for index in indices)
                for label in (LEGITIMATE, PHISHING)
            }
            for scenario, indices in sorted(scenarios.items())
        },
        "security_risk_indicators": indicator_report,
        "saved_model_metrics": model_reports,
        "indicators_on_model_misses": indicators_catch_missed,
        "model_prediction_disagreements": sum(
            model_predictions[model_names[0]][index]
            != model_predictions[model_names[1]][index]
            for index in range(len(labels))
        )
        if len(model_names := sorted(model_predictions)) == 2
        else None,
        "limitations": [
            "The independent set is small, hand-authored, and entirely synthetic.",
            "Rule coverage measures only the configured textual patterns and does not establish real-world detection performance.",
            "Indicator output is evaluated independently and does not alter model classifications.",
            "Scenario counts are small and should be interpreted descriptively.",
        ],
    }
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def run_saved_independent_challenge(
    v2_model_path: Path = DEFAULT_V2_MODEL,
    v3_model_path: Path = DEFAULT_V3_MODEL,
    results_path: Path = INDEPENDENT_RESULTS_PATH,
) -> dict:
    models = {
        "v2_grouped": joblib.load(v2_model_path),
        "v3_grouped": joblib.load(v3_model_path),
    }
    return evaluate_independent_challenge(models, results_path=results_path)
