import json
import tempfile
import unittest
from pathlib import Path

from phishguard.clean_data import normalize_email_text
from phishguard.data import LABELS
from phishguard.independent_challenge_evaluation import (
    evaluate_independent_challenge,
)
from phishguard.independent_synthetic_challenge import (
    INDEPENDENT_SYNTHETIC_CHALLENGE_SET,
)
from phishguard.risk_indicators import detect_security_risk_indicators
from phishguard.synthetic_challenge import (
    LEGITIMATE,
    PHISHING,
    SYNTHETIC_CHALLENGE_SET,
)


class FixedModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, texts):
        if len(texts) != len(self.predictions):
            raise AssertionError("Unexpected independent challenge input size.")
        return self.predictions


class IndependentChallengeEvaluationTests(unittest.TestCase):
    def test_challenge_is_balanced_fictional_and_independent(self):
        challenge = INDEPENDENT_SYNTHETIC_CHALLENGE_SET
        self.assertEqual(len(challenge), 60)
        self.assertEqual(sum(item.label == LEGITIMATE for item in challenge), 30)
        self.assertEqual(sum(item.label == PHISHING for item in challenge), 30)
        self.assertFalse(any("@" in item.text for item in challenge))
        self.assertTrue(
            all(
                ".example.test" in token
                for item in challenge
                for token in item.text.split()
                if token.startswith(("http://", "https://"))
            )
        )
        old_messages = {
            normalize_email_text(item.text) for item in SYNTHETIC_CHALLENGE_SET
        }
        self.assertFalse(
            any(normalize_email_text(item.text) in old_messages for item in challenge)
        )
        self.assertGreaterEqual(len({item.scenario for item in challenge}), 10)

    def test_indicator_metrics_and_model_miss_overlap_are_aggregate_only(self):
        labels = [item.label for item in INDEPENDENT_SYNTHETIC_CHALLENGE_SET]
        v2_predictions = labels.copy()
        v3_predictions = labels.copy()
        v2_predictions[0] = PHISHING
        v2_predictions[30] = LEGITIMATE
        v3_predictions[1] = PHISHING
        v3_predictions[31] = LEGITIMATE
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "metrics.json"
            report = evaluate_independent_challenge(
                {
                    "v2_grouped": FixedModel(v2_predictions),
                    "v3_grouped": FixedModel(v3_predictions),
                },
                results_path=output_path,
            )
            saved_text = output_path.read_text(encoding="utf-8")

        indicator_labels = [
            PHISHING
            if detect_security_risk_indicators(item.text)
            else LEGITIMATE
            for item in INDEPENDENT_SYNTHETIC_CHALLENGE_SET
        ]
        self.assertEqual(
            report["security_risk_indicators"]["samples"], len(labels)
        )
        self.assertEqual(
            report["security_risk_indicators"]["true_positives"],
            sum(a == PHISHING and p == PHISHING for a, p in zip(labels, indicator_labels)),
        )
        self.assertEqual(
            report["saved_model_metrics"]["v2_grouped"]["false_positives"], 1
        )
        self.assertEqual(
            report["saved_model_metrics"]["v2_grouped"]["false_negatives"], 1
        )
        self.assertEqual(
            report["indicators_on_model_misses"]["v2_grouped"][
                "model_false_negatives"
            ],
            1,
        )
        self.assertIsNotNone(report["model_prediction_disagreements"])
        self.assertNotIn(INDEPENDENT_SYNTHETIC_CHALLENGE_SET[0].text, saved_text)
        self.assertEqual(set(LABELS), {LEGITIMATE, PHISHING})
        self.assertEqual(json.loads(saved_text)["challenge_size"], 60)

    def test_rule_evaluation_is_independent_of_model_outputs(self):
        labels = [item.label for item in INDEPENDENT_SYNTHETIC_CHALLENGE_SET]
        all_safe = [LEGITIMATE] * len(labels)
        report = evaluate_independent_challenge(
            {"v2_grouped": FixedModel(all_safe)}
        )

        self.assertEqual(
            report["security_risk_indicators"],
            evaluate_independent_challenge(
                {"v2_grouped": FixedModel(labels)}
            )["security_risk_indicators"],
        )
        self.assertNotEqual(
            report["saved_model_metrics"]["v2_grouped"],
            evaluate_independent_challenge(
                {"v2_grouped": FixedModel(labels)}
            )["saved_model_metrics"]["v2_grouped"],
        )


if __name__ == "__main__":
    unittest.main()
