import json
import tempfile
import unittest
from pathlib import Path

from phishguard.synthetic_challenge import (
    LEGITIMATE,
    PHISHING,
    SYNTHETIC_CHALLENGE_SET,
)
from phishguard.synthetic_challenge_evaluation import evaluate_challenge_models


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, texts):
        if len(texts) != len(self.predictions):
            raise AssertionError("Unexpected challenge input size.")
        return self.predictions


class SyntheticChallengeEvaluationTests(unittest.TestCase):
    def test_challenge_has_forty_fictional_labeled_messages(self):
        self.assertEqual(len(SYNTHETIC_CHALLENGE_SET), 40)
        self.assertEqual(
            sum(item.label == LEGITIMATE for item in SYNTHETIC_CHALLENGE_SET), 20
        )
        self.assertEqual(
            sum(item.label == PHISHING for item in SYNTHETIC_CHALLENGE_SET), 20
        )
        self.assertFalse(any("@" in item.text for item in SYNTHETIC_CHALLENGE_SET))
        self.assertTrue(
            all(
                ".example.test" in token
                for item in SYNTHETIC_CHALLENGE_SET
                for token in item.text.split()
                if token.startswith(("http://", "https://"))
            )
        )

    def test_reports_aggregate_model_and_scenario_metrics_and_disagreements(self):
        labels = [item.label for item in SYNTHETIC_CHALLENGE_SET]
        v2_predictions = labels.copy()
        v3_predictions = labels.copy()
        v2_predictions[0] = PHISHING
        v3_predictions[20] = LEGITIMATE
        models = {
            "v2_grouped": FixedPredictionModel(v2_predictions),
            "v3_grouped": FixedPredictionModel(v3_predictions),
        }
        training_texts = ["unrelated fictional training notice"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "challenge" / "metrics.json"
            report = evaluate_challenge_models(
                models,
                training_texts=training_texts,
                results_path=results_path,
            )
            saved_report = json.loads(results_path.read_text(encoding="utf-8"))
            saved_text = results_path.read_text(encoding="utf-8")

        self.assertEqual(report["challenge_size"], 40)
        self.assertEqual(report["models"]["v2_grouped"]["overall"]["false_positives"], 1)
        self.assertEqual(report["models"]["v3_grouped"]["overall"]["false_negatives"], 1)
        self.assertEqual(report["prediction_disagreements"]["total_disagreements"], 2)
        self.assertEqual(
            report["training_duplicate_check"]["matching_challenge_messages"],
            {"exact": 0, "normalized": 0},
        )
        self.assertIn(
            "course_registration",
            report["models"]["v2_grouped"]["by_scenario"],
        )
        self.assertNotIn(SYNTHETIC_CHALLENGE_SET[0].text, saved_text)
        self.assertEqual(saved_report["evaluation"], report["evaluation"])

    def test_normalized_training_duplicate_is_counted_without_echoing_text(self):
        challenge = (SYNTHETIC_CHALLENGE_SET[0],)
        report = evaluate_challenge_models(
            {"v2": FixedPredictionModel([LEGITIMATE] * 40)},
            challenge_set=challenge * 40,
            training_texts=[challenge[0].text.upper()],
        )

        self.assertEqual(
            report["training_duplicate_check"]["matching_challenge_messages"][
                "normalized"
            ],
            40,
        )
        self.assertNotIn("Riley Example", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
