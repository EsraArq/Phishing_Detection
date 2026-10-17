import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from phishguard.models import build_models
from phishguard.v3_experiment import build_v3_model, run_v3_experiment
from phishguard.v3_preprocessing import mask_sensitive_information


def _synthetic_dataset() -> pd.DataFrame:
    safe_messages = [
        f"Your library course schedule is available for class {index}."
        for index in range(12)
    ]
    phishing_messages = [
        f"Urgent verify your password and payment account at http://site{index}.test"
        for index in range(12)
    ]
    texts = safe_messages + phishing_messages
    labels = ["Safe Email"] * len(safe_messages) + ["Phishing Email"] * len(
        phishing_messages
    )
    texts.extend(
        [
            safe_messages[0].upper(),
            "Urgent verify your password and payment account at "
            "http://site0.test",
            "Urgent verify your password and payment account at "
            "http://site11.test",
        ]
    )
    labels.extend(["Safe Email", "Phishing Email", "Safe Email"])
    return pd.DataFrame({"Email Text": texts, "Email Type": labels})


class Version3ExperimentTests(unittest.TestCase):
    def test_models_use_matching_svm_parameters_and_v3_masks_at_inference(self):
        v2 = build_models()["svm"]
        v3 = build_v3_model()

        self.assertEqual(
            {
                key: v2.named_steps["classifier"].get_params()[key]
                for key in ("C", "gamma", "kernel", "class_weight")
            },
            {
                key: v3.named_steps["classifier"].get_params()[key]
                for key in ("C", "gamma", "kernel", "class_weight")
            },
        )
        self.assertEqual(
            mask_sensitive_information("contact user@example.test"),
            "contact pii_email",
        )
        self.assertEqual(v3.named_steps["mask"].func.__name__, "mask_email_batch")

    def test_experiment_uses_one_leakage_safe_split_and_saves_reloadable_pipeline(self):
        source_data = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_directory = root / "models" / "v3" / "experiment"
            results_path = root / "results" / "v3" / "experiment" / "metrics.json"

            result = run_v3_experiment(source_data, model_directory, results_path)

            self.assertEqual(result["split"]["random_state"], 42)
            self.assertEqual(result["split"]["test_size"], 0.3)
            self.assertEqual(result["split"]["training_test_normalized_text_overlap"], 0)
            self.assertGreater(result["split"]["cleaning_counts"]["duplicate_rows_removed"], 0)
            self.assertGreater(
                result["split"]["cleaning_counts"][
                    "conflicting_label_groups_remaining_after_placeholder_filter"
                ],
                0,
            )
            self.assertTrue(result["artifacts"]["v3_save_reload_prediction_match"])
            self.assertEqual(
                result["prediction_differences"]["agreements"]
                + result["prediction_differences"]["disagreements"],
                result["split"]["test_samples"],
            )
            self.assertTrue(results_path.is_file())
            reloaded = joblib.load(model_directory / "v3_masked.joblib")
            self.assertEqual(
                reloaded.predict(["urgent verify account at user@example.test"])[0],
                "Phishing Email",
            )
            self.assertTrue(
                callable(reloaded.named_steps["mask"].func),
            )

    def test_saved_metrics_contain_no_message_text(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results_path = root / "results" / "metrics.json"
            result = run_v3_experiment(
                _synthetic_dataset(),
                root / "models",
                results_path,
            )
            serialized = results_path.read_text(encoding="utf-8")

        self.assertNotIn("library course schedule", serialized)
        self.assertNotIn("password and payment account at", serialized)
        self.assertIn("pr_auc_average_precision", result["metrics"]["v3_masked"])


if __name__ == "__main__":
    unittest.main()
