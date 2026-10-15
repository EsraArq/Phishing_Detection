import unittest

import pandas as pd

from phishguard.clean_data import (
    normalize_email_text,
    prepare_clean_dataset,
    split_clean_dataset,
)
from phishguard.evaluation import evaluate_predictions
from phishguard.models import build_models


class CleanDataTests(unittest.TestCase):
    def test_duplicate_emails_are_normalized_and_deduplicated(self):
        source = pd.DataFrame(
            {
                "Email Text": [
                    "  Confirm   your course registration.\n",
                    "confirm your COURSE registration.",
                    "A different safe email",
                    "A different phishing email",
                ],
                "Email Type": [
                    "Safe Email",
                    "Safe Email",
                    "Safe Email",
                    "Phishing Email",
                ],
            }
        )

        cleaned, counts = prepare_clean_dataset(source)

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(counts["normalized_duplicate_groups"], 1)
        self.assertEqual(counts["duplicate_rows_removed"], 1)
        self.assertEqual(
            normalize_email_text("  Confirm   your course registration.\n"),
            normalize_email_text("confirm your COURSE registration."),
        )

    def test_placeholder_and_conflicting_label_groups_are_removed(self):
        source = pd.DataFrame(
            {
                "Email Text": [
                    None,
                    "   ",
                    "EMPTY",
                    " empty ",
                    "Same text",
                    " same   TEXT ",
                    "Safe example",
                    "Phishing example",
                ],
                "Email Type": [
                    "Safe Email",
                    "Safe Email",
                    "Phishing Email",
                    "Safe Email",
                    "Safe Email",
                    "Phishing Email",
                    "Safe Email",
                    "Phishing Email",
                ],
            }
        )

        cleaned, counts = prepare_clean_dataset(source)

        self.assertEqual(counts["missing_text_or_label_rows_removed"], 1)
        self.assertEqual(counts["placeholder_rows_removed"], 3)
        self.assertEqual(counts["conflicting_label_groups_excluded"], 2)
        self.assertEqual(counts["conflicting_label_rows_excluded"], 4)
        self.assertEqual(counts["conflicting_rows_removed_as_placeholders"], 2)
        self.assertEqual(
            counts["conflicting_label_rows_removed_after_placeholder_filter"], 2
        )
        self.assertNotIn("same text", cleaned["_normalized_text"].tolist())
        self.assertEqual(len(cleaned), 2)

    def test_normalized_email_text_never_crosses_train_test(self):
        source = pd.DataFrame(
            {
                "Email Text": [
                    *(f"Safe example {i}" for i in range(10)),
                    *(f"Phishing example {i}" for i in range(10)),
                ],
                "Email Type": [
                    *(["Safe Email"] * 10),
                    *(["Phishing Email"] * 10),
                ],
            }
        )
        cleaned, _ = prepare_clean_dataset(source)

        x_train, x_test, y_train, y_test = split_clean_dataset(
            cleaned, test_size=0.3, random_state=42
        )

        self.assertTrue(
            set(cleaned.loc[x_train.index, "_normalized_text"]).isdisjoint(
                set(cleaned.loc[x_test.index, "_normalized_text"])
            )
        )
        self.assertEqual(len(x_train), 14)
        self.assertEqual(len(x_test), 6)
        self.assertEqual(set(y_train), {"Safe Email", "Phishing Email"})
        self.assertEqual(set(y_test), {"Safe Email", "Phishing Email"})


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_explicit_safe_then_phishing_order(self):
        metrics = evaluate_predictions(
            ["Phishing Email", "Safe Email"],
            ["Phishing Email", "Phishing Email"],
        )

        self.assertEqual(metrics["labels"], ["Safe Email", "Phishing Email"])
        self.assertEqual(metrics["confusion_matrix"], [[0, 1], [0, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)


class ModelTests(unittest.TestCase):
    def test_both_models_include_tfidf(self):
        models = build_models()

        self.assertEqual(set(models), {"svm", "random_forest"})
        for model in models.values():
            self.assertIn("tfidf", model.named_steps)
            self.assertIn("classifier", model.named_steps)


if __name__ == "__main__":
    unittest.main()
