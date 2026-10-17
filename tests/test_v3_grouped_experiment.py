import tempfile
import unittest
from pathlib import Path

import pandas as pd

from phishguard.v3_grouped_experiment import (
    group_similar_messages,
    run_grouped_v3_experiment,
    split_by_similarity_groups,
)


class SimilarityGroupingTests(unittest.TestCase):
    def test_groups_exact_normalized_and_highly_similar_messages(self):
        texts = [
            "URGENT verify your password at secure.example/login now",
            "urgent   VERIFY your password at secure.example/login now",
            "Urgent verify your password at secure.example/login now today",
            "Your weekly course schedule is ready for review",
        ]

        groups, statistics = group_similar_messages(texts, similarity_threshold=0.75)

        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])
        self.assertEqual(statistics["multi_message_groups"], 1)
        self.assertGreater(statistics["verified_similar_pairs"], 0)

    def test_group_split_is_deterministic_balanced_and_has_no_group_overlap(self):
        labels = ["Safe Email", "Phishing Email"] * 50
        groups = [index // 2 for index in range(len(labels))]

        train_a, test_a, stats_a = split_by_similarity_groups(
            labels, groups, test_size=0.3, random_state=42
        )
        train_b, test_b, stats_b = split_by_similarity_groups(
            labels, groups, test_size=0.3, random_state=42
        )

        self.assertEqual(train_a.tolist(), train_b.tolist())
        self.assertEqual(test_a.tolist(), test_b.tolist())
        self.assertEqual(stats_a, stats_b)
        self.assertEqual(
            set(groups[index] for index in train_a)
            & set(groups[index] for index in test_a),
            set(),
        )
        self.assertAlmostEqual(stats_a["test_fraction"], 0.3, delta=0.03)
        self.assertEqual(
            stats_a["test_class_counts"]["Safe Email"],
            stats_a["test_class_counts"]["Phishing Email"],
        )

    def test_grouped_experiment_keeps_identical_inputs_and_saves_aggregate_only(self):
        safe = [
            f"Your class schedule is available for course group {index}."
            for index in range(20)
        ]
        phishing = [
            f"Urgent verify your account password at secure{index}.example/login."
            for index in range(20)
        ]
        source = pd.DataFrame(
            {
                "Email Text": safe + phishing,
                "Email Type": ["Safe Email"] * 20 + ["Phishing Email"] * 20,
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            previous_path = root / "previous.json"
            previous_path.write_text(
                '{"metrics": {"v2_baseline": {}, "v3_masked": {}}}',
                encoding="utf-8",
            )
            report = run_grouped_v3_experiment(
                source,
                root / "models" / "grouped",
                root / "results" / "grouped" / "metrics.json",
                previous_path,
            )
            saved = (root / "results" / "grouped" / "metrics.json").read_text(
                encoding="utf-8"
            )

        self.assertEqual(report["split"]["cross_split_group_overlap"], 0)
        self.assertLessEqual(abs(report["split"]["test_fraction"] - 0.3), 0.03)
        self.assertIn("false_negatives", report["metrics"]["v2_baseline"])
        self.assertIn("pr_auc_average_precision", report["metrics"]["v3_masked"])
        self.assertTrue(report["artifacts"]["v3_save_reload_prediction_match"])
        self.assertNotIn("class schedule is available", saved)
        self.assertNotIn("password at secure", saved)

    def test_rejects_invalid_similarity_threshold(self):
        with self.assertRaises(ValueError):
            group_similar_messages(["one message"], similarity_threshold=0)


if __name__ == "__main__":
    unittest.main()
