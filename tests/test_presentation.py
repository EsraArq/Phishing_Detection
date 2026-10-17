import unittest

from phishguard.presentation import build_analysis_presentation


class AnalysisPresentationTests(unittest.TestCase):
    def test_legitimate_prediction_with_indicators_shows_prominent_caution(self):
        presentation = build_analysis_presentation(
            "Safe Email",
            [
                {
                    "category": "Password or credential disclosure",
                    "explanation": "Original rule explanation.",
                    "evidence": "fictional submitted text",
                },
                {
                    "category": "QR code requests sign-in or payment",
                    "explanation": "Original rule explanation.",
                    "evidence": "another fictional excerpt",
                },
            ],
        )

        self.assertEqual(presentation["model_result"], "Likely Legitimate Email")
        self.assertFalse(presentation["is_phishing"])
        self.assertIn("may still be unsafe", presentation["caution"])
        self.assertEqual(
            [item["category"] for item in presentation["indicators"]],
            [
                "Password or credential disclosure",
                "QR code requests sign-in or payment",
            ],
        )
        self.assertGreaterEqual(len(presentation["recommended_actions"]), 2)
        self.assertNotIn("fictional submitted text", repr(presentation))
        self.assertNotIn("another fictional excerpt", repr(presentation))

    def test_phishing_prediction_and_indicators_remain_separate(self):
        presentation = build_analysis_presentation(
            "Phishing Email",
            [
                {
                    "category": "Attachment active-content instruction",
                    "explanation": "Ignored dynamic explanation.",
                    "evidence": "private excerpt",
                }
            ],
        )

        self.assertEqual(presentation["model_result"], "Potential Phishing Email")
        self.assertTrue(presentation["is_phishing"])
        self.assertIsNone(presentation["caution"])
        self.assertIn(
            "Attachment active-content instruction",
            [item["category"] for item in presentation["indicators"]],
        )

    def test_no_indicators_does_not_imply_safety(self):
        presentation = build_analysis_presentation("Safe Email", [])

        self.assertEqual(presentation["indicators"], [])
        self.assertIn(
            "does not guarantee",
            presentation["no_indicators_message"],
        )
        self.assertTrue(presentation["recommended_actions"])

    def test_unknown_model_label_raises_generic_error(self):
        with self.assertRaisesRegex(ValueError, "unsupported classification"):
            build_analysis_presentation("untrusted model output", [])

    def test_unknown_indicator_category_is_summarized_without_echoing_input(self):
        presentation = build_analysis_presentation(
            "Phishing Email",
            [{"category": "private source text", "evidence": "secret"}],
        )

        self.assertEqual(
            presentation["indicators"][0]["category"],
            "Additional security risk indicator",
        )
        self.assertNotIn("private source text", repr(presentation))
        self.assertNotIn("secret", repr(presentation))


if __name__ == "__main__":
    unittest.main()
