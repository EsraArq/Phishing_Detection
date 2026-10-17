import unittest

from phishguard.v3_preprocessing import (
    build_v3_tfidf_vectorizer,
    mask_sensitive_information,
)


class Version3PreprocessingTests(unittest.TestCase):
    def test_masks_identifiers_and_preserves_phishing_signals(self):
        source = (
            "URGENT: verify your credentials now. Email "
            "person.name+alerts@example.test or call +1 (212) 555-0198. "
            "Visit http://secure-login.example.test/pay and send payment "
            "for account number: 0048123499."
        )

        masked = mask_sensitive_information(source)

        for sensitive_value in (
            "person.name+alerts@example.test",
            "+1 (212) 555-0198",
            "secure-login.example.test/pay",
            "0048123499",
        ):
            self.assertNotIn(sensitive_value.casefold(), masked.casefold())
        for marker in (
            "pii_email",
            "pii_phone",
            "pii_url",
            "pii_url_http",
            "pii_account",
        ):
            self.assertIn(marker, masked)
        for signal in ("urgent", "verify", "credentials", "send payment"):
            self.assertIn(signal, masked.casefold())

    def test_masks_url_embedded_email_and_ip_as_one_overlapping_value(self):
        source = (
            "Click https://203.0.113.42/reset?email=member@example.test "
            "immediately to restore access."
        )

        masked = mask_sensitive_information(source)

        self.assertNotIn("203.0.113.42", masked)
        self.assertNotIn("member@example.test", masked)
        self.assertIn("pii_url", masked)
        self.assertIn("pii_url_https", masked)
        self.assertIn("pii_url_ip", masked)
        self.assertIn("click", masked.casefold())
        self.assertIn("restore access", masked.casefold())

    def test_masks_standalone_ip_addresses_and_account_identifiers(self):
        source = (
            "Sign in from 192.0.2.17. Customer ID is CUST-837261 and "
            "reference # 4729186. Keep the account secure."
        )

        masked = mask_sensitive_information(source)

        for sensitive_value in ("192.0.2.17", "CUST-837261", "4729186"):
            self.assertNotIn(sensitive_value.casefold(), masked.casefold())
        self.assertIn("pii_ip", masked)
        self.assertEqual(masked.count("pii_account"), 2)
        self.assertIn("account secure", masked)

    def test_masks_payment_government_and_iban_identifiers(self):
        source = (
            "Card number 4111 1111 1111 1111; SSN 123-45-6789; "
            "IBAN GB82 WEST 1234 5698 7654 32."
        )

        masked = mask_sensitive_information(source)

        for sensitive_value in (
            "4111 1111 1111 1111",
            "123-45-6789",
            "GB82 WEST 1234 5698 7654 32",
        ):
            self.assertNotIn(sensitive_value.casefold(), masked.casefold())
        self.assertIn("pii_payment", masked)
        self.assertIn("pii_gov_id", masked)
        self.assertIn("pii_iban", masked)

    def test_masks_obfuscated_addresses_defanged_urls_and_phone_extensions(self):
        source = (
            "Contact analyst [at] example [dot] test. Open "
            "hxxps://secure[.]example[.]test/login, or call "
            "(212) 555.0198 ext. 123."
        )

        masked = mask_sensitive_information(source)

        for sensitive_value in (
            "analyst [at] example [dot] test",
            "secure[.]example[.]test/login",
            "(212) 555.0198 ext. 123",
        ):
            self.assertNotIn(sensitive_value.casefold(), masked.casefold())
        self.assertIn("pii_email", masked)
        self.assertIn("pii_url", masked)
        self.assertIn("pii_url_defanged", masked)
        self.assertIn("pii_phone", masked)
        self.assertIn("open", masked.casefold())
        self.assertIn("or call", masked.casefold())

    def test_preserves_date_and_unlabeled_account_language(self):
        source = "The account review is scheduled for 2025-06-12."

        masked = mask_sensitive_information(source)

        self.assertEqual(masked, source)

    def test_masks_malformed_bracketed_ip_url_without_failing(self):
        masked = mask_sensitive_information("Visit https://[1:2:3:4:5:6:7:8:9]/login now.")

        self.assertIn("pii_url", masked)
        self.assertIn("visit", masked.casefold())
        self.assertIn("now", masked.casefold())

    def test_tfidf_tokenizer_keeps_category_markers_as_single_features(self):
        vectorizer = build_v3_tfidf_vectorizer()
        vectors = vectorizer.fit_transform(
            [
                "pii_email pii_phone pii_url pii_url_http pii_url_ip "
                "pii_account pii_payment pii_gov_id pii_iban pii_ip"
            ]
        )

        self.assertEqual(
            set(vectorizer.get_feature_names_out()),
            {
                "pii_email",
                "pii_phone",
                "pii_url",
                "pii_url_http",
                "pii_url_ip",
                "pii_account",
                "pii_payment",
                "pii_gov_id",
                "pii_iban",
                "pii_ip",
            },
        )
        self.assertEqual(vectors.shape[1], 10)

    def test_rejects_non_text_input(self):
        with self.assertRaises(TypeError):
            mask_sensitive_information(None)


if __name__ == "__main__":
    unittest.main()
