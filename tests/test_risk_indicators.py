import unittest

from phishguard.risk_indicators import detect_security_risk_indicators


class SecurityRiskIndicatorTests(unittest.TestCase):
    def test_detects_macro_enable_instructions_with_evidence(self):
        email = (
            "Open the attached payroll workbook and enable Excel macros "
            "to view your updated pay details."
        )

        indicators = detect_security_risk_indicators(email)

        self.assertEqual(len(indicators), 1)
        self.assertEqual(
            indicators[0]["category"], "Attachment active-content instruction"
        )
        self.assertIn("enable Excel macros", indicators[0]["evidence"])
        self.assertIn("attachment", indicators[0]["explanation"])

    def test_detects_password_disclosure_and_link_login_requests(self):
        email = (
            "Reply with your password to confirm your account. Click the link "
            "https://example.invalid/login and enter your login credentials. "
            "Then use the verification page when entering your current password."
        )

        indicators = detect_security_risk_indicators(email)

        categories = {indicator["category"] for indicator in indicators}
        self.assertEqual(
            categories,
            {
                "Password or credential disclosure",
                "Credentials requested through a link",
            },
        )
        self.assertTrue(all(indicator["evidence"] for indicator in indicators))

    def test_detects_reply_requests_for_password_and_security_codes(self):
        emails = (
            "URGENT: Your account will be suspended within 24 hours. To prevent "
            "losing access, reply immediately with your password and MFA "
            "verification code.",
            "Reply with your password to restore access to your fictional account.",
            "Reply promptly with your MFA code to verify your fictional sign-in.",
            "Please reply directly with the verification code to continue.",
            "Reply now with your recovery code to unlock the fictional account.",
        )

        for email in emails:
            with self.subTest(email=email):
                categories = {
                    item["category"]
                    for item in detect_security_risk_indicators(email)
                }
                self.assertIn("Password or credential disclosure", categories)

    def test_security_awareness_mentions_do_not_trigger_credential_indicator(self):
        emails = (
            "Security awareness reminder: passwords and MFA verification codes "
            "are secrets; never share them in a message.",
            "Security training note: recovery codes and passwords should be "
            "stored securely. Legitimate support will not ask for MFA codes.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertNotIn(
                    "Password or credential disclosure",
                    {
                        item["category"]
                        for item in detect_security_risk_indicators(email)
                    },
                )

    def test_legitimate_and_negated_security_guidance_does_not_trigger(self):
        email = (
            "Your course registration is confirmed. For your safety, do not "
            "enable macros, never share your password, and do not enter "
            "credentials through links."
        )

        self.assertEqual(detect_security_risk_indicators(email), [])

    def test_no_indicator_for_routine_course_registration(self):
        email = (
            "Your security course registration is confirmed. The agenda and "
            "attendee details are available in the learning portal."
        )

        self.assertEqual(detect_security_risk_indicators(email), [])

    def test_detects_requests_for_mfa_and_account_recovery_codes(self):
        email = (
            "Reply with your MFA code and send us your account recovery code "
            "to restore access."
        )

        indicators = detect_security_risk_indicators(email)

        self.assertEqual(
            {item["category"] for item in indicators},
            {"Password or credential disclosure"},
        )

    def test_does_not_flag_legitimate_reset_or_security_code_guidance(self):
        emails = (
            "Your Example password reset was requested. Use the official account "
            "page to choose a new password; support will never ask you to share "
            "your password or MFA code.",
            "Your sign-in verification code is ready. Enter it only on the sign-in "
            "page you opened; we will never ask you to send it by email.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertEqual(detect_security_risk_indicators(email), [])

    def test_detects_qr_code_sign_in_and_payment_requests(self):
        emails = (
            "Scan the QR code to sign in to your Example account.",
            "Use the QR code to pay the overdue fictional parking fee.",
            "An email from Example Cafe says to scan its QR code and enter "
            "card details to complete the remote payment.",
        )

        for email in emails:
            with self.subTest(email=email):
                indicators = detect_security_risk_indicators(email)
                self.assertIn(
                    "QR code requests sign-in or payment",
                    {item["category"] for item in indicators},
                )

    def test_does_not_flag_qr_code_for_unrelated_or_routine_tasks(self):
        emails = (
            "Scan the QR code to view the fictional lunch menu.",
            "Your shipping confirmation includes a QR code for parcel tracking.",
            "At the Example Museum counter, scan the printed QR to pay in person "
            "beside the card terminal.",
            "The QR opens the Example app listing, not a sign-in page.",
            "The receipt QR code is not used for payment or account access.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertEqual(detect_security_risk_indicators(email), [])

    def test_qr_negation_does_not_suppress_a_separate_suspicious_request(self):
        email = (
            "We will never ask for payment details. Separately, scan the QR "
            "code in this notice to enter your account password on the login page."
        )

        indicators = detect_security_risk_indicators(email)

        self.assertIn(
            "QR code requests sign-in or payment",
            {item["category"] for item in indicators},
        )

    def test_routine_in_person_context_does_not_trust_business_names_alone(self):
        emails = (
            "Example Cafe emailed a QR code. Scan it to enter your card details "
            "on the remote checkout page.",
            "At Example Cafe, scan the table QR code to pay in person at the "
            "cashier's card terminal.",
        )

        self.assertIn(
            "QR code requests sign-in or payment",
            {
                item["category"]
                for item in detect_security_risk_indicators(emails[0])
            },
        )
        self.assertNotIn(
            "QR code requests sign-in or payment",
            {
                item["category"]
                for item in detect_security_risk_indicators(emails[1])
            },
        )

    def test_detects_attachment_macro_and_active_file_instructions(self):
        emails = (
            "Open the attached payroll workbook and enable Excel macros.",
            "Open Example Security Update.hta and approve the prompt.",
            "The attached document says to enable active content to view the notice.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertIn(
                    "Attachment active-content instruction",
                    {item["category"] for item in detect_security_risk_indicators(email)},
                )

    def test_does_not_flag_ordinary_or_explicitly_safe_attachments(self):
        emails = (
            "The attached invoice PDF is available for your records.",
            "Do not enable macros in the attached payroll workbook.",
            "The security update is a read-only document; no active content is needed.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertEqual(detect_security_risk_indicators(email), [])

    def test_detects_payment_redirection_and_gift_card_requests(self):
        emails = (
            "Our supplier changed banks. Urgently wire the invoice to the "
            "replacement account below.",
            "Purchase gift cards and send the codes to complete the payroll refund.",
            "Return the difference using this gift card code request.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertIn(
                    "Payment redirection or gift-card request",
                    {item["category"] for item in detect_security_risk_indicators(email)},
                )

    def test_does_not_flag_legitimate_invoice_or_shipping_payment(self):
        emails = (
            "Example invoice EX-302 is due next week under the existing payment "
            "terms; review it in the usual customer portal.",
            "Your shipping confirmation is ready. No payment or account changes "
            "are required.",
        )

        for email in emails:
            with self.subTest(email=email):
                self.assertEqual(detect_security_risk_indicators(email), [])

    def test_flags_delivery_release_fee_but_not_shipping_tracking(self):
        positive = (
            "Your parcel is on hold. Pay the release charge at "
            "https://parcel.example.test/release."
        )
        negative = (
            "Your parcel has shipped. Track delivery at "
            "https://tracking.example.test/order."
        )
        negated = (
            "Your parcel is on hold. Do not pay a release fee through a link; "
            "contact the carrier through its official site."
        )

        self.assertIn(
            "Delivery release-fee payment request",
            {item["category"] for item in detect_security_risk_indicators(positive)},
        )
        self.assertNotIn(
            "Delivery release-fee payment request",
            {item["category"] for item in detect_security_risk_indicators(negative)},
        )
        self.assertNotIn(
            "Delivery release-fee payment request",
            {item["category"] for item in detect_security_risk_indicators(negated)},
        )

    def test_urgent_account_threat_requires_credential_or_payment_request(self):
        positive = (
            "Final alert: your account will be locked. Verify your password "
            "immediately to keep access."
        )
        negatives = (
            "Final alert: your account will be locked. Contact support through "
            "the official website; do not share your password.",
            "Urgent: the invoice is due today. Pay it through the usual portal; "
            "your account is not at risk.",
            "Your Example security notification reports a new sign-in. No action "
            "is required.",
        )

        categories = {
            item["category"] for item in detect_security_risk_indicators(positive)
        }
        self.assertIn(
            "Urgent account threat with credential or payment request", categories
        )
        for email in negatives:
            with self.subTest(email=email):
                self.assertNotIn(
                    "Urgent account threat with credential or payment request",
                    {
                        item["category"]
                        for item in detect_security_risk_indicators(email)
                    },
                )


if __name__ == "__main__":
    unittest.main()
