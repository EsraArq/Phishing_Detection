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
        self.assertEqual(indicators[0]["category"], "Office macro instruction")
        self.assertIn("enable Excel macros", indicators[0]["evidence"])
        self.assertIn("enable Office or Excel macros", indicators[0]["explanation"])

    def test_detects_password_disclosure_and_link_login_requests(self):
        email = (
            "Reply with your password to confirm your account. Click the link "
            "https://example.invalid/login and enter your login credentials."
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


if __name__ == "__main__":
    unittest.main()
