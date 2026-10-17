"""Fictional, offline-only challenge emails for model evaluation."""

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class ChallengeEmail:
    """One wholly fictional challenge message and its expected labels."""

    label: str
    scenario: str
    text: str


LEGITIMATE = "Safe Email"
PHISHING = "Phishing Email"

SYNTHETIC_CHALLENGE_SET = (
    ChallengeEmail(
        LEGITIMATE,
        "course_registration",
        "Hello Riley Example, your registration for the fictional course "
        "BIO-204 at Example College is confirmed for September 14. View the "
        "schedule in the campus portal at https://portal.example.test/courses.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "course_registration",
        "Your place in the Example Institute evening writing seminar is reserved. "
        "The first class meets in Room 12B; no payment is due today.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "course_registration",
        "Hi Morgan Sample, the registration office received your request for "
        "Course EX-318. Your fictional enrollment reference is REG-EX-2048.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "course_registration",
        "Example Academy confirms your spring workshop enrollment. Bring your "
        "student card to the welcome desk; the course materials are included.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "A password reset was requested for your Example Learning account. If "
        "this was you, visit https://accounts.example.test/reset and choose a "
        "new password. The reset request expires in 30 minutes.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "Example Portal received your request to change your password. If you "
        "did not make this request, ignore this notice and contact support "
        "through the number printed on your membership card.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "Your Example Library password has been updated successfully. No action "
        "is needed if you made this change.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "invoice",
        "Invoice EX-4102 from Example Office Supply is ready. The total is "
        "$28.40, due October 30. Review the itemized statement at "
        "https://billing.example.test/invoices/EX-4102.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "invoice",
        "Example Workshop attached the fictional invoice summary for order "
        "INV-EX-731. Payment is due under your usual 30-day terms; bank details "
        "have not changed.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "invoice",
        "This is a reminder that Example Printing invoice 6038 is due next "
        "week. You can also review it in the customer portal at "
        "https://billing.example.test/account/invoices.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "payroll_notification",
        "Example Works payroll: your October payslip is available in the employee "
        "portal at https://staff.example.test/payroll. Pay date is October 25.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "payroll_notification",
        "Your Example Research Cooperative direct-deposit statement is ready. "
        "This routine notice does not request a reply or updated banking details.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "payroll_notification",
        "Payroll calendar update: Example Studio will process the monthly pay "
        "run on Friday. No action is required from employees.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "Example Parcel confirms shipment EX-SHIP-5502. The package is expected "
        "Thursday. Track it at https://tracking.example.test/EX-SHIP-5502.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "Your Example Books order has shipped. The carrier has the parcel and "
        "will provide an estimated delivery window after its next scan.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "Delivery confirmation: the fictional supply order EX-8821 was left at "
        "the reception desk. Contact the store using its printed receipt if "
        "you need help.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Example Account Security noticed a sign-in from a new browser. If you "
        "recognize it, no action is needed. Review recent sessions at "
        "https://security.example.test/sessions.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Routine security notice from Example Workspace: multifactor "
        "authentication was enabled for your account. We will never ask you "
        "to send a password by email.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Example Forum blocked several unsuccessful sign-in attempts. Your "
        "account remains active. Visit the site directly if you want to review "
        "your security settings.",
    ),
    ChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Your one-time sign-in code for Example Service is 482-731. Enter it "
        "only on the sign-in page you opened; support will not ask you to reply "
        "with this code.",
    ),
    ChallengeEmail(
        PHISHING,
        "impersonation",
        "URGENT: Example Bank security team detected suspicious activity. "
        "Confirm your identity immediately at https://verify.example.test/bank "
        "or your account will be permanently closed.",
    ),
    ChallengeEmail(
        PHISHING,
        "impersonation",
        "This is Casey Fiction from the Example College registrar. We lost the "
        "enrollment records. Reply with your student password and date of birth "
        "so I can restore your course access today.",
    ),
    ChallengeEmail(
        PHISHING,
        "impersonation",
        "Example Parcel claims your delivery is on hold. Pay a fictional "
        "$1.25 release charge at https://parcel-fee.example.test/release now.",
    ),
    ChallengeEmail(
        PHISHING,
        "credential_theft",
        "Your Example Workspace mailbox will be disabled in 15 minutes. Sign in "
        "at https://login-check.example.test/mail and enter your password to "
        "keep access.",
    ),
    ChallengeEmail(
        PHISHING,
        "credential_theft",
        "We are validating Example Learning accounts. Send your username, "
        "password, and one-time code in a reply to prevent suspension.",
    ),
    ChallengeEmail(
        PHISHING,
        "credential_theft",
        "The shared document cannot be viewed until you authenticate. Open "
        "https://docs-auth.example.test/view and type your organization "
        "credentials.",
    ),
    ChallengeEmail(
        PHISHING,
        "credential_theft",
        "Your account sign-in failed. Restore access now by entering your "
        "current password at https://secure-session.example.test/restore.",
    ),
    ChallengeEmail(
        PHISHING,
        "payment_fraud",
        "Our supplier changed banks. Urgently wire the outstanding Example "
        "invoice balance to the replacement account below and do not call the "
        "usual contact to verify.",
    ),
    ChallengeEmail(
        PHISHING,
        "payment_fraud",
        "Final payment warning: send $840 today to fictional account "
        "EX-ACCT-884211 or legal action will begin. Reply with the transfer "
        "receipt immediately.",
    ),
    ChallengeEmail(
        PHISHING,
        "payment_fraud",
        "Example Payroll overpaid you. Return the difference using this gift "
        "card code request; do not discuss it with your manager.",
    ),
    ChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "The fictional purchase order is attached as PO-EX-1042.zip. Enable "
        "macros in the spreadsheet to view the urgent revised payment details.",
    ),
    ChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "Please open the attached Example Security Update.hta and approve the "
        "prompt to restore your mailbox. This action is required immediately.",
    ),
    ChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "An overdue invoice file is attached as Invoice-EX-551.docm. Enable "
        "content and enter your credentials when the document requests them.",
    ),
    ChallengeEmail(
        PHISHING,
        "qr_code_scam",
        "Scan the QR code in the attached image to keep your Example account "
        "active. The code opens https://qr-login.example.test/auth and asks "
        "for your password.",
    ),
    ChallengeEmail(
        PHISHING,
        "qr_code_scam",
        "A QR code is required to claim your fictional payroll bonus. Scan now "
        "and submit your bank login before midnight.",
    ),
    ChallengeEmail(
        PHISHING,
        "qr_code_scam",
        "Parking citation EX-CITE-1902 is overdue. Scan this email QR code to "
        "pay immediately and avoid vehicle registration suspension.",
    ),
    ChallengeEmail(
        PHISHING,
        "urgent_account_warning",
        "FINAL ALERT: unusual activity will lock your Example account in ten "
        "minutes. Verify your password and payment card at "
        "https://account-lock.example.test/verify.",
    ),
    ChallengeEmail(
        PHISHING,
        "urgent_account_warning",
        "Your Example email storage is full and incoming messages will be "
        "deleted. Sign in at https://storage-fix.example.test now to prevent "
        "permanent data loss.",
    ),
    ChallengeEmail(
        PHISHING,
        "urgent_account_warning",
        "Security Alert: your account has been flagged. Reply with your "
        "password and recovery code immediately; do not wait for the help desk.",
    ),
    ChallengeEmail(
        PHISHING,
        "urgent_account_warning",
        "Your fictional subscription is suspended. Pay the overdue balance "
        "within one hour at https://renew-now.example.test/pay or lose access.",
    ),
)


def validate_challenge_set() -> None:
    """Fail fast if labels, counts, or fictional-domain safeguards regress."""
    if len(SYNTHETIC_CHALLENGE_SET) != 40:
        raise ValueError("The synthetic challenge set must contain exactly 40 emails.")
    label_counts = {
        label: sum(item.label == label for item in SYNTHETIC_CHALLENGE_SET)
        for label in (LEGITIMATE, PHISHING)
    }
    if label_counts != {LEGITIMATE: 20, PHISHING: 20}:
        raise ValueError("The challenge set must contain 20 examples per label.")
    if any("@" in item.text for item in SYNTHETIC_CHALLENGE_SET):
        raise ValueError("Challenge text must not contain email addresses.")
    for item in SYNTHETIC_CHALLENGE_SET:
        for token in item.text.split():
            if token.startswith(("http://", "https://")):
                hostname = urlparse(token.rstrip(".,;:!?)]}")).hostname or ""
                if not hostname.endswith(".example.test"):
                    raise ValueError("Challenge URLs must use reserved example.test domains.")
    if len({item.scenario for item in SYNTHETIC_CHALLENGE_SET}) < 10:
        raise ValueError("Challenge set must cover all documented scenario categories.")


validate_challenge_set()
