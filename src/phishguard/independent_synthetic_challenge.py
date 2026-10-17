"""Independent fictional challenge emails for rule and model evaluation."""

from dataclasses import dataclass
from urllib.parse import urlparse

from phishguard.synthetic_challenge import LEGITIMATE, PHISHING


@dataclass(frozen=True)
class IndependentChallengeEmail:
    label: str
    scenario: str
    text: str


INDEPENDENT_SYNTHETIC_CHALLENGE_SET = (
    IndependentChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Example Workspace noticed a sign-in from a browser you have not used "
        "before. Review the activity from your saved bookmark if you do not "
        "recognize it.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "A new recovery device was added to your Example account. This message "
        "is informational; the security team will never ask you to send a code.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Example Cloud blocked several failed sign-in attempts. Your profile "
        "remains available and no response is needed.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Security notice: a passkey was registered for Example Notes. If this "
        "was not you, open the app directly and review devices.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "security_alert",
        "Example Forum completed its scheduled security maintenance. Your "
        "account settings and access are unchanged.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "We received a request to reset your Example Travel password. Continue "
        "from the reset page you opened yourself; ignore this notice if you "
        "did not make the request.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "Your Example Market password was changed successfully. For your "
        "protection, support will not ask you to share your new password.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "Password recovery for Example Reader is ready. Use the button inside "
        "the app to set a replacement password.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "password_reset",
        "Example Learning sent this routine password reset confirmation. "
        "No reply is required; if you did not start it, visit the site directly.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "invoice",
        "Example Garden Center invoice EX-GC-408 is available in your account. "
        "The balance is payable under the standard net-30 terms.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "invoice",
        "Attached is the PDF receipt for your Example Print order. It records "
        "the payment already processed at checkout.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "invoice",
        "Example Studio statement for September is ready. No bank information "
        "has changed; use your usual billing portal for questions.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "invoice",
        "Your annual Example Archive subscription invoice is due next month. "
        "Review the itemized charges from the account page you normally use.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "qr_payment",
        "At the Example Museum desk, scan the printed QR code to pay the "
        "admission fee in person. The code is posted beside the card terminal.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "qr_payment",
        "Example Cafe offers an optional table QR menu. To settle the bill, "
        "you may also pay the cashier using the usual card terminal.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "qr_payment",
        "Your Example Transit pass can be renewed through the official app. "
        "The station poster QR code opens the app store listing, not a sign-in page.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "qr_payment",
        "Example Books placed a QR code on the printed receipt for loyalty "
        "points. It does not request account or payment details.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "attachment_notification",
        "The monthly Example Housing statement is attached as a read-only PDF "
        "for your records.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "attachment_notification",
        "Example Lab shared the meeting agenda document. Open it in the "
        "organization's viewer; macros and active content are not needed.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "attachment_notification",
        "Your Example Arts membership receipt is attached. This standard "
        "document contains no account action request.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "attachment_notification",
        "Example School sent the term calendar as a PDF attachment. You can "
        "read it in your browser without enabling content.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "Example Outfitters handed parcel EX-OUT-882 to the carrier. Track "
        "progress from the order history in your account.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "The Example Kitchen package is scheduled for delivery on Thursday. "
        "No extra fee is due to the carrier.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "Your Example Library book shipment left the depot this morning. "
        "Delivery updates will appear in the library app.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "shipping_confirmation",
        "Example Parts confirms dispatch of order EX-PART-719. Contact the "
        "store through the phone number on your receipt if needed.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "account_verification",
        "Example Benefits verified your new mailing address. This routine "
        "profile check is complete; you do not need to send documents by email.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "account_verification",
        "To finish creating your Example Garden account, enter the confirmation "
        "code on the registration page that you opened.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "account_verification",
        "Example Makerspace confirmed your membership details. Review them in "
        "the member portal whenever convenient.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "account_verification",
        "Your Example Clinic account verification is complete. This note "
        "contains no request for a password, payment, or reply.",
    ),
    IndependentChallengeEmail(
        LEGITIMATE,
        "account_verification",
        "Example Records needs you to confirm your preferred notification "
        "language in the app. Your sign-in credentials are not requested.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "credential_theft",
        "To keep your Example Desk profile active, send your login password "
        "and username in a reply to this message.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "credential_theft",
        "The Example Storage team cannot restore your files until you provide "
        "your current account credentials through the form at "
        "https://restore.example.test/form.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "credential_theft",
        "A shared Example document is waiting. Authenticate at "
        "https://share-check.example.test/open and type your workplace password.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "credential_theft",
        "We are migrating Example accounts tonight. Email your password to "
        "the migration desk so it can be copied to the new system.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "credential_theft",
        "Your Example profile needs revalidation. Enter the password you use "
        "for the account at https://profile-confirm.example.test/continue.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "mfa_code_theft",
        "The Example login attempt is waiting. Read your authenticator code "
        "to us on this call so we can cancel the session.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "mfa_code_theft",
        "For identity review, reply with the six-digit MFA passcode currently "
        "on your device.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "mfa_code_theft",
        "Your recovery is almost complete. Send the one-time authentication "
        "token and backup code to the service agent.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "mfa_code_theft",
        "We cannot finish the Example security check until you disclose the "
        "verification code from your phone.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "qr_payment_scam",
        "A QR code is attached to your Example parking notice. Scan it to pay "
        "the overdue charge at https://pay-now.example.test/citation.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "qr_payment_scam",
        "Use the QR image in this message to sign in and release your Example "
        "refund. It will ask for your account password.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "qr_payment_scam",
        "Scan the QR code printed in this email to pay the fictional event "
        "registration surcharge before entry.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "qr_payment_scam",
        "The QR on your Example parcel notice opens a page to enter card "
        "details for a small re-delivery payment.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "Open the attached EX-AGENDA.scr file and approve the warning to "
        "view the revised Example meeting schedule.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "The attached Example account report requires you to launch the "
        "included .js file to decrypt its contents.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "We attached the updated workbook. Enable active content before "
        "viewing the urgent payroll changes.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "malicious_attachment",
        "Run the attached Example Scan.vbs utility and allow the prompt to "
        "complete your document review.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "payment_redirection",
        "Our finance office uses a replacement bank account this month. "
        "Transfer the open Example invoice balance to the new details below.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "payment_redirection",
        "The vendor's banking information has been updated. Remit today's "
        "payment to the alternate account and skip the usual callback.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "payment_redirection",
        "Example Projects has a confidential request: buy four gift cards "
        "and send their codes to the coordinator.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "payment_redirection",
        "A supplier invoice is overdue. Use the new wire instructions in this "
        "message instead of the account on file.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "payment_redirection",
        "Example Payroll overpaid your allowance. Refund the amount with gift "
        "card vouchers and send the redemption numbers to payroll.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "impersonation",
        "I am Jordan Example, the temporary director. Update our vendor's bank "
        "account using the replacement details I sent and keep this from finance.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "impersonation",
        "This is the Example Archive administrator. Share your password so I "
        "can preserve your account before the records move.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "impersonation",
        "The Example courier desk needs a card payment through the QR code in "
        "this email before they can release the parcel.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "impersonation",
        "I am calling from Example IT support. Read me the sign-in code so I "
        "can verify the alert and stop the unauthorized login.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "account_threat",
        "Your Example account will be disabled in twelve minutes. Enter your "
        "password at https://reactivate.example.test/session to avoid closure.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "account_threat",
        "Final warning: incoming Example messages will be erased unless you "
        "authenticate through https://mailbox-check.example.test/restore.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "account_threat",
        "Your Example membership is suspended. Pay the reinstatement charge "
        "within one hour to prevent permanent loss of access.",
    ),
    IndependentChallengeEmail(
        PHISHING,
        "account_threat",
        "Urgent account review: provide your recovery token immediately or "
        "your profile will be permanently closed.",
    ),
)


def validate_independent_challenge_set() -> None:
    if len(INDEPENDENT_SYNTHETIC_CHALLENGE_SET) != 60:
        raise ValueError("Independent challenge must contain exactly 60 messages.")
    counts = {
        label: sum(item.label == label for item in INDEPENDENT_SYNTHETIC_CHALLENGE_SET)
        for label in (LEGITIMATE, PHISHING)
    }
    if counts != {LEGITIMATE: 30, PHISHING: 30}:
        raise ValueError("Independent challenge must contain 30 messages per class.")
    if any("@" in item.text for item in INDEPENDENT_SYNTHETIC_CHALLENGE_SET):
        raise ValueError("Challenge text must not contain email addresses.")
    for item in INDEPENDENT_SYNTHETIC_CHALLENGE_SET:
        for token in item.text.split():
            if token.startswith(("http://", "https://")):
                hostname = urlparse(token.rstrip(".,;:!?)]}")).hostname or ""
                if not hostname.endswith(".example.test"):
                    raise ValueError("Challenge URLs must use reserved example.test domains.")


validate_independent_challenge_set()
