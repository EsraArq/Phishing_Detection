"""Privacy-conscious presentation helpers for model and rule-based results."""

from collections.abc import Mapping, Sequence

PHISHING_LABEL = "Phishing Email"
LEGITIMATE_LABEL = "Safe Email"

INDICATOR_EXPLANATIONS = {
    "Password or credential disclosure": (
        "The message asks for a password, passcode, or login credentials. "
        "Legitimate support should not ask you to send these by email."
    ),
    "Credentials requested through a link": (
        "A request to enter login details through a message-provided link may "
        "lead to credential theft."
    ),
    "QR code requests sign-in or payment": (
        "A QR code connected to account sign-in or payment may lead to an "
        "untrusted page or transaction."
    ),
    "Attachment active-content instruction": (
        "Instructions to enable macros or active content in an attachment can "
        "cause unsafe code to run."
    ),
    "Payment redirection or gift-card request": (
        "Changed payment details or gift-card requests may be attempts to "
        "divert funds."
    ),
    "Delivery release-fee payment request": (
        "A delivery hold paired with a fee and payment link may be a "
        "fraudulent payment request."
    ),
    "Urgent account threat with credential or payment request": (
        "Urgency combined with an account threat and a credential or payment "
        "request is a common pressure tactic."
    ),
}

INDICATOR_ACTIONS = {
    "Password or credential disclosure": (
        "Do not share passwords, MFA codes, or recovery codes in a message."
    ),
    "Credentials requested through a link": (
        "Do not enter credentials through a link in the message. Visit the "
        "service using a known address or app."
    ),
    "QR code requests sign-in or payment": (
        "Do not sign in or pay through the QR code until you verify the "
        "request independently."
    ),
    "Attachment active-content instruction": (
        "Do not enable macros or active content. Verify the attachment with "
        "the sender through a known channel."
    ),
    "Payment redirection or gift-card request": (
        "Verify changed payment instructions or gift-card requests using a "
        "previously known contact method."
    ),
    "Delivery release-fee payment request": (
        "Check shipment status and any fee through the carrier's official "
        "site or app."
    ),
    "Urgent account threat with credential or payment request": (
        "Pause and check account status directly through the official site "
        "or app, not through contact details in the message."
    ),
}

INDEPENDENT_VERIFICATION_ACTION = (
    "For unexpected requests, contact the organization using a trusted "
    "channel you already know."
)
GENERIC_INDICATOR_CATEGORY = "Additional security risk indicator"
GENERIC_INDICATOR_EXPLANATION = (
    "A configured rule detected a potentially concerning request. Verify it "
    "using an independent, trusted channel."
)


def build_analysis_presentation(
    prediction: str,
    indicators: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """Build display-safe result text without returning message excerpts."""
    if prediction == PHISHING_LABEL:
        model_result = "Potential Phishing Email"
        is_phishing = True
    elif prediction == LEGITIMATE_LABEL:
        model_result = "Likely Legitimate Email"
        is_phishing = False
    else:
        raise ValueError("The model returned an unsupported classification.")

    displayed_indicators = []
    for item in indicators:
        category = item.get("category")
        if isinstance(category, str) and category in INDICATOR_EXPLANATIONS:
            displayed_indicators.append(
                {
                    "category": category,
                    "explanation": INDICATOR_EXPLANATIONS[category],
                }
            )
        else:
            displayed_indicators.append(
                {
                    "category": GENERIC_INDICATOR_CATEGORY,
                    "explanation": GENERIC_INDICATOR_EXPLANATION,
                }
            )
    categories = {item["category"] for item in displayed_indicators}
    recommended_actions = [
        INDEPENDENT_VERIFICATION_ACTION,
        *(
            action
            for category, action in INDICATOR_ACTIONS.items()
            if category in categories
        ),
    ]

    return {
        "model_result": model_result,
        "is_phishing": is_phishing,
        "indicators": displayed_indicators,
        "no_indicators_message": (
            "No configured indicators were detected. This does not guarantee "
            "that the message is safe."
            if not displayed_indicators
            else None
        ),
        "caution": (
            "The model predicts likely legitimate, but the separate "
            "rule-based indicators found behavior that warrants caution. "
            "The message may still be unsafe; verify independently before "
            "acting."
            if not is_phishing and displayed_indicators
            else None
        ),
        "recommended_actions": recommended_actions,
    }
