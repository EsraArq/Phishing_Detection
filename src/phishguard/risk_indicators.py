import re

MACRO_ACTION_PATTERN = re.compile(
    r"\b(?:enable|turn\s+on|allow|activate)\s+"
    r"(?:(?:all|the|microsoft|office|excel)\s+){0,3}"
    r"(?:macros?|content|active\s+content)\b"
    r"|\bmacros?\b[^.!?\n]{0,100}"
    r"\b(?:enable|turn\s+on|allow|activate)\s+content\b",
    re.IGNORECASE,
)
PASSWORD_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:send|share|provide|reply\s+(?:(?:immediately|promptly|directly|now)\s+)?with|"
    r"email|tell|disclose|give|submit)\s+"
    r"(?:(?:me|us)\s+)?(?:(?:your|the)\s+)?"
    r"(?:(?:current|student|account|login|email|one[- ]time|mfa|"
    r"recovery|backup|verification|authentication)\s+){0,2}"
    r"(?:usernames?(?:\s*,\s*(?:and\s+)?passwords?)?|passwords?|"
    r"passcodes?|login\s+credentials|credentials|"
    r"(?:(?:mfa|multi[- ]factor|authentication|verification|one[- ]time|"
    r"account\s+recovery|recovery|backup)\s+)"
    r"(?:codes?|tokens?|passcodes?))\b",
    re.IGNORECASE,
)
LINK_CREDENTIAL_PATTERN = re.compile(
    r"(?:\b(?:click|open|follow|visit|use)\s+(?:on\s+)?"
    r"(?:the\s+)?(?:link|url|button|here)\b|https?://\S+)"
    r"[^.!?\n]{0,180}\b(?:enter(?:ing)?|provide|submit|type|sign\s+in|log\s+in)\s+"
    r"(?:your\s+)?(?:(?:current|organization|workplace|account|student|email)\s+)?"
    r"(?:login\s+)?(?:password|credentials|username\s+and\s+password)\b"
    r"|\b(?:enter(?:ing)?|provide|submit|type)\s+(?:your\s+)?"
    r"(?:(?:current|organization|workplace|account|student|email)\s+)?"
    r"(?:login\s+)?(?:password|credentials|username\s+and\s+password)\b"
    r"[^.!?\n]{0,120}(?:\b(?:at|through|using)\s+(?:the\s+)?"
    r"(?:link|url|website)\b|https?://\S+)",
    re.IGNORECASE,
)
NEGATED_INSTRUCTION_PATTERN = re.compile(
    r"(?:\bdo\s+not\b|\bdon['’]?t\b|\bnever\b|\bavoid\b)"
    r"(?:\s+\w+){0,6}\s*$",
    re.IGNORECASE,
)
NEGATED_REQUEST_PATTERN = re.compile(
    r"\b(?:we|our\s+team|support|staff)\b[^.!?\n]{0,45}"
    r"\b(?:never|will\s+not|won['’]t|do\s+not|don['’]t)\s+"
    r"(?:ask|request|require)\s+(?:you\s+)?to\s*$",
    re.IGNORECASE,
)
ATTACHMENT_CONTEXT_PATTERN = re.compile(
    r"\b(?:attached|attachment|workbook|spreadsheet|document|file)\b",
    re.IGNORECASE,
)
ACTIVE_ATTACHMENT_PATTERN = re.compile(
    r"\b(?:open|run|launch|execute|approve|enable)\b[^.!?\n]{0,140}"
    r"\.(?:hta|exe|js|vbs|scr)\b"
    r"|\.(?:hta|exe|js|vbs|scr)\b[^.!?\n]{0,140}"
    r"\b(?:open|run|launch|execute|approve|enable)\b",
    re.IGNORECASE,
)
QR_REFERENCE_PATTERN = re.compile(r"\bqr(?:\s+code)?\b", re.IGNORECASE)
QR_SUSPICIOUS_ACTION_PATTERN = re.compile(
    r"\b(?:scan|use|open|follow|visit|direct|redirect|lead|"
    r"sign\s*in|log\s*in|authenticate|enter|submit|pay|purchase|"
    r"charge|request)\b",
    re.IGNORECASE,
)
QR_SENSITIVE_TARGET_PATTERN = re.compile(
    r"\b(?:sign\s*in|log\s*in|login|authenticate|credentials?|password|"
    r"passcodes?|mfa|verification\s+codes?|recovery\s+codes?|"
    r"card\s+details|payment\s+details|bank\s+details|"
    r"pay(?:ment)?|fee|charge|purchase|bank)\b",
    re.IGNORECASE,
)
QR_NEGATION_PREFIX_PATTERN = re.compile(
    r"\b(?:not|never|no|does\s+not|doesn't|will\s+not|won't|"
    r"is\s+not|isn't|are\s+not|aren't)\b(?:\s+\w+){0,5}\s*$",
    re.IGNORECASE,
)
QR_NEGATION_SUFFIX_PATTERN = re.compile(
    r"^\s*(?:(?:is|are|was|were)\s+)?"
    r"(?:not\s+(?:required|requested|needed|asked\s+for)|"
    r"never\s+(?:required|requested|needed|asked\s+for))\b",
    re.IGNORECASE,
)
IN_PERSON_PAYMENT_CONTEXT_PATTERN = re.compile(
    r"\b(?:in\s+person|in[- ]store|at\s+(?:the\s+)?"
    r"(?:cashier|counter|register|terminal)|"
    r"(?:cashier|clerk)\s+(?:at|using)\s+(?:the\s+)?"
    r"(?:card\s+)?terminal|beside\s+the\s+card\s+terminal)\b",
    re.IGNORECASE,
)
PAYMENT_REDIRECTION_PATTERN = re.compile(
    r"(?:\b(?:changed|new|updated|replacement|alternate)\s+"
    r"(?:banks?|payment|wire|banking)\b[^?\n]{0,180}"
    r"\b(?:send|wire|transfer|pay|remit|update)\b"
    r"|\b(?:send|wire|transfer|pay|remit|update)\b[^.!?\n]{0,160}"
    r"\b(?:new|replacement|alternate|changed)\s+"
    r"(?:(?:bank|payment|wire|banking)\s+)?(?:details|information|account|"
    r"number|instructions)\b"
    r"|\b(?:buy|purchase|get|obtain|send)\b[^.!?\n]{0,80}"
    r"\bgift\s+cards?(?:\s+codes?)?\b"
    r"|\bgift\s+cards?(?:\s+codes?)?\b[^.!?\n]{0,80}"
    r"\b(?:buy|purchase|get|obtain|send)\b)",
    re.IGNORECASE,
)
GIFT_CARD_REIMBURSEMENT_PATTERN = re.compile(
    r"\b(?:return|refund|reimburse)\b[^.!?\n]{0,100}"
    r"\b(?:using|with|through|by)\b[^.!?\n]{0,60}"
    r"\bgift\s+cards?(?:\s+codes?)?\b",
    re.IGNORECASE,
)
DELIVERY_RELEASE_PAYMENT_PATTERN = re.compile(
    r"(?=.*\b(?:delivery|parcel|shipment|package|courier)\b)"
    r"(?=.*\b(?:hold|held|release|redelivery)\b)"
    r"(?=.*\b(?:pay|fee|charge)\b)"
    r"(?=.*https?://).+",
    re.IGNORECASE,
)
DELIVERY_PAYMENT_ACTION_PATTERN = re.compile(
    r"\b(?:pay|paying|submit\s+(?:the\s+)?payment|"
    r"make\s+(?:the\s+)?payment)\b",
    re.IGNORECASE,
)
ACCOUNT_THREAT_PATTERN = re.compile(
    r"(?:\b(?:account|mailbox|subscription|access|data|messages?)\b"
    r"[^.!?\n]{0,100}\b(?:lock(?:ed)?|disable[ds]?|suspend(?:ed)?|"
    r"clos(?:e|ed)|delet(?:e|ed)|terminat(?:e|ed)|lose|lost)\b"
    r"|\b(?:lock|disable|suspend|close|delete|terminate|lose)\b"
    r"[^.!?\n]{0,100}\b(?:account|mailbox|subscription|access|data|messages?)\b)",
    re.IGNORECASE,
)
REQUESTED_CREDENTIAL_OR_PAYMENT_PATTERN = re.compile(
    r"(?:\b(?:reply\s+(?:(?:immediately|promptly|directly|now)\s+)?with|"
    r"send|share|provide|enter|submit|type|verify|confirm|"
    r"sign\s*in|log\s*in|authenticate|pay|wire|transfer|purchase|buy)\b"
    r"[^.!?\n]{0,160}\b(?:passwords?|passcodes?|credentials?|usernames?|"
    r"(?:mfa|recovery|verification|one[- ]time)\s+(?:codes?|tokens?)|"
    r"bank(?:ing)?\s+(?:details|account|number)|payment|money|fees?|"
    r"balances?|invoices?|bills?|charges?|gift\s+cards?)\b"
    r"|\b(?:sign\s*in|log\s*in|authenticate)\b[^!?\n]{0,160}"
    r"(?:https?://\S+|\b(?:link|website|portal)\b))",
    re.IGNORECASE,
)
URGENT_THREAT_PATTERN = re.compile(
    r"\b(?:final\s+alert|urgent(?:ly)?|immediately|within\s+\w+\s+"
    r"(?:minutes?|hours?|days?)|in\s+\w+\s+(?:minutes?|hours?)|"
    r"before\s+midnight|or\s+you(?:r)?\s+account|"
    r"prevent\s+permanent\s+(?:data|access|account)\s+loss|"
    r"permanent\s+(?:data|access|account)\s+loss)\b",
    re.IGNORECASE,
)


def _is_negated(text: str, match_start: int) -> bool:
    context = text[max(0, match_start - 100) : match_start]
    return (
        NEGATED_INSTRUCTION_PATTERN.search(context) is not None
        or NEGATED_REQUEST_PATTERN.search(context) is not None
    )


def _append_matches(
    indicators: list[dict[str, str]],
    text: str,
    pattern: re.Pattern[str],
    category: str,
    explanation: str,
) -> None:
    for match in pattern.finditer(text):
        if not _is_negated(text, match.start()):
            indicators.append(
                {
                    "category": category,
                    "evidence": match.group(0).strip(),
                    "explanation": explanation,
                }
            )


def _append_attachment_matches(
    indicators: list[dict[str, str]],
    text: str,
) -> None:
    for pattern in (MACRO_ACTION_PATTERN, ACTIVE_ATTACHMENT_PATTERN):
        for match in pattern.finditer(text):
            context = text[max(0, match.start() - 220) : match.end() + 220]
            if (
                ATTACHMENT_CONTEXT_PATTERN.search(context)
                or ACTIVE_ATTACHMENT_PATTERN.search(context)
            ) and not _is_negated(text, match.start()):
                indicators.append(
                    {
                        "category": "Attachment active-content instruction",
                        "evidence": match.group(0).strip(),
                        "explanation": (
                            "The email pairs an attachment with instructions to "
                            "enable macros/content or open active file content."
                        ),
                    }
                )


def _sentence_spans(text: str):
    start = 0
    for boundary in re.finditer(r"[.!?](?=\s|$)", text):
        end = boundary.end()
        yield start, end
        start = end
    if start < len(text):
        yield start, len(text)


def _qr_cue_is_negated(
    sentence: str,
    cue_start: int,
    cue_end: int,
) -> bool:
    prefix = sentence[max(0, cue_start - 60) : cue_start]
    suffix = sentence[cue_end : cue_end + 60]
    return bool(
        QR_NEGATION_PREFIX_PATTERN.search(prefix)
        or QR_NEGATION_SUFFIX_PATTERN.search(suffix)
    )


def _append_qr_matches(
    indicators: list[dict[str, str]],
    text: str,
) -> None:
    sentence_spans = list(_sentence_spans(text))
    for index, (start, end) in enumerate(sentence_spans):
        sentence = text[start:end]
        qr_matches = list(QR_REFERENCE_PATTERN.finditer(sentence))
        qr_available = bool(qr_matches)
        payment_context = sentence
        if not qr_matches and index:
            previous_start, previous_end = sentence_spans[index - 1]
            previous_sentence = text[previous_start:previous_end]
            refers_to_previous_qr = re.match(
                r"\s*(?:scan|use|open|follow)\s+"
                r"(?:it|this\s+code|that\s+code)\b",
                sentence,
                re.IGNORECASE,
            )
            if (
                refers_to_previous_qr
                and QR_REFERENCE_PATTERN.search(previous_sentence)
                and start - previous_end <= 180
            ):
                qr_available = True
                payment_context = previous_sentence + sentence
        if not qr_available:
            continue
        actions = list(QR_SUSPICIOUS_ACTION_PATTERN.finditer(sentence))
        targets = list(QR_SENSITIVE_TARGET_PATTERN.finditer(sentence))
        if not actions or not targets:
            continue

        for action in actions:
            for target in targets:
                if abs(action.start() - target.start()) > 120:
                    continue
                if _qr_cue_is_negated(sentence, target.start(), target.end()):
                    continue
                if qr_matches and not any(
                    min(
                        abs(qr.start() - action.start()),
                        abs(qr.start() - target.start()),
                    )
                    <= 180
                    for qr in qr_matches
                ):
                    continue

                is_payment = bool(
                    re.search(
                        r"\b(?:pay(?:ment)?|fee|charge|purchase|bank|"
                        r"card\s+details|payment\s+details)\b",
                        target.group(0),
                        re.IGNORECASE,
                    )
                )
                if is_payment and IN_PERSON_PAYMENT_CONTEXT_PATTERN.search(
                    payment_context
                ):
                    continue

                indicators.append(
                    {
                        "category": "QR code requests sign-in or payment",
                        "evidence": sentence.strip(),
                        "explanation": (
                            "The QR code is linked to an account authentication "
                            "or payment action."
                        ),
                    }
                )
                break
            else:
                continue
            break


def _append_urgent_threat_matches(
    indicators: list[dict[str, str]],
    text: str,
) -> None:
    threats = list(ACCOUNT_THREAT_PATTERN.finditer(text))
    actions = list(REQUESTED_CREDENTIAL_OR_PAYMENT_PATTERN.finditer(text))
    urgency_matches = list(URGENT_THREAT_PATTERN.finditer(text))
    for threat in threats:
        for action in actions:
            if abs(threat.start() - action.start()) > 280:
                continue
            if _is_negated(text, action.start()):
                continue
            if not any(
                min(
                    abs(urgency.start() - threat.end()),
                    abs(urgency.start() - action.start()),
                )
                <= 180
                for urgency in urgency_matches
            ):
                continue
            start = min(threat.start(), action.start())
            end = max(threat.end(), action.end())
            indicators.append(
                {
                    "category": "Urgent account threat with credential or payment request",
                    "evidence": text[start:end].strip(),
                    "explanation": (
                        "The email combines an urgent account/access threat with "
                        "a request for credentials or payment."
                    ),
                }
            )
            break
        else:
            continue
        break


def _append_delivery_release_matches(
    indicators: list[dict[str, str]],
    text: str,
) -> None:
    for action in DELIVERY_PAYMENT_ACTION_PATTERN.finditer(text):
        context = text[max(0, action.start() - 220) : action.end() + 220]
        if (
            DELIVERY_RELEASE_PAYMENT_PATTERN.search(context)
            and not _is_negated(text, action.start())
        ):
            indicators.append(
                {
                    "category": "Delivery release-fee payment request",
                    "evidence": "delivery hold/release plus fee and payment link",
                    "explanation": (
                        "The email combines a delivery hold or release claim with "
                        "a fee request and payment link."
                    ),
                }
            )
            return


def detect_security_risk_indicators(text: str) -> list[dict[str, str]]:
    """Find a limited set of explicit risky email instructions and their evidence."""
    indicators: list[dict[str, str]] = []

    _append_attachment_matches(indicators, text)

    _append_matches(
        indicators,
        text,
        PASSWORD_DISCLOSURE_PATTERN,
        "Password or credential disclosure",
        "The email asks the recipient to disclose a password, passcode, or login "
        "credentials in a reply or message.",
    )
    _append_matches(
        indicators,
        text,
        LINK_CREDENTIAL_PATTERN,
        "Credentials requested through a link",
        "The email directs the recipient to enter a password or login credentials "
        "through a link or website.",
    )
    _append_qr_matches(indicators, text)
    _append_matches(
        indicators,
        text,
        PAYMENT_REDIRECTION_PATTERN,
        "Payment redirection or gift-card request",
        "The email requests payment using changed or replacement payment "
        "details, or asks the recipient to obtain gift cards.",
    )
    _append_matches(
        indicators,
        text,
        GIFT_CARD_REIMBURSEMENT_PATTERN,
        "Payment redirection or gift-card request",
        "The email requests a refund or reimbursement using gift-card codes.",
    )
    _append_delivery_release_matches(indicators, text)
    _append_urgent_threat_matches(indicators, text)

    unique_indicators: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for indicator in indicators:
        key = (indicator["category"], indicator["evidence"].casefold())
        if key not in seen:
            seen.add(key)
            unique_indicators.append(indicator)
    return unique_indicators
