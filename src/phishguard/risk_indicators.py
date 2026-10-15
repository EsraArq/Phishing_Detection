import re

MACRO_INSTRUCTION_PATTERNS = (
    re.compile(
        r"\b(?:enable|turn\s+on|allow|activate)\s+"
        r"(?:all\s+)?(?:the\s+)?(?:microsoft\s+)?(?:office\s+)?"
        r"(?:excel\s+)?macros?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:enable|turn\s+on|allow|activate)\s+content\b"
        r"[^.!?\n]{0,100}\bmacros?\b"
        r"|\bmacros?\b[^.!?\n]{0,100}"
        r"\b(?:enable|turn\s+on|allow|activate)\s+content\b",
        re.IGNORECASE,
    ),
)
PASSWORD_DISCLOSURE_PATTERN = re.compile(
    r"\b(?:send|share|provide|reply\s+with|email|tell|disclose)\s+"
    r"(?:me\s+)?(?:your\s+)?"
    r"(?:passwords?|passcodes?|login\s+credentials|credentials)\b",
    re.IGNORECASE,
)
LINK_CREDENTIAL_PATTERN = re.compile(
    r"(?:\b(?:click|open|follow|visit|use)\s+(?:on\s+)?"
    r"(?:the\s+)?(?:link|url|button|here)\b|https?://\S+)"
    r"[^.!?\n]{0,180}\b(?:enter|provide|submit|type|sign\s+in|log\s+in)\s+"
    r"(?:your\s+)?(?:login\s+)?"
    r"(?:password|credentials|username\s+and\s+password)\b"
    r"|\b(?:enter|provide|submit|type)\s+(?:your\s+)?"
    r"(?:login\s+)?(?:password|credentials|username\s+and\s+password)\b"
    r"[^.!?\n]{0,120}(?:\b(?:at|through|using)\s+(?:the\s+)?"
    r"(?:link|url|website)\b|https?://\S+)",
    re.IGNORECASE,
)
NEGATED_INSTRUCTION_PATTERN = re.compile(
    r"(?:\bdo\s+not\b|\bdon['’]?t\b|\bnever\b|\bavoid\b)"
    r"(?:\s+\w+){0,3}\s*$",
    re.IGNORECASE,
)


def _is_negated(text: str, match_start: int) -> bool:
    context = text[max(0, match_start - 48) : match_start]
    return NEGATED_INSTRUCTION_PATTERN.search(context) is not None


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


def detect_security_risk_indicators(text: str) -> list[dict[str, str]]:
    """Find a limited set of explicit risky email instructions and their evidence."""
    indicators: list[dict[str, str]] = []

    for pattern in MACRO_INSTRUCTION_PATTERNS:
        _append_matches(
            indicators,
            text,
            pattern,
            "Office macro instruction",
            "The email asks the reader to enable Office or Excel macros/content, "
            "which can allow active content to run.",
        )

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

    unique_indicators: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for indicator in indicators:
        key = (indicator["category"], indicator["evidence"].casefold())
        if key not in seen:
            seen.add(key)
            unique_indicators.append(indicator)
    return unique_indicators
