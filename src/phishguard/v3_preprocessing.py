"""Deterministic privacy-focused text preprocessing for a future Version 3 model."""

import ipaddress
import re
from urllib.parse import urlsplit

from sklearn.feature_extraction.text import TfidfVectorizer

V3_TFIDF_TOKEN_PATTERN = r"(?u)\b(?:pii_[a-z_]+|[^\W_][\w']{1,})\b"

_EMAIL_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![\w.+-])"
    r"[\w.!#$%&'*+/=?^`{|}~-]+"
    r"\s*(?:@|%40|\[at\]|\(at\)|\{at\}|\s+at\s+)\s*"
    r"[a-z0-9-]+"
    r"(?:(?:\s*(?:%2e|\[dot\]|\(dot\)|\{dot\})\s*|\.)[a-z0-9-]+)+"
    r"(?![\w-])"
)
_URL_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![\w@.-])"
    r"(?:(?P<scheme>https?|hxxps?)://\s*)?"
    r"(?P<www>www(?:\.|\[\.\])?)?"
    r"(?P<host>"
    r"\[[0-9a-f:]+\]"
    r"|(?:\d{1,3}\.){3}\d{1,3}"
    r"|(?:[a-z0-9-]+(?:\.|\[\.\]|\(dot\)))+[a-z]{2,63}"
    r")"
    r"(?::\d{1,5})?"
    r"(?:[/][^\s<>\"'`{}\[\]]*|[?#][^\s<>\"'`{}\[\]]*)?"
)
_URL_WITH_SCHEME_PATTERN = re.compile(r"(?i)^(?:https?|hxxps?)://")
_URL_SCHEME_PATTERN = re.compile(r"(?i)^(https?|hxxps?)://")
_URL_DEFANGED_PATTERN = re.compile(r"(?i)^hxxps?://")
_URL_PUNCTUATION = ".,!?;:)]}"
_SHORTENER_DOMAINS = {
    "bit.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "t.co",
    "tinyurl.com",
}
_SSN_PATTERN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IBAN_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])[a-z]{2}\d{2}(?:[ -]?[a-z0-9]){11,30}(?![a-z0-9])"
)
_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_LABELED_ACCOUNT_PATTERN = re.compile(
    r"(?ix)"
    r"(?P<label>\b(?:account|acct|customer|member|client|user|reference|ref|"
    r"order|invoice|ticket|case|claim|policy|tracking)\b"
    r"\s*(?:(?:number|no\.?|id|identifier)\s*)?"
    r"(?::|=|\#|\bis\b)?\s*)"
    r"(?P<value>[a-z0-9][a-z0-9-]{3,})"
)
_IPV4_CANDIDATE_PATTERN = re.compile(
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])"
)
_IPV6_CANDIDATE_PATTERN = re.compile(
    r"(?i)(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])"
)
_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+\s*)?\(?\s*\d(?:[\s().-]*\d){6,14}\s*\)?"
    r"(?:\s*(?:ext\.?|extension|x)\s*\d{1,6})?(?!\d)",
    re.IGNORECASE,
)
_DATE_PATTERN = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"
)


def _mask_url(match: re.Match[str]) -> str:
    url = match.group(0)
    masked_url = url.rstrip(_URL_PUNCTUATION)
    trailing_punctuation = url[len(masked_url) :]
    normalized_url = (
        masked_url.replace("[.]", ".")
        .replace("(dot)", ".")
        .replace("[dot]", ".")
        .replace("{dot}", ".")
    )
    if not _URL_WITH_SCHEME_PATTERN.match(normalized_url):
        normalized_url = f"//{normalized_url}"

    markers = ["pii_url"]
    scheme_match = _URL_SCHEME_PATTERN.match(masked_url)
    if scheme_match and scheme_match.group(1).casefold() == "http":
        markers.append("pii_url_http")
    elif scheme_match:
        markers.append("pii_url_https")
    if _URL_DEFANGED_PATTERN.match(masked_url):
        markers.append("pii_url_defanged")

    hostname = urlsplit(normalized_url).hostname
    if hostname:
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            if hostname.casefold().startswith("xn--") or ".xn--" in hostname.casefold():
                markers.append("pii_url_punycode")
            if hostname.casefold() in _SHORTENER_DOMAINS:
                markers.append("pii_url_shortener")
        else:
            if (
                not scheme_match
                and not match.group("www")
                and normalized_url.removeprefix("//").rstrip("/")
                == hostname
            ):
                return f"pii_ip{trailing_punctuation}"
            markers.append("pii_url_ip")

    return f"{' '.join(markers)}{trailing_punctuation}"


def _mask_labeled_account(match: re.Match[str]) -> str:
    value = match.group("value")
    if not any(character.isdigit() for character in value):
        return match.group(0)
    return f"{match.group('label')}pii_account"


def _mask_ip_candidate(match: re.Match[str]) -> str:
    candidate = match.group(0)
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return candidate
    return "pii_ip"


def _mask_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = "".join(character for character in candidate if character.isdigit())
    number_digits = len(digits)
    date_candidate = candidate.strip().rstrip(".,!?;:")
    if not 7 <= number_digits <= 15 or _DATE_PATTERN.fullmatch(date_candidate):
        return candidate
    return "pii_phone"


def mask_sensitive_information(text: str) -> str:
    """Mask common direct identifiers while retaining surrounding email text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    masked = _URL_PATTERN.sub(_mask_url, text)
    masked = _EMAIL_PATTERN.sub("pii_email", masked)
    masked = _IBAN_PATTERN.sub("pii_iban", masked)
    masked = _SSN_PATTERN.sub("pii_gov_id", masked)
    masked = _CARD_PATTERN.sub("pii_payment", masked)
    masked = _LABELED_ACCOUNT_PATTERN.sub(_mask_labeled_account, masked)
    masked = _IPV4_CANDIDATE_PATTERN.sub(_mask_ip_candidate, masked)
    masked = _IPV6_CANDIDATE_PATTERN.sub(_mask_ip_candidate, masked)
    return _PHONE_PATTERN.sub(_mask_phone, masked)


def build_v3_tfidf_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer that keeps Version 3 privacy markers whole."""
    return TfidfVectorizer(
        stop_words="english",
        token_pattern=V3_TFIDF_TOKEN_PATTERN,
    )
