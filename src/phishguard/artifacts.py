"""Load pinned model and evaluation artifacts from disk, Hugging Face, or HTTPS."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException


ARTIFACT_SHA256 = {
    "models/svm.joblib": "4acfba0e501bddb9fe30836a4fded7cf1a2d197340143b527018010edbad5751",
    "models/cleaned/svm.joblib": "c78cd91cd7ce4097f713d364f70571f130030711beb5f1e4062e98a7e2d5068e",
    "results/metrics.json": "f64d162af6b6090a2dc306aa673fa6a38740d13ef45c5d79f272d58689c030f4",
    "results/cleaned/metrics.json": "25e36ee3fad653ca112fbed45254082a778f5954da47fc493278354c5931cc13",
}
ARTIFACT_REMOTE_NAMES = {
    "models/svm.joblib": "svm.joblib",
    "models/cleaned/svm.joblib": "cleaned-svm.joblib",
    "results/metrics.json": "metrics.json",
    "results/cleaned/metrics.json": "cleaned-metrics.json",
}
DOWNLOAD_TIMEOUT_SECONDS = 20
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
DEFAULT_HF_REPO_ID = "EsraArq/PhishGuard-AI-Models"
DEFAULT_HF_REVISION = "main"


class ArtifactIntegrityError(ValueError):
    """Raised when a downloaded or local artifact fails its pinned digest."""


class ArtifactDownloadError(OSError):
    """Raised when a remote artifact cannot be downloaded safely."""


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, initial_host: str):
        super().__init__()
        self.initial_host = initial_host
        self.github_asset_hosts = {
            "release-assets.githubusercontent.com",
            "objects.githubusercontent.com",
        }

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        target = urlsplit(new_url)
        if (
            target.scheme != "https"
            or target.username
            or target.password
            or target.hostname
            not in {self.initial_host, *self.github_asset_hosts}
            or target.port not in (None, 443)
        ):
            raise ArtifactDownloadError("Artifact redirect target is not allowed.")
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def _validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Artifact base URL must be HTTPS and must not contain credentials or a query."
        )
    return base_url.rstrip("/") + "/"


def _verify_artifact(artifact_name: str, content: bytes) -> bytes:
    expected_digest = ARTIFACT_SHA256[artifact_name]
    actual_digest = hashlib.sha256(content).hexdigest()
    if actual_digest != expected_digest:
        raise ArtifactIntegrityError(
            f"SHA-256 verification failed for artifact {artifact_name}."
        )
    return content


def load_artifact_bytes(
    artifact_name: str,
    project_root: Path,
    artifact_base_url: str | None = None,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
    *,
    hf_token: str | None = None,
    hf_revision: str = DEFAULT_HF_REVISION,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
) -> bytes:
    """Load one known artifact and verify it against its pinned SHA-256 digest."""
    if artifact_name not in ARTIFACT_SHA256:
        raise ValueError(f"Unsupported artifact: {artifact_name}")
    if timeout <= 0:
        raise ValueError("Download timeout must be greater than zero.")

    local_path = project_root / artifact_name
    try:
        content = local_path.read_bytes()
    except FileNotFoundError:
        content = None
    except OSError as error:
        raise OSError(f"Unable to read local artifact {artifact_name}: {error}") from error

    if content is not None:
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ArtifactIntegrityError(f"Artifact exceeds size limit: {artifact_name}.")
        return _verify_artifact(artifact_name, content)

    if hf_token is not None:
        if not hf_token.strip():
            raise ValueError("HF_TOKEN must be a non-empty string.")
        if not hf_revision.strip():
            raise ValueError("Hugging Face revision must be a non-empty string.")
        try:
            downloaded_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=ARTIFACT_REMOTE_NAMES[artifact_name],
                repo_type="model",
                revision=hf_revision,
                token=hf_token,
                etag_timeout=timeout,
            )
            with Path(downloaded_path).open("rb") as artifact_file:
                content = artifact_file.read(MAX_ARTIFACT_BYTES + 1)
        except (HfHubHTTPError, RequestException, OSError, TimeoutError):
            raise ArtifactDownloadError(
                f"Unable to download artifact {artifact_name} from Hugging Face."
            ) from None
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ArtifactDownloadError(
                f"Downloaded artifact exceeds size limit: {artifact_name}."
            )
        return _verify_artifact(artifact_name, content)

    if artifact_base_url is None:
        raise FileNotFoundError(
            f"Artifact {artifact_name} is missing locally and no HF_TOKEN or artifact URL is configured."
        )

    base_url = _validate_base_url(artifact_base_url)
    parsed_base = urlsplit(base_url)
    artifact_url = urljoin(
        base_url, quote(ARTIFACT_REMOTE_NAMES[artifact_name], safe="")
    )
    request = Request(artifact_url, headers={"User-Agent": "PhishGuard-AI/0.1"})
    opener = build_opener(_SafeRedirectHandler(parsed_base.hostname))
    try:
        with opener.open(request, timeout=timeout) as response:
            final_url = urlsplit(response.geturl())
            if (
                final_url.scheme != "https"
                or final_url.hostname
                not in {
                    parsed_base.hostname,
                    "release-assets.githubusercontent.com",
                    "objects.githubusercontent.com",
                }
            ):
                raise ArtifactDownloadError(
                    f"Artifact host changed during download: {artifact_name}."
                )
            content = response.read(MAX_ARTIFACT_BYTES + 1)
    except ArtifactDownloadError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise ArtifactDownloadError(
            f"Unable to download artifact {artifact_name}: {error}"
        ) from error

    if len(content) > MAX_ARTIFACT_BYTES:
        raise ArtifactDownloadError(
            f"Downloaded artifact exceeds size limit: {artifact_name}."
        )
    return _verify_artifact(artifact_name, content)
