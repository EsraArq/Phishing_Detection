import hashlib
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from phishguard.artifacts import (
    ARTIFACT_SHA256,
    DEFAULT_HF_REPO_ID,
    ArtifactDownloadError,
    ArtifactIntegrityError,
    _SafeRedirectHandler,
    load_artifact_bytes,
)


class ArtifactLoaderTests(unittest.TestCase):
    artifact_name = "models/svm.joblib"
    valid_content = b"test model artifact"

    def setUp(self):
        self.expected_digest = hashlib.sha256(self.valid_content).hexdigest()
        self.digest_patch = patch.dict(
            ARTIFACT_SHA256, {self.artifact_name: self.expected_digest}
        )
        self.digest_patch.start()
        self.addCleanup(self.digest_patch.stop)

    def test_loads_and_verifies_existing_local_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / self.artifact_name
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(self.valid_content)

            result = load_artifact_bytes(self.artifact_name, root)

        self.assertEqual(result, self.valid_content)

    def test_downloads_and_verifies_artifact_from_https(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            response = Mock()
            response.read.return_value = self.valid_content
            response.geturl.return_value = (
                "https://artifacts.example.test/release/svm.joblib"
            )
            opener = Mock()
            opener.open.return_value = nullcontext(response)

            with patch("phishguard.artifacts.build_opener", return_value=opener):
                result = load_artifact_bytes(
                    self.artifact_name,
                    Path(temporary_directory),
                    "https://artifacts.example.test/release",
                    timeout=7,
                )

        self.assertEqual(result, self.valid_content)
        request, = opener.open.call_args.args
        self.assertEqual(
            request.full_url,
            "https://artifacts.example.test/release/svm.joblib",
        )
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 7)

    def test_downloads_and_verifies_artifact_from_private_hugging_face_repo(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cached_artifact = Path(temporary_directory) / "cached-svm.joblib"
            cached_artifact.write_bytes(self.valid_content)

            with patch(
                "phishguard.artifacts.hf_hub_download",
                return_value=str(cached_artifact),
            ) as download:
                result = load_artifact_bytes(
                    self.artifact_name,
                    Path(temporary_directory) / "project",
                    hf_token="test-token-not-real",
                    hf_revision="0123456789abcdef",
                )

        self.assertEqual(result, self.valid_content)
        download.assert_called_once_with(
            repo_id=DEFAULT_HF_REPO_ID,
            filename="svm.joblib",
            repo_type="model",
            revision="0123456789abcdef",
            token="test-token-not-real",
            etag_timeout=20,
        )

    def test_rejects_hugging_face_artifact_with_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cached_artifact = Path(temporary_directory) / "cached-svm.joblib"
            cached_artifact.write_bytes(b"tampered artifact")

            with patch(
                "phishguard.artifacts.hf_hub_download",
                return_value=str(cached_artifact),
            ):
                with self.assertRaises(ArtifactIntegrityError):
                    load_artifact_bytes(
                        self.artifact_name,
                        Path(temporary_directory) / "project",
                        hf_token="test-token-not-real",
                    )

    def test_reports_missing_hugging_face_token_without_attempting_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch("phishguard.artifacts.hf_hub_download") as download:
                with self.assertRaisesRegex(FileNotFoundError, "HF_TOKEN"):
                    load_artifact_bytes(
                        self.artifact_name, Path(temporary_directory)
                    )

        download.assert_not_called()

    def test_hides_token_when_hugging_face_download_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "phishguard.artifacts.hf_hub_download",
                side_effect=OSError("failure containing private details"),
            ):
                with self.assertRaises(ArtifactDownloadError) as error:
                    load_artifact_bytes(
                        self.artifact_name,
                        Path(temporary_directory),
                        hf_token="test-token-not-real",
                    )

        self.assertNotIn("test-token-not-real", str(error.exception))

    def test_rejects_download_with_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            response = Mock()
            response.read.return_value = b"tampered artifact"
            response.geturl.return_value = (
                "https://artifacts.example.test/release/svm.joblib"
            )
            opener = Mock()
            opener.open.return_value = nullcontext(response)

            with patch("phishguard.artifacts.build_opener", return_value=opener):
                with self.assertRaises(ArtifactIntegrityError):
                    load_artifact_bytes(
                        self.artifact_name,
                        Path(temporary_directory),
                        "https://artifacts.example.test/release",
                    )

    def test_rejects_artifact_with_wrong_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifact_path = root / self.artifact_name
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_bytes(b"tampered artifact")

            with self.assertRaises(ArtifactIntegrityError):
                load_artifact_bytes(self.artifact_name, root)

    def test_reports_missing_artifact_without_download_location(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                load_artifact_bytes(
                    self.artifact_name, Path(temporary_directory)
                )

    def test_rejects_non_https_download_location(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                load_artifact_bytes(
                    self.artifact_name,
                    Path(temporary_directory),
                    "http://artifacts.example.test/release",
                )

    def test_redirects_only_to_https_configured_or_github_asset_hosts(self):
        handler = _SafeRedirectHandler("github.com")
        request = Mock()
        expected_redirect = Mock()

        with patch(
            "urllib.request.HTTPRedirectHandler.redirect_request",
            return_value=expected_redirect,
        ) as parent_redirect:
            redirected = handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://release-assets.githubusercontent.com/file",
            )

        self.assertIs(redirected, expected_redirect)
        parent_redirect.assert_called_once()
        for target_url in (
            "http://github.com/file",
            "https://malicious.example.test/file",
            "https://release-assets.githubusercontent.com:8443/file",
        ):
            with self.subTest(target_url=target_url):
                with self.assertRaises(ArtifactDownloadError):
                    handler.redirect_request(
                        request, None, 302, "Found", {}, target_url
                    )

    def test_reports_download_timeout(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            opener = Mock()
            opener.open.side_effect = TimeoutError("timed out")

            with patch("phishguard.artifacts.build_opener", return_value=opener):
                with self.assertRaises(ArtifactDownloadError):
                    load_artifact_bytes(
                        self.artifact_name,
                        Path(temporary_directory),
                        "https://artifacts.example.test/release",
                    )


if __name__ == "__main__":
    unittest.main()
