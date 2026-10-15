import io
import json
import os
import pickle
from pathlib import Path

import joblib
import streamlit as st

from phishguard.artifacts import ArtifactDownloadError, load_artifact_bytes
from phishguard.risk_indicators import detect_security_risk_indicators

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSIONS = {
    "Version 2 — Cleaned SVM": {
        "model_artifact": "models/cleaned/svm.joblib",
        "metrics_artifact": "results/cleaned/metrics.json",
        "version": "Version 2 (cleaned dataset)",
    },
    "Version 1 — Original SVM": {
        "model_artifact": "models/svm.joblib",
        "metrics_artifact": "results/metrics.json",
        "version": "Version 1 (original dataset)",
    },
}

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
)


def get_artifact_base_url() -> str | None:
    """Read the optional public artifact base URL from env or Streamlit secrets."""
    base_url = os.environ.get("PHISHGUARD_ARTIFACT_BASE_URL")
    if base_url is None:
        try:
            base_url = st.secrets["PHISHGUARD_ARTIFACT_BASE_URL"]
        except (FileNotFoundError, KeyError):
            return None
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("PHISHGUARD_ARTIFACT_BASE_URL must be a non-empty URL.")
    return base_url.strip()


@st.cache_resource
def load_svm_model(artifact_name: str, artifact_base_url: str | None):
    model_bytes = load_artifact_bytes(artifact_name, PROJECT_ROOT, artifact_base_url)
    return joblib.load(io.BytesIO(model_bytes))


@st.cache_data
def load_research_metrics(
    artifact_name: str, artifact_base_url: str | None
) -> dict:
    metrics_bytes = load_artifact_bytes(artifact_name, PROJECT_ROOT, artifact_base_url)
    return json.loads(metrics_bytes)


def render_research_results() -> None:
    st.header("Research Results")
    try:
        artifact_base_url = get_artifact_base_url()
        version_reports = {}
        for choice, version in MODEL_VERSIONS.items():
            version_reports[choice] = load_research_metrics(
                version["metrics_artifact"], artifact_base_url
            )

        rows = []
        for choice, results in version_reports.items():
            model_results = results["models"]["svm"]
            report = model_results["classification_report"]
            phishing = model_results["phishing_class"]
            weighted = report["weighted avg"]
            rows.append(
                {
                    "Version": MODEL_VERSIONS[choice]["version"],
                    "Dataset samples": results.get(
                        "cleaning", {}
                    ).get("clean_samples", results.get("dataset_samples")),
                    "Accuracy": model_results["accuracy"],
                    "Weighted precision": weighted["precision"],
                    "Weighted recall": weighted["recall"],
                    "Weighted F1": weighted["f1-score"],
                    "Phishing precision": phishing["precision"],
                    "Phishing recall": phishing["recall"],
                    "False positives": model_results["confusion_matrix"][0][1],
                    "False negatives": model_results["confusion_matrix"][1][0],
                }
            )
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        st.error(f"Unable to read the saved research metrics: {error}")
        return

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            metric: st.column_config.NumberColumn(metric, format="percent")
            for metric in (
                "Accuracy",
                "Weighted precision",
                "Weighted recall",
                "Weighted F1",
                "Phishing precision",
                "Phishing recall",
            )
        },
    )

    st.markdown(
        "**Comparison note:** Version 1 and Version 2 were evaluated on different "
        "test datasets because Version 2 removes missing/placeholder records, "
        "deduplicates normalized email text, and excludes conflicting labels. "
        "Their scores are informative but not a direct like-for-like comparison."
    )

    st.subheader("Version 2 methodology")
    clean_results = version_reports["Version 2 — Cleaned SVM"]
    cleaning = clean_results["cleaning"]
    st.markdown(
        f"""
- **Raw records:** {cleaning["raw_records"]:,}; **cleaned records:** {cleaning["clean_samples"]:,}.
- **Excluded:** {cleaning["missing_text_or_label_rows_removed"]:,} rows with missing values, {cleaning["placeholder_rows_removed"]:,} blank/placeholder rows, {cleaning["conflicting_label_rows_excluded"]:,} rows in conflicting-label groups, and {cleaning["duplicate_rows_removed"]:,} duplicate rows.
- **Evaluation:** stratified {clean_results["test_size"]:.0%} holdout, with {clean_results["training_samples"]:,} training and {clean_results["test_samples"]:,} test emails; random seed `{clean_results["random_state"]}`.
- **Features:** word-level TF-IDF with English stop-word removal, fit within the training pipeline. The SVM uses an RBF kernel with `C=10`, `gamma=scale`, and balanced class weights.
"""
    )

    st.subheader("Model limitations")
    st.markdown(
        """
- These results describe one held-out split of this dataset and may not generalize to newer campaigns, other languages, or different email populations.
- TF-IDF learns word patterns from the training data; it does not inspect links, attachments, sender identity, or other email metadata.
- The interface reports the classifier's label only. It does not calculate confidence scores or explain which email features drove a prediction.
- A “Likely Legitimate Email” result is not proof that a message is safe.
"""
    )


st.title("PhishGuard AI: Machine Learning-Based Email Threat Detection")
st.write(
    "Choose either saved SVM version to analyze email text. Predictions are "
    "machine-learning classifications, not a guarantee of email safety."
)
st.caption(
    "Privacy notice: submitted email text is processed by this hosted application. "
    "Do not submit confidential or sensitive email content."
)

selected_version = st.selectbox(
    "SVM model version",
    options=list(MODEL_VERSIONS),
    index=0,
)
selected_model = MODEL_VERSIONS[selected_version]
st.caption(f"Currently selected: **{selected_model['version']}**")
try:
    artifact_base_url = get_artifact_base_url()
except ValueError as error:
    artifact_base_url = None
    st.error(f"Artifact configuration is invalid: {error}")

email_text = st.text_area(
    "Paste email content",
    height=240,
    placeholder="Paste the email text you want to analyze...",
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Enter email content before analyzing.")
    elif artifact_base_url is None and not (
        PROJECT_ROOT / selected_model["model_artifact"]
    ).is_file():
        st.error(
            f"Trained SVM model not found: {selected_model['model_artifact']}. "
            "Configure PHISHGUARD_ARTIFACT_BASE_URL to enable HTTPS downloads."
        )
    else:
        try:
            model = load_svm_model(
                selected_model["model_artifact"], artifact_base_url
            )
        except (
            OSError,
            EOFError,
            pickle.UnpicklingError,
            ImportError,
            AttributeError,
            ValueError,
        ) as error:
            st.error(f"Unable to load the saved SVM model: {error}")
        else:
            try:
                prediction = model.predict([email_text])[0]
            except (ValueError, TypeError, RuntimeError, AttributeError) as error:
                st.error(f"The model could not analyze this email: {error}")
            else:
                if prediction == "Phishing Email":
                    st.error("Potential Phishing Email")
                elif prediction == "Safe Email":
                    st.success("Likely Legitimate Email")
                else:
                    st.error(f"The model returned an unsupported label: {prediction}")
                st.info(
                    f"{selected_model['version']} provides a machine-learning "
                    "prediction, not a guarantee of email safety. Verify suspicious "
                    "messages through trusted channels."
                )
                st.subheader("Security Risk Indicators")
                indicators = detect_security_risk_indicators(email_text)
                if indicators:
                    st.warning(
                        "These rule-based warnings are separate from the model "
                        "prediction and can appear with either classification."
                    )
                    for indicator in indicators:
                        st.markdown(f"**{indicator['category']}**")
                        st.write(indicator["explanation"])
                        st.code(indicator["evidence"], language=None)
                else:
                    st.success("No configured security risk indicator was detected.")
                st.caption(
                    "Checks cover selected macro and credential-request patterns "
                    "only; absence of a warning does not establish that an email is safe."
                )

st.divider()
render_research_results()
