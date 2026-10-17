"""Controlled Version 2 versus Version 3 SVM training experiment."""

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from phishguard.clean_data import prepare_clean_dataset, split_clean_dataset
from phishguard.data import LABELS
from phishguard.models import build_models
from phishguard.v3_preprocessing import (
    V3_TFIDF_TOKEN_PATTERN,
    build_v3_tfidf_vectorizer,
    mask_email_batch,
)

PHISHING_LABEL = "Phishing Email"
EXPERIMENT_TEST_SIZE = 0.3
EXPERIMENT_RANDOM_STATE = 42
V3_MODEL_DIRECTORY = Path("models/v3/experiment")
V3_RESULTS_PATH = Path("results/v3/experiment/metrics.json")


def build_v3_model() -> Pipeline:
    """Build the V3 masking, TF-IDF, and SVM inference pipeline."""
    baseline_classifier = build_models()["svm"].named_steps["classifier"]
    return Pipeline(
        [
            (
                "mask",
                FunctionTransformer(mask_email_batch, validate=False),
            ),
            ("tfidf", build_v3_tfidf_vectorizer()),
            (
                "classifier",
                SVC(
                    C=baseline_classifier.C,
                    gamma=baseline_classifier.gamma,
                    kernel=baseline_classifier.kernel,
                    class_weight=baseline_classifier.class_weight,
                ),
            ),
        ]
    )


def _metrics(model: Pipeline, texts, labels) -> tuple[dict, list[str]]:
    predictions = model.predict(texts)
    classifier = model.named_steps["classifier"]
    decision_scores = model.decision_function(texts)
    if classifier.classes_[-1] != PHISHING_LABEL:
        decision_scores = -decision_scores

    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "phishing_precision": float(
            precision_score(
                labels, predictions, pos_label=PHISHING_LABEL, zero_division=0
            )
        ),
        "phishing_recall": float(
            recall_score(
                labels, predictions, pos_label=PHISHING_LABEL, zero_division=0
            )
        ),
        "phishing_f1": float(
            f1_score(labels, predictions, pos_label=PHISHING_LABEL, zero_division=0)
        ),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "pr_auc_average_precision": float(
            average_precision_score(
                [label == PHISHING_LABEL for label in labels], decision_scores
            )
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, predictions.tolist()


def _save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_v3_experiment(
    source_data,
    model_directory: Path = V3_MODEL_DIRECTORY,
    results_path: Path = V3_RESULTS_PATH,
) -> dict:
    """Train/evaluate both models on one deduplicated split; persist aggregates only."""
    cleaned_data, cleaning_counts = prepare_clean_dataset(source_data)
    x_train, x_test, y_train, y_test = split_clean_dataset(
        cleaned_data,
        test_size=EXPERIMENT_TEST_SIZE,
        random_state=EXPERIMENT_RANDOM_STATE,
    )

    v2_model = build_models()["svm"]
    v3_model = build_v3_model()
    v2_model.fit(x_train, y_train)
    v3_model.fit(x_train, y_train)

    v2_metrics, v2_predictions = _metrics(v2_model, x_test, y_test)
    v3_metrics, v3_predictions = _metrics(v3_model, x_test, y_test)

    model_directory.mkdir(parents=True, exist_ok=True)
    v2_path = model_directory / "v2_baseline.joblib"
    v3_path = model_directory / "v3_masked.joblib"
    joblib.dump(v2_model, v2_path)
    joblib.dump(v3_model, v3_path)
    reloaded_v3_model = joblib.load(v3_path)
    reloaded_v3_predictions = reloaded_v3_model.predict(x_test).tolist()
    reload_verified = reloaded_v3_predictions == v3_predictions
    if not reload_verified:
        raise RuntimeError("Reloaded Version 3 pipeline predictions do not match.")

    transition_matrix = [
        [
            sum(
                old == old_label and new == new_label
                for old, new in zip(v2_predictions, v3_predictions)
            )
            for new_label in LABELS
        ]
        for old_label in LABELS
    ]
    prediction_differences = {
        "disagreements": sum(
            old != new for old, new in zip(v2_predictions, v3_predictions)
        ),
        "agreements": sum(
            old == new for old, new in zip(v2_predictions, v3_predictions)
        ),
        "transition_matrix_labels": list(LABELS),
        "v2_to_v3_prediction_transitions": transition_matrix,
    }
    result = {
        "experiment": "fresh V2 baseline versus privacy-masked V3 SVM",
        "split": {
            "method": "stratified random split after normalized-text deduplication and conflicting-label exclusion",
            "test_size": EXPERIMENT_TEST_SIZE,
            "random_state": EXPERIMENT_RANDOM_STATE,
            "clean_samples": len(cleaned_data),
            "training_samples": len(x_train),
            "test_samples": len(x_test),
            "training_test_normalized_text_overlap": len(
                set(cleaned_data.loc[x_train.index, "_normalized_text"])
                & set(cleaned_data.loc[x_test.index, "_normalized_text"])
            ),
            "cleaning_counts": cleaning_counts,
        },
        "svm_parameters": {
            key: value
            for key, value in v2_model.named_steps["classifier"]
            .get_params(deep=False)
            .items()
            if key in {"C", "gamma", "kernel", "class_weight"}
        },
        "v3_tfidf_token_pattern": V3_TFIDF_TOKEN_PATTERN,
        "metrics": {"v2_baseline": v2_metrics, "v3_masked": v3_metrics},
        "prediction_differences": prediction_differences,
        "artifacts": {
            "v2_baseline": str(v2_path),
            "v3_masked_pipeline_with_preprocessing": str(v3_path),
            "v3_save_reload_prediction_match": reload_verified,
        },
    }
    _save_json(results_path, result)
    return result
