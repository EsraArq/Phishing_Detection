from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from phishguard.data import LABELS


def evaluate_predictions(y_true, y_pred) -> dict:
    """Return metrics with explicit, stable class ordering."""
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "labels": list(LABELS),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "phishing_class": {
            "precision": report["Phishing Email"]["precision"],
            "recall": report["Phishing Email"]["recall"],
            "f1_score": report["Phishing Email"]["f1-score"],
        },
        "classification_report": report,
    }
