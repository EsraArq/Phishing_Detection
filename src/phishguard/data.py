from pathlib import Path

import pandas as pd

LABELS = ("Safe Email", "Phishing Email")


def load_dataset(
    csv_path: str | Path,
    text_column: str = "Email Text",
    label_column: str = "Email Type",
) -> tuple[pd.Series, pd.Series]:
    """Load and validate email text and labels from a CSV file."""
    data = pd.read_csv(csv_path)
    missing_columns = {text_column, label_column}.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    data = data.dropna(subset=[text_column, label_column])
    labels = set(data[label_column].unique())
    unexpected_labels = labels.difference(LABELS)
    if unexpected_labels:
        unexpected = ", ".join(sorted(map(str, unexpected_labels)))
        raise ValueError(f"CSV contains unsupported email label(s): {unexpected}")

    missing_labels = set(LABELS).difference(labels)
    if missing_labels:
        missing = ", ".join(sorted(missing_labels))
        raise ValueError(f"CSV must contain both classes; missing: {missing}")

    return data[text_column].astype(str), data[label_column].astype(str)
