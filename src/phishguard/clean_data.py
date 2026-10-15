import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from phishguard.data import LABELS

PLACEHOLDER_TEXTS = {"", "empty", "nan", "none", "null", "n/a"}


def normalize_email_text(text: str) -> str:
    """Canonicalize Unicode, case, and whitespace without stripping content."""
    unicode_normalized = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", unicode_normalized).strip().casefold()


def prepare_clean_dataset(
    data: pd.DataFrame,
    text_column: str = "Email Text",
    label_column: str = "Email Type",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove invalid/conflicting records and deduplicate normalized email text."""
    missing_columns = {text_column, label_column}.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    raw_records = len(data)
    valid = data.dropna(subset=[text_column, label_column]).copy()
    missing_rows = raw_records - len(valid)
    valid[text_column] = valid[text_column].astype(str)
    valid[label_column] = valid[label_column].astype(str)
    valid["_normalized_text"] = valid[text_column].map(normalize_email_text)

    prefilter_label_counts = valid.groupby("_normalized_text", sort=False)[
        label_column
    ].nunique()
    prefilter_conflicting_keys = set(
        prefilter_label_counts[prefilter_label_counts > 1].index
    )
    prefilter_conflicting_rows = int(
        valid["_normalized_text"].isin(prefilter_conflicting_keys).sum()
    )
    placeholder_mask = valid["_normalized_text"].isin(PLACEHOLDER_TEXTS)
    placeholder_rows = int(placeholder_mask.sum())
    conflicting_rows_removed_as_placeholders = int(
        (
            placeholder_mask
            & valid["_normalized_text"].isin(prefilter_conflicting_keys)
        ).sum()
    )
    valid = valid.loc[~placeholder_mask].copy()

    unexpected_labels = set(valid[label_column].unique()).difference(LABELS)
    if unexpected_labels:
        unexpected = ", ".join(sorted(map(str, unexpected_labels)))
        raise ValueError(f"CSV contains unsupported email label(s): {unexpected}")

    exact_group_sizes = valid.groupby(text_column, sort=False).size()
    normalized_group_sizes = valid.groupby("_normalized_text", sort=False).size()
    normalized_label_counts = valid.groupby("_normalized_text", sort=False)[
        label_column
    ].nunique()
    conflicting_keys = set(normalized_label_counts[normalized_label_counts > 1].index)
    conflicting_rows = int(valid["_normalized_text"].isin(conflicting_keys).sum())

    conflict_free = valid.loc[~valid["_normalized_text"].isin(conflicting_keys)]
    deduplicated = conflict_free.drop_duplicates("_normalized_text", keep="first")
    deduplicated = deduplicated.reset_index(drop=True)

    present_labels = set(deduplicated[label_column].unique())
    missing_labels = set(LABELS).difference(present_labels)
    if missing_labels:
        missing = ", ".join(sorted(missing_labels))
        raise ValueError(f"Cleaned dataset must contain both classes; missing: {missing}")

    counts = {
        "raw_records": raw_records,
        "missing_text_or_label_rows_removed": missing_rows,
        "placeholder_rows_removed": placeholder_rows,
        "conflicting_label_groups_excluded": len(prefilter_conflicting_keys),
        "conflicting_label_rows_excluded": prefilter_conflicting_rows,
        "conflicting_rows_removed_as_placeholders": (
            conflicting_rows_removed_as_placeholders
        ),
        "valid_rows_before_deduplication": len(valid),
        "exact_duplicate_groups": int((exact_group_sizes > 1).sum()),
        "normalized_duplicate_groups": int((normalized_group_sizes > 1).sum()),
        "conflicting_label_groups_remaining_after_placeholder_filter": len(
            conflicting_keys
        ),
        "conflicting_label_rows_removed_after_placeholder_filter": conflicting_rows,
        "duplicate_rows_removed": len(conflict_free) - len(deduplicated),
        "clean_samples": len(deduplicated),
    }
    return deduplicated, counts


def split_clean_dataset(
    data: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    text_column: str = "Email Text",
    label_column: str = "Email Type",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create a stratified split while keeping each normalized email in one set."""
    x_train, x_test, y_train, y_test, normalized_train, normalized_test = train_test_split(
        data[text_column],
        data[label_column],
        data["_normalized_text"],
        test_size=test_size,
        random_state=random_state,
        stratify=data[label_column],
    )
    if not set(y_train.unique()).issubset(LABELS) or not set(y_test.unique()).issubset(
        LABELS
    ):
        raise ValueError("Split contains an unsupported email label.")
    if not set(x_train.index).isdisjoint(x_test.index):
        raise RuntimeError("Training and test rows overlap.")
    if not set(normalized_train).isdisjoint(set(normalized_test)):
        raise RuntimeError("Normalized email text overlaps between train and test.")
    return x_train, x_test, y_train, y_test
