"""Near-duplicate-grouped evaluation for Version 2 and Version 3 SVMs."""

import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from phishguard.clean_data import prepare_clean_dataset
from phishguard.data import LABELS
from phishguard.models import build_models
from phishguard.v3_experiment import PHISHING_LABEL, build_v3_model

GROUPED_EXPERIMENT_RANDOM_STATE = 42
GROUPED_EXPERIMENT_TEST_SIZE = 0.3
SIMILARITY_THRESHOLD = 0.85
MINHASH_PERMUTATIONS = 32
MINHASH_BANDS = 8
MINHASH_ROWS_PER_BAND = MINHASH_PERMUTATIONS // MINHASH_BANDS
V3_GROUPED_MODEL_DIRECTORY = Path("models/v3/grouped-experiment")
V3_GROUPED_RESULTS_PATH = Path("results/v3/grouped-experiment/metrics.json")
PREVIOUS_EXPERIMENT_RESULTS_PATH = Path("results/v3/experiment/metrics.json")
_SHINGLE_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_UINT64_MASK = (1 << 64) - 1


def _stable_shingles(text: str) -> set[int]:
    tokens = _SHINGLE_TOKEN_PATTERN.findall(text.casefold())
    if len(tokens) >= 3:
        shingles = (
            " ".join(tokens[index : index + 3])
            for index in range(len(tokens) - 2)
        )
    else:
        shingles = iter(tokens)
    return {
        int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for shingle in shingles
    }


def _minhash_signature(shingles: set[int]) -> tuple[int, ...]:
    if not shingles:
        return tuple([_UINT64_MASK] * MINHASH_PERMUTATIONS)

    values = np.fromiter(shingles, dtype=np.uint64)
    coefficients = []
    for index in range(MINHASH_PERMUTATIONS):
        seed = index.to_bytes(4, "big")
        digest = hashlib.blake2b(seed, digest_size=16, person=b"phishguard-v3").digest()
        multiplier = int.from_bytes(digest[:8], "big") | 1
        offset = int.from_bytes(digest[8:], "big")
        coefficients.append((multiplier & _UINT64_MASK, offset))
    multipliers = np.asarray([item[0] for item in coefficients], dtype=np.uint64)
    offsets = np.asarray([item[1] for item in coefficients], dtype=np.uint64)
    permuted = values[:, None] * multipliers[None, :] + offsets[None, :]
    return tuple(int(value) for value in permuted.min(axis=0))


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def group_similar_messages(
    texts,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Group similar texts with deterministic MinHash LSH and verified Jaccard."""
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be greater than 0 and at most 1.")

    text_values = [str(text) for text in texts]
    shingles = [_stable_shingles(text) for text in text_values]
    signatures = [_minhash_signature(features) for features in shingles]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for row_index, signature in enumerate(signatures):
        for band_index in range(MINHASH_BANDS):
            start = band_index * MINHASH_ROWS_PER_BAND
            band = signature[start : start + MINHASH_ROWS_PER_BAND]
            buckets[(band_index, band)].append(row_index)

    candidate_pairs = {
        (left, right)
        for bucket_members in buckets.values()
        if len(bucket_members) > 1
        for left, right in combinations(bucket_members, 2)
    }
    disjoint_set = _DisjointSet(len(text_values))
    verified_similar_pairs = 0
    for left, right in sorted(candidate_pairs):
        left_features = shingles[left]
        right_features = shingles[right]
        union_size = len(left_features | right_features)
        similarity = (
            len(left_features & right_features) / union_size if union_size else 1.0
        )
        if similarity >= similarity_threshold:
            disjoint_set.union(left, right)
            verified_similar_pairs += 1

    roots = [disjoint_set.find(index) for index in range(len(text_values))]
    root_to_group: dict[int, int] = {}
    group_ids = np.fromiter(
        (
            root_to_group.setdefault(root, len(root_to_group))
            for root in roots
        ),
        dtype=np.int64,
        count=len(roots),
    )
    group_sizes = np.bincount(group_ids) if len(group_ids) else np.array([], dtype=int)
    statistics: dict[str, int | float] = {
        "similarity_threshold": similarity_threshold,
        "minhash_permutations": MINHASH_PERMUTATIONS,
        "minhash_bands": MINHASH_BANDS,
        "rows_per_band": MINHASH_ROWS_PER_BAND,
        "candidate_pairs_verified": len(candidate_pairs),
        "verified_similar_pairs": verified_similar_pairs,
        "groups": len(group_sizes),
        "rows_in_multi_message_groups": int(group_sizes[group_sizes > 1].sum()),
        "multi_message_groups": int((group_sizes > 1).sum()),
        "largest_group_size": int(group_sizes.max()) if len(group_sizes) else 0,
    }
    return group_ids, statistics


def split_by_similarity_groups(
    labels,
    group_ids,
    test_size: float = GROUPED_EXPERIMENT_TEST_SIZE,
    random_state: int = GROUPED_EXPERIMENT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Choose a deterministic group holdout closest to requested class ratios."""
    label_values = np.asarray(labels)
    groups = np.asarray(group_ids)
    if len(label_values) != len(groups):
        raise ValueError("labels and group_ids must have equal lengths.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    overall_counts = {
        label: int(np.count_nonzero(label_values == label)) for label in LABELS
    }
    best_split = None
    best_score = None
    splitter = GroupShuffleSplit(
        n_splits=256,
        test_size=test_size,
        random_state=random_state,
    )
    dummy_features = np.zeros((len(label_values), 1), dtype=np.uint8)
    for train_indices, test_indices in splitter.split(
        dummy_features, label_values, groups
    ):
        test_counts = {
            label: int(np.count_nonzero(label_values[test_indices] == label))
            for label in LABELS
        }
        size_error = abs(len(test_indices) - test_size * len(label_values)) / len(
            label_values
        )
        class_error = sum(
            abs(test_counts[label] - test_size * overall_counts[label])
            / max(overall_counts[label], 1)
            for label in LABELS
        )
        score = size_error + class_error
        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_indices, test_indices, test_counts)

    if best_split is None:
        raise RuntimeError("Unable to produce a group-preserving split.")
    train_indices, test_indices, test_counts = best_split
    train_groups = set(groups[train_indices])
    test_groups = set(groups[test_indices])
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError("A similarity group appears in both train and test.")

    split_statistics = {
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_fraction": len(train_indices) / len(label_values),
        "test_fraction": len(test_indices) / len(label_values),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),
        "cross_split_group_overlap": len(overlap),
        "overall_class_counts": overall_counts,
        "test_class_counts": test_counts,
        "test_class_fraction_by_label": {
            label: test_counts[label] / max(overall_counts[label], 1)
            for label in LABELS
        },
    }
    return train_indices, test_indices, split_statistics


def _metrics(model, texts, labels) -> dict:
    predictions = model.predict(texts)
    classifier = model.named_steps["classifier"]
    decision_scores = model.decision_function(texts)
    if classifier.classes_[-1] != PHISHING_LABEL:
        decision_scores = -decision_scores
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "phishing_precision": float(
            precision_score(labels, predictions, pos_label=PHISHING_LABEL, zero_division=0)
        ),
        "phishing_recall": float(
            recall_score(labels, predictions, pos_label=PHISHING_LABEL, zero_division=0)
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


def run_grouped_v3_experiment(
    source_data,
    model_directory: Path = V3_GROUPED_MODEL_DIRECTORY,
    results_path: Path = V3_GROUPED_RESULTS_PATH,
    previous_results_path: Path = PREVIOUS_EXPERIMENT_RESULTS_PATH,
) -> dict:
    """Train and compare fresh V2/V3 models on a near-duplicate-grouped split."""
    cleaned_data, cleaning_counts = prepare_clean_dataset(source_data)
    group_ids, grouping_statistics = group_similar_messages(
        cleaned_data["Email Text"]
    )
    train_indices, test_indices, split_statistics = split_by_similarity_groups(
        cleaned_data["Email Type"],
        group_ids,
    )
    texts = cleaned_data["Email Text"].reset_index(drop=True)
    labels = cleaned_data["Email Type"].reset_index(drop=True)

    v2_model = build_models()["svm"]
    v3_model = build_v3_model()
    v2_model.fit(texts.iloc[train_indices], labels.iloc[train_indices])
    v3_model.fit(texts.iloc[train_indices], labels.iloc[train_indices])
    metrics = {
        "v2_baseline": _metrics(
            v2_model, texts.iloc[test_indices], labels.iloc[test_indices]
        ),
        "v3_masked": _metrics(
            v3_model, texts.iloc[test_indices], labels.iloc[test_indices]
        ),
    }

    model_directory.mkdir(parents=True, exist_ok=True)
    v2_path = model_directory / "v2_baseline_grouped.joblib"
    v3_path = model_directory / "v3_masked_grouped.joblib"
    joblib.dump(v2_model, v2_path)
    joblib.dump(v3_model, v3_path)
    reloaded_v3 = joblib.load(v3_path)
    reload_verified = reloaded_v3.predict(texts.iloc[test_indices]).tolist() == (
        v3_model.predict(texts.iloc[test_indices]).tolist()
    )
    if not reload_verified:
        raise RuntimeError("Reloaded grouped Version 3 pipeline changed predictions.")

    previous_report = json.loads(previous_results_path.read_text(encoding="utf-8"))
    result = {
        "experiment": "near-duplicate-grouped fresh V2 versus V3 SVM comparison",
        "grouping": grouping_statistics,
        "split": {
            "method": (
                "seeded 256-candidate GroupShuffleSplit selection minimizing "
                "test-size and per-class fraction deviation"
            ),
            "random_state": GROUPED_EXPERIMENT_RANDOM_STATE,
            "requested_test_fraction": GROUPED_EXPERIMENT_TEST_SIZE,
            **split_statistics,
            "cleaning_counts": cleaning_counts,
        },
        "svm_parameters": {
            key: value
            for key, value in v2_model.named_steps["classifier"]
            .get_params(deep=False)
            .items()
            if key in {"C", "gamma", "kernel", "class_weight"}
        },
        "metrics": metrics,
        "previous_random_split_metrics": previous_report["metrics"],
        "artifacts": {
            "v2_grouped": str(v2_path),
            "v3_grouped_with_preprocessing": str(v3_path),
            "v3_save_reload_prediction_match": reload_verified,
        },
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
