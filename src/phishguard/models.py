from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Create comparable TF-IDF classifier pipelines."""
    return {
        "svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                (
                    "classifier",
                    SVC(C=10, gamma="scale", kernel="rbf", class_weight="balanced"),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
