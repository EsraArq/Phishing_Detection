# PhishGuard AI: Machine Learning-Based Email Threat Detection

PhishGuard AI compares TF-IDF-based Support Vector Machine (SVM) and Random
Forest classifiers for phishing email detection. The Streamlit application
offers two saved SVM versions and displays a small set of separate,
rule-based security risk indicators.

## Project structure

```text
.
├── app/
│   └── streamlit_app.py       # Streamlit interface and Version 1/2 selector
├── data/                      # Local Kaggle CSV; excluded from Git
├── models/
│   ├── svm.joblib             # Version 1 SVM; generated locally
│   ├── random_forest.joblib   # Version 1 Random Forest; generated locally
│   └── cleaned/               # Version 2 SVM and Random Forest; generated locally
├── results/
│   ├── metrics.json           # Version 1 results; generated locally
│   └── cleaned/metrics.json   # Version 2 results; generated locally
├── .gitignore
├── pyproject.toml
├── requirements.txt           # Pinned runtime dependencies and local package
├── scripts/
│   ├── train.py               # Train Version 1 SVM and Random Forest
│   ├── train_cleaned.py       # Clean data, train, and evaluate Version 2
│   └── evaluate.py            # Evaluate a saved model on labeled CSV data
├── src/phishguard/
│   ├── artifacts.py           # Local/Hub/HTTPS artifact loading with SHA-256 checks
│   ├── clean_data.py          # Text normalization, conflict handling, and split
│   ├── data.py                # Dataset loading and validation
│   ├── evaluation.py          # Classification metrics
│   ├── models.py              # TF-IDF SVM and Random Forest pipelines
│   ├── presentation.py        # Privacy-conscious prediction and warning wording
│   └── risk_indicators.py     # Separate rule-based warning checks
├── tests/
│   ├── test_artifacts.py
│   ├── test_phishguard.py
│   └── test_risk_indicators.py
├── phishing_detection.py      # Preserved original research script
└── RFC vs SVM                 # Preserved original research notes
```

Local datasets, trained pipelines, evaluation results, virtual environments,
and common private configuration/credential files are excluded by
[.gitignore](./.gitignore).

## Dataset source, license, and download

The project uses the Kaggle
[Phishing Email Detection dataset by subhajournal](https://www.kaggle.com/datasets/subhajournal/phishingemails).
Kaggle's dataset metadata currently lists **GNU Lesser General Public License
3.0 (LGPL-3.0)**. This is the license reported by Kaggle, not independent legal
verification of the dataset contents or all rights in included email text.
Review the [license terms](https://www.gnu.org/licenses/lgpl-3.0.html) and
confirm the current dataset terms and any third-party content rights before
reuse or redistribution. The dataset is not included in this repository.

To obtain the expected CSV:

1. Sign in to Kaggle, open the dataset page linked above, and use its Download
   action.
2. Extract the downloaded archive.
3. Place the file named `Phishing_Email.csv` at `data/Phishing_Email.csv`
   (create the `data` directory if needed).

The training scripts expect columns named `Email Text` and `Email Type`, with
labels `Safe Email` and `Phishing Email`. The cleaned pipeline removes rows
with missing text or labels, blank/placeholder text, conflicting-label groups,
and duplicate normalized email texts.

## Installation

Use Python 3.11 or newer. From the repository root, create and activate a
virtual environment if needed, then install dependencies and the project in
editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file pins versions tested with the saved models and installs
the local `phishguard` package in editable mode.

## Train and reproduce the models

After downloading the CSV to `data/Phishing_Email.csv`, reproduce the cleaned
Version 2 models and report with:

```bash
python scripts/train_cleaned.py
```

This uses a reproducible stratified 70/30 split with random seed 42 and fits
TF-IDF only within each model pipeline on training data. It saves models under
`models/cleaned/` and metrics to `results/cleaned/metrics.json`. The original
Version 1 training workflow is available separately:

```bash
python scripts/train.py
```

This writes to `models/` and `results/`; it does not create the cleaned
Version 2 artifacts. Both commands require the dataset described above.

## Version 3 privacy preprocessing (pre-training phase)

The isolated `phishguard.v3_preprocessing` module provides deterministic
masking for email addresses, URLs, phone numbers, IP addresses, and common
account, payment, government-ID, and IBAN values. It preserves surrounding
message text and emits category markers that the Version 3 TF-IDF vectorizer
keeps as individual features. This preprocessing is not connected to either
existing training script or model. Its rule-based detection cannot guarantee
that every identifier is removed, and masking can affect phishing-detection
performance; the separate Version 3 training and evaluation plan must measure
both before any model is produced.

The separate near-duplicate-grouped comparison uses deterministic MinHash
candidate grouping followed by exact shingle-Jaccard verification, then selects
a seeded, class-balanced group holdout. It does not fit a vectorizer to
held-out data during similarity grouping:

```bash
python -m scripts.experiment_v3_grouped
```

This writes new artifacts under Git-ignored `models/v3/grouped-experiment/`
and `results/v3/grouped-experiment/`; it preserves the earlier Version 3
experiment directory.

To run the controlled Version 2 versus Version 3 SVM experiment against the
existing dataset, use:

```bash
python -m scripts.experiment_v3
```

The experiment checks and removes conflicting-label and normalized duplicate
groups before making one stratified 70/30 split (seed 42). Both fresh SVMs use
the same training examples and parameters; Version 3 masking runs inside its
saved prediction pipeline. Aggregate metrics and both models are written only
to Git-ignored `results/v3/experiment/` and `models/v3/experiment/`
directories. The report contains no email examples or per-message
identifiers.

## Synthetic Version 2/3 challenge evaluation

Evaluate the existing grouped Version 2 and Version 3 models against 40
fictional, labeled messages without retraining:

```bash
python -m scripts.evaluate_synthetic_challenge
```

The command reconstructs the grouped training partition only to check for
exact and normalized challenge-message duplicates. It reports aggregate
overall and scenario metrics, disagreements, and duplicate counts only; it
does not include message text or alter the official experiment reports. The
challenge set is small and synthetic, so its scores are diagnostic rather than
representative estimates of production performance. Results are stored in
Git-ignored `results/v3/synthetic-challenge/`.

An independent 60-message fictional challenge, with separate rule-based
indicator and saved-model metrics, can be run without retraining:

```bash
python -m scripts.evaluate_independent_challenge
```

Its aggregate-only report is stored separately in
Git-ignored `results/v3/synthetic-challenge-independent/`. It does not change
the earlier challenge or official experiment results.

Run the full unit test suite with:

```bash
python -m unittest discover -s tests -v
```

## Run the Streamlit application

From the repository root, run the app using the project environment:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

The equivalent command after activating `.venv` is
`python -m streamlit run app/streamlit_app.py`. The application supports the
saved Version 1 original SVM and Version 2 cleaned SVM; it defaults to Version
2. Version 3 is experimental and is not integrated into the app.

### Application architecture

1. Streamlit accepts email text in memory and passes it to the selected
   existing SVM pipeline. The application does not train models at startup.
2. The artifact loader reads local model files first. If a model is absent, it
   can use the configured private Hugging Face repository or optional HTTPS
   artifact source. The bytes are checked against a pinned SHA-256 digest
   before `joblib` deserializes the model.
3. The SVM classification is displayed separately from the independent
   Security Risk Indicators. Indicator matches do not override the prediction.
   A likely legitimate classification alongside an indicator triggers a
   prominent caution. Indicator details use generic explanations, not matched
   email excerpts.
4. Recommended Actions are selected from detected indicator categories.
   Neither a likely legitimate classification nor an empty indicator list
   guarantees safety.
5. Aggregate research metrics are loaded and shown separately from an
   individual email analysis.

The app reports generic errors without echoing submitted message text. The
local privacy notice advises against submitting confidential email content.
Hosting infrastructure and provider-level telemetry are outside the app's
control.

## Model artifact hosting status

**Public model hosting and public deployment are on hold** pending verification
of dataset provenance, licensing, and redistribution rights for the source
emails and derived model artifacts. Kaggle currently reports LGPL-3.0 for the
dataset, but that metadata and the rights to all included email text have not
been independently verified. Private hosting does not resolve those questions.
Do not upload models or deploy this app publicly until the review is complete.

For local inference, put the expected model and metrics files in their normal
ignored `models/` and `results/` locations; no credentials are needed when
those files are present. If authorized private Hub access is needed, set
`HF_TOKEN` and optionally `PHISHGUARD_HF_REVISION` in the environment or
Git-ignored `.streamlit/secrets.toml`. The token is passed to the Hub client
only; do not place it in source or committed files. The revision defaults to
`main`; a full commit SHA is preferable for reproducibility. The optional
`PHISHGUARD_ARTIFACT_BASE_URL` setting requires HTTPS. Keep any artifact source
private and access-controlled until provenance and licensing review is done.

### Version 2 evaluation results

These are **original-dataset benchmark results**, not estimates of guaranteed
real-world performance. Version 1's SVM achieved **97.93% accuracy** on 5,591
held-out emails from 18,634 records (stratified 70/30 split, seed 42); it
recorded 81 false positives and 35 false negatives. Version 2's SVM achieved
**98.67% accuracy** on 5,254 held-out emails from 17,513 cleaned records
(stratified 70/30 split, seed 42); it recorded 24 false positives and 46 false
negatives, with phishing precision 98.76% and recall 97.65%.

Version 1 and Version 2 results are **not directly comparable**: Version 2
removes missing, placeholder, duplicate, and conflicting-label data before
splitting, so its held-out test dataset differs from Version 1's. See the
locally generated `results/metrics.json` and `results/cleaned/metrics.json`
for the respective full reports.

The separate near-duplicate-grouped Version 2/3 experiment used the same
group-safe training and test examples for both models, with no similarity
group crossing the split. On that held-out partition, both models achieved
98.55% accuracy. Version 2 recorded 31 false positives and 45 false negatives;
experimental Version 3 recorded 27 false positives and 49 false negatives.
This is one dataset partition and is not evidence of a general performance
improvement. Version 3 remains experimental and is not selectable in Streamlit.

### Synthetic challenge results (not research benchmarks)

Synthetic challenge figures are **not real-world detection accuracy** and are
kept separate from the original-dataset benchmark:

- On the 40-message fictional challenge, saved grouped Version 2 and Version 3
  each achieved 57.5% accuracy and 35% phishing recall, with 4 false positives
  and 13 false negatives. The hand-authored set is small and tests selected
  scenarios only.
- On the independent 60-message fictional challenge (30 phishing, 30
  legitimate), the pre-QR-refinement indicator report recorded 20 true
  positives, 2 false positives, 28 true negatives, and 10 false negatives.
  The grouped Version 2 model recorded 14 true positives, 6 false positives,
  24 true negatives, and 16 false negatives; grouped Version 3 recorded 11,
  6, 24, and 19 respectively.
- A subsequent QR-rule refinement eliminated both previously identified
  legitimate QR-payment alerts on that frozen challenge while retaining all
  20 previously detected phishing messages. The combined Version 2 OR
  indicator outcome was 28 true positives, 6 false positives, 24 true
  negatives, and 2 false negatives. These are challenge-set observations only;
  the set is synthetic, small, and hand-authored.

Do not interpret synthetic results as population-level performance. The
fictional text and reserved domains do not represent the variability of real
mail, and scenario-level counts are small. The indicator rules are also
limited to explicitly configured text patterns.

## Security Risk Indicators and limitations

Security Risk Indicators are a separate rule-based warning layer. Configured
checks look for credential/MFA/recovery-code disclosure, QR-assisted sign-in or
payment, active attachment instructions, payment redirection or gift-card
requests, delivery release-fee links, and urgent account threats paired with a
credential or payment request. Rules require combinations of context rather
than isolated words. Warnings do not override or change the SVM classification.
These checks cover only configured patterns and may miss novel wording or
produce false positives. No warning does not establish that a message is safe,
and a warning is not itself a model prediction.

The text classifiers learn patterns in a labeled email corpus. They do not
independently verify senders, inspect link destinations or attachments, or
understand organizational context. Dataset bias, duplicate examples, label
quality, changing attack patterns, and domain/language differences can affect
results. Treat both model labels and rule warnings as decision support, not as
a substitute for security review.

## Preserved original research

The original [phishing_detection.py](./phishing_detection.py) and
[RFC vs SVM](./RFC%20vs%20SVM) research artifacts are retained unchanged. The
script's dataset read uses the Windows-specific path
`C:\Users\18434\Downloads\Phishing_Email.csv\PhishingEmail.csv`, which must be
replaced or adapted to run that legacy script on another machine. Its original
results and methodology are historical research records; the current Version 1
and Version 2 reports are generated by the new pipelines.
