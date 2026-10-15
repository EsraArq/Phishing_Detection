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
├── scripts/
│   ├── train.py               # Train Version 1 SVM and Random Forest
│   ├── train_cleaned.py       # Clean data, train, and evaluate Version 2
│   └── evaluate.py            # Evaluate a saved model on labeled CSV data
├── src/phishguard/
│   ├── clean_data.py          # Text normalization, conflict handling, and split
│   ├── data.py                # Dataset loading and validation
│   ├── evaluation.py          # Classification metrics
│   ├── models.py              # TF-IDF SVM and Random Forest pipelines
│   └── risk_indicators.py     # Separate rule-based warning checks
├── tests/
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

Use Python 3.10 or newer. From the repository root, create and activate a
virtual environment if needed, then install dependencies and the project in
editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install makes `phishguard` importable without setting
`PYTHONPATH`.

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

Run the full unit test suite with:

```bash
python -m unittest discover -s tests -v
```

## Run the Streamlit application

With the virtual environment activated and the editable install complete:

```bash
python -m streamlit run app/streamlit_app.py
```

The interface defaults to **Version 2 — Cleaned SVM** and also supports
**Version 1 — Original SVM**. Both pipelines classify the same pasted email
content; the displayed result is a machine-learning prediction, not a
guarantee that an email is safe.

### Version 2 evaluation results

Version 2 SVM accuracy was **98.67%** on its held-out test split (5,254 emails,
from 17,513 cleaned records). This is performance on that dataset and split,
not guaranteed real-world accuracy or a promise of performance on new email.
The saved report also records weighted precision, recall, and F1 of about
98.67%, phishing-class precision of 98.76%, phishing recall of 97.65%, 24 false
positives, and 46 false negatives.

Version 1 and Version 2 results are **not directly comparable**: Version 2
removes missing, placeholder, duplicate, and conflicting-label data before
splitting, so its held-out test dataset differs from Version 1's. See the
locally generated `results/metrics.json` and `results/cleaned/metrics.json`
for the respective full reports.

## Security Risk Indicators and limitations

Security Risk Indicators are separate rule-based checks for selected requests
to enable Office/Excel macros, disclose passwords, or enter credentials through
links. When a rule matches, the interface displays the triggering text and an
explanation independently of the ML label; a warning may appear even when the
model predicts “Likely Legitimate Email.” These checks cover only configured
patterns. No warning does not establish that a message is safe, and a warning
is not itself a model prediction.

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
