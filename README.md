# FraudGuard

FraudGuard is a Python-first fraud-risk portfolio project that ingests the Kaggle credit-card fraud dataset, trains and compares classifiers, then replays historical records as a simulated transaction stream. It is not live banking infrastructure or a production fraud-detection claim.

## Problem and approach

Fraud events are scarce, so accuracy alone is misleading. FraudGuard uses chronological evaluation where the `Time` column exists, class-weighted candidate models, PR-AUC champion selection, and a measured F1 operating threshold. The source is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud); the “real-time” element is explicitly a historical replay simulation.

## Architecture

```text
creditcard.csv → chunked validation → SQLite transaction_ledger
                                      ↓
chronological training → joblib model → replay predictions → SQLite registries
                                      ↓                       ↓
                               Streamlit Risk Intelligence Console ← quality / PSI drift
```

## Stack

Python, Pandas, NumPy, SQLite, scikit-learn, joblib, Plotly, Streamlit, and pytest.

## Setup and run

Place `creditcard.csv` in `data/raw/creditcard.csv` (the CSV is intentionally ignored by Git), then:

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
python scripts/initialize_database.py
python scripts/run_pipeline.py
python scripts/train_model.py
python scripts/simulate_stream.py --batch-size 25 --delay 0.2 --transactions 500
streamlit run app/streamlit_app.py
```

Set `FRAUDGUARD_DATASET_PATH` if the CSV lives elsewhere; see `.env.example`.

## Data engineering and monitoring

Ingestion is chunked, validated, transactional, and restartable through idempotent SQLite inserts. Quality score is `100 × valid rows / received rows`; invalid rows are missing required fields, duplicates, negative amounts, or invalid labels. SQLite stores source transactions, predictions, model metadata, quality snapshots, drift results, and pipeline runs. Drift uses actual Population Stability Index comparisons over adjacent historical windows; `STABLE`, `WATCH`, and `DRIFT` thresholds are centralized in configuration.

## Verified local run

On the supplied dataset, the validation run accepted 283,726 records and rejected 1,081 duplicate source records. The chronological holdout selected Random Forest by PR-AUC: precision **0.9815**, recall **0.7162**, F1 **0.8281**, ROC-AUC **0.9762**, and PR-AUC **0.8115** at a measured 0.50 threshold. These are local-run results, not a production performance guarantee.

## Dashboard

The Streamlit console provides a live replay feed, model-metric registry, data-quality history, and drift status. All displayed data comes from the local database and saved training artifact.

The **Live Fraud Monitor** is interactive:

- **Replay next batch** advances the historical simulation and saves new predictions to SQLite.
- **Current Replay Batch** isolates transactions processed by the latest click.
- **Recent Simulated Events** retains the latest 500 predictions and marks latest-batch rows as `NEW`.
- **Transaction Risk Lab** accepts source-compatible manual inputs and returns a what-if fraud probability, risk band, and review decision at the selected threshold.
- Risk Lab scenarios update an in-session gauge and comparison chart. They are intentionally kept separate from the labeled historical ledger because manual scenarios have no verified ground truth.

## Project structure

```text
src/fraudguard/   pipeline, ingestion, validation, database, modeling, monitoring
scripts/          initialize, ingest, train, simulate
app/              Streamlit console and four pages
tests/            ingestion, validation, features, database, model, monitoring
```

## What we built and where it helps

FraudGuard demonstrates the lifecycle around a fraud model rather than treating machine learning as a one-time notebook exercise.

| Capability | What FraudGuard does | Why it is useful |
|---|---|---|
| Data pipeline | Reads large CSV files in chunks, validates each batch, and persists clean records in SQLite. | Demonstrates repeatable, auditable model inputs outside a notebook. |
| Fraud modelling | Compares class-weighted Logistic Regression and Random Forest models using chronological evaluation and PR-AUC selection. | Demonstrates suitable thinking for rare-event classification, where accuracy is not enough. |
| Simulated operations | Replays historical transactions in controlled batches and stores every prediction. | Demonstrates streaming-style processing without falsely claiming live banking data. |
| Analyst interaction | Provides threshold controls and a what-if Risk Lab for manual transaction evaluation. | Shows how an analyst can explore model decisions and threshold trade-offs. |
| Monitoring | Tracks data quality, prediction volume, risk distribution, observed historical performance, and PSI drift. | Shows that a model requires ongoing evidence and health checks. |

This project is useful as a portfolio demonstration for data engineering, machine learning engineering, and analytics engineering roles. Its patterns can also be adapted to other rare-event decision systems, including anti-money-laundering triage, account-abuse detection, claim-risk screening, and anomaly monitoring—provided those systems use appropriate domain data, governance, and human review.


