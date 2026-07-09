# Customer Churn Prediction System

A production-style machine learning system that predicts customer churn for a telecom provider, built with an emphasis on clean software engineering practices, imbalance-aware evaluation, and a config-driven, dataset-agnostic pipeline.

**Live demo:** run locally with `streamlit run app.py` (see Setup below)

---

## Overview

Customer churn — when a paying customer stops using a service — is one of the most expensive problems for subscription businesses to solve reactively. This project builds an end-to-end pipeline that predicts churn risk *before* it happens, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features).

The project intentionally avoids notebook-only development. Every stage — cleaning, preprocessing, training, inference — lives in modular, reusable Python scripts, wired together by a single YAML config file.

---

## Key Findings from EDA

- **Class imbalance:** the dataset is ~73% "No Churn" / 27% "Churn." This makes accuracy a misleading metric — a model that predicts "no churn" for every customer scores ~73% accuracy while catching zero actual churners. This finding shaped every downstream modeling decision.
- **A hidden data quality bug:** `TotalCharges` was stored as a string column (not numeric) due to 11 rows containing blank values — all belonging to brand-new customers with `tenure = 0`. Coerced to numeric and imputed as `0`, since zero tenure genuinely means zero billed charges.
- **Contract type, tenure, and monthly charges** emerged as visibly strong churn drivers during exploratory visualization (see `notebooks/01_exploration.ipynb`).

---

## Architecture




customer-churn-prediction/
│
├── config/
│   └── config.yaml           # Single source of truth: column names, paths, model settings
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned train/test splits (generated, not committed)
│
├── notebooks/
│   └── 01_exploration.ipynb  # EDA, class balance, confusion matrices, ROC curves
│
├── src/
│   ├── data_pipeline.py      # Cleaning, stratified split, sklearn preprocessing Pipeline
│   ├── train.py              # Trains & compares Logistic Regression, Random Forest, XGBoost
│   └── predict.py            # Inference on new customer data
│
├── saved_models/              # Trained model + fitted preprocessor (generated, not committed)
├── app.py                     # Streamlit UI for interactive predictions
└── requirements.txt




### Design principle: config-driven, dataset-agnostic pipeline

Every script reads column names, paths, and target labels from `config/config.yaml` rather than hardcoding them. This means the same `data_pipeline.py` / `train.py` / `predict.py` code can be repointed at a structurally different dataset (e.g. an e-commerce churn dataset) by swapping in a new config file — no code changes required.

### Design principle: no test-set leakage

Data is split into train/test **before** any scaling or encoding happens. The `StandardScaler` and `OneHotEncoder` are fit only on training data, then applied to the test set — never the reverse. This is a common mistake that silently inflates evaluation metrics, and avoiding it was a deliberate architectural choice.

---

## Modeling & Evaluation

Three models were trained and compared: **Logistic Regression**, **Random Forest**, and **XGBoost** — each handling class imbalance via `class_weight='balanced'` (LR/RF) or `scale_pos_weight` (XGBoost, computed dynamically from the training data's class ratio).

### Why F1, not accuracy

Given the 73/27 class imbalance, accuracy was explicitly rejected as the selection metric. Instead, models were compared on **precision, recall, F1, ROC-AUC, and PR-AUC**, with **F1 score chosen as the primary selection criterion** — a deliberate balance between catching real churners (recall) and not overwhelming the business with false alarms (precision).

### Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.504 | **0.783** | **0.614** | **0.842** | **0.633** |
| Random Forest | **0.635** | 0.489 | 0.553 | 0.825 | 0.621 |
| XGBoost | 0.533 | 0.695 | 0.603 | 0.819 | 0.611 |

**Logistic Regression was selected** — despite being the simplest model of the three. It achieved the best recall (catching 78% of actual churners) and the best F1, ROC-AUC, and PR-AUC. This was a genuinely interesting result: with a moderately-sized, mostly categorical dataset, a well-regularized linear model held its own against tree-based ensembles — a useful reminder that model complexity isn't automatically correlated with performance.

Confusion matrices and ROC curve comparisons for all three models are available in `notebooks/01_exploration.ipynb`.

---


---

## Generalization Test: E-commerce Dataset

To validate the config-driven architecture's core claim — that the same pipeline code can handle a structurally different dataset by swapping only the config file — the pipeline was pointed at a second dataset: an [e-commerce customer churn dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) (5,630 customers, 20 features, mostly behavioral/numerical rather than Telco's mostly-categorical composition).

### Pipeline changes required

The pipeline did **not** run unmodified — and that gap is itself an informative result. Two changes were needed:

1. **Excel file support** — the e-commerce dataset ships as `.xlsx` (with a "Data Dict" sheet alongside the real "E Comm" data sheet), while Telco is `.csv`. `load_raw_data()` was updated to branch on file extension.
2. **Generalized missing-value imputation** — Telco's original cleaning logic contained a hardcoded, dataset-specific fix (`TotalCharges` blank-string coercion, imputed with `0`). The e-commerce dataset has genuine `NaN` values spread across 7 different numerical columns (`Tenure`, `WarehouseToHome`, `HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder` — each ~4.5-5.5% missing). `clean_data()` was refactored to apply **median imputation across all configured numerical columns generically**, rather than a single hardcoded column-specific fix. This still correctly handles Telco's `TotalCharges` case, though with a minor tradeoff: median imputation is less semantically precise than the original "0 tenure = $0 charges" domain-specific reasoning.
3. **CLI-configurable entrypoints** — both `data_pipeline.py` and `train.py` now accept a `--config` argument, and artifact filenames are derived from `config["dataset"]["name"]` so running the pipeline on one dataset never overwrites the other's saved models.

### Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.441 | 0.847 | 0.580 | 0.886 | 0.681 |
| Random Forest | 0.994 | 0.863 | 0.924 | 0.999 | 0.994 |
| **XGBoost** | **0.969** | **0.979** | **0.974** | 0.999 | 0.991 |

**XGBoost was selected** (F1 = 0.974) — a stark contrast to Telco, where Logistic Regression won and all models sat in the 0.55-0.61 F1 range.

### Investigating the unusually high performance

An F1 score above 0.97 is unusual enough to warrant suspicion rather than celebration — in real-world ML work, results that look "too good" are more often a sign of data leakage than genuine model quality. Before accepting this result, two checks were run:

- **Correlation analysis** across all 13 numerical features against `Churn` — no single feature showed a strong linear correlation (`Tenure`: -0.35, `Complain`: 0.25, all others below 0.16), ruling out an obvious leaking column.
- **Duplicate check** — zero duplicate rows and zero duplicate `CustomerID`s, ruling out train/test contamination via repeated records.

**Conclusion:** the elevated performance is most likely explained by the e-commerce dataset being genuinely more cleanly separable — a combination of its smaller size (5,630 vs. 7,043 rows) and a more curated feature set — rather than a pipeline defect or leakage. Tree-based models (Random Forest, XGBoost) likely captured non-linear interactions across several moderate-strength features that no single correlation coefficient would reveal.

This distinguishes two very different situations that can look identical from a metrics table alone — "the pipeline is broken" versus "the data is genuinely easier" — and reaching a defensible answer required verification, not just accepting a good-looking number.

---

## Setup

```bash
git clone https://github.com/crazyEj/Customer-Church-Prediction-System.git
cd Customer-Church-Prediction-System

py -3.12 -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

Place the [Telco Customer Churn CSV](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) at `data/raw/telco_churn.csv`.

### Run the pipeline

```bash
python src/data_pipeline.py   # clean data, split, fit preprocessor
python src/train.py           # train & compare 3 models, save best
python src/predict.py         # example inference on sample data
streamlit run app.py          # launch interactive UI
```

---

## Lessons Learned

- **Git hygiene matters from commit #1.** A virtual environment accidentally committed early in this project ballooned the repo to 340MB and got rejected by GitHub's file size limits. Resolved by rewriting git history (`git init` fresh + `.gitignore` in place before the first commit) — a good reminder to set up `.gitignore` *before* running `git add .` for the first time, not after.
- **F1 vs. Recall is a business decision, not just a technical one.** F1 was chosen here as a balanced default, but a real deployment would involve discussing with stakeholders whether missing a churner (false negative) is costlier than a false alarm (false positive) — that conversation should happen before finalizing a selection metric, not after.

---

## Future Work

- Validate pipeline generalization on a second, structurally different churn dataset (e-commerce/subscription domain) by swapping in a new config file
- Hyperparameter tuning (GridSearchCV/Optuna) on the selected model
- Model explainability (SHAP values) to surface *why* a given customer is predicted to churn, not just that they are
