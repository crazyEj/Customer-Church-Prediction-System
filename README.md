A production-style machine learning system that predicts customer churn, built with an emphasis on clean software engineering practices, imbalance-aware evaluation, model explainability, and a config-driven, dataset-agnostic pipeline. Validated end-to-end on two structurally different datasets (Telco and e-commerce) without changing a line of pipeline code.

Live demo: run locally with streamlit run app.py (see Setup)

Table of Contents


Overview
Key Findings from EDA
Architecture
Design Principles
Modeling & Evaluation
Explainability with SHAP
Generalization Test: E-commerce Dataset
Setup
Lessons Learned
Future Work


Overview

Customer churn — when a paying customer stops using a service — is one of the most expensive problems for subscription businesses to solve reactively. This project builds an end-to-end pipeline that predicts churn risk before it happens, using the Telco Customer Churn dataset (7,043 customers, 21 features) as the primary case study, and a second e-commerce dataset to stress-test generalization.

The project intentionally avoids notebook-only development. Every stage — cleaning, preprocessing, training, inference, and explanation — lives in modular, reusable Python scripts, wired together by a single YAML config file, with a Streamlit UI on top for interactive, per-customer predictions.

Key Findings from EDA


Class imbalance: the dataset is ~73% "No Churn" / 27% "Churn." This makes accuracy a misleading metric — a model that predicts "no churn" for every customer scores ~73% accuracy while catching zero actual churners. This finding shaped every downstream modeling decision.
A hidden data quality bug: TotalCharges was stored as a string column (not numeric) due to 11 rows containing blank values — all belonging to brand-new customers with tenure = 0. Coerced to numeric and imputed as 0, since zero tenure genuinely means zero billed charges.
Contract type, tenure, and monthly charges emerged as visibly strong churn drivers during exploratory visualization (see notebooks/01_exploration.ipynb).


Architecture

customer-churn-prediction/
│
├── config/
│   └── config.yaml          # Single source of truth: column names, paths, model settings
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned train/test splits (generated, not committed)
│
├── notebooks/
│   └── 01_exploration.ipynb # EDA, class balance, confusion matrices, ROC curves
│
├── src/
│   ├── data_pipeline.py     # Cleaning, stratified split, sklearn preprocessing Pipeline
│   ├── train.py             # Trains & compares Logistic Regression, Random Forest, XGBoost
│   ├── predict.py           # Inference on new customer data
│   └── explain.py           # SHAP value computation for global + per-prediction explanations
│
├── saved_models/            # Trained model, fitted preprocessor, SHAP artifacts (generated, not committed)
├── app.py                   # Streamlit UI for interactive predictions + SHAP explanations
└── requirements.txt

Design Principles

Config-driven, dataset-agnostic pipeline. Every script reads column names, paths, and target labels from config/config.yaml rather than hardcoding them. This means the same data_pipeline.py / train.py / predict.py code can be repointed at a structurally different dataset (e.g. an e-commerce churn dataset) by swapping in a new config file — no code changes required. This claim is validated directly in the generalization test below.

No test-set leakage. Data is split into train/test before any scaling or encoding happens. The StandardScaler and OneHotEncoder are fit only on training data, then applied to the test set — never the reverse. This is a common mistake that silently inflates evaluation metrics, and avoiding it was a deliberate architectural choice.

Modeling & Evaluation

Three models were trained and compared: Logistic Regression, Random Forest, and XGBoost — each handling class imbalance via class_weight='balanced' (LR/RF) or scale_pos_weight (XGBoost, computed dynamically from the training data's class ratio).

Why F1, not accuracy. Given the 73/27 class imbalance, accuracy was explicitly rejected as the selection metric. Instead, models were compared on precision, recall, F1, ROC-AUC, and PR-AUC, with F1 score chosen as the primary selection criterion — a deliberate balance between catching real churners (recall) and not overwhelming the business with false alarms (precision).

Results — Telco Dataset

ModelPrecisionRecallF1ROC-AUCPR-AUCLogistic Regression0.5040.7830.6140.8420.633Random Forest0.6350.4890.5530.8250.621XGBoost0.5330.6950.6030.8190.611

Logistic Regression was selected — despite being the simplest model of the three. It achieved the best recall (catching 78% of actual churners) and the best F1, ROC-AUC, and PR-AUC. This was a genuinely interesting result: with a moderately-sized, mostly categorical dataset, a well-regularized linear model held its own against tree-based ensembles — a useful reminder that model complexity isn't automatically correlated with performance.

Confusion matrices and ROC curve comparisons for all three models are available in notebooks/01_exploration.ipynb.

Explainability with SHAP

A model that flags a customer as high-risk isn't actionable on its own — a retention team needs to know why. SHAP (SHapley Additive exPlanations) values are computed and surfaced at two levels:


Global explanations (training time): after the best model is selected, explain.py computes SHAP values across the full test set and saves summary plots to saved_models/, showing which features drive churn risk overall (e.g. contract type, tenure, monthly charges) and in which direction.
Per-prediction explanations (inference time): the Streamlit UI computes a SHAP force/waterfall plot for each individual customer prediction, showing exactly which features pushed that specific customer's risk score up or down — turning "this customer is 82% likely to churn" into "this customer is 82% likely to churn, driven primarily by a month-to-month contract and low tenure, partially offset by low monthly charges."


This closes the loop between "the model works" and "the model is usable by a non-technical stakeholder" — arguably the more important claim for a production-style system.

Generalization Test: E-commerce Dataset

To validate the config-driven architecture's core claim — that the same pipeline code can handle a structurally different dataset by swapping only the config file — the pipeline was pointed at a second dataset: an e-commerce customer churn dataset (5,630 customers, 20 features, mostly behavioral/numerical rather than Telco's mostly-categorical composition).

Pipeline changes required

The pipeline did not run unmodified — and that gap is itself an informative result. Three changes were needed:


Excel file support — the e-commerce dataset ships as .xlsx (with a "Data Dict" sheet alongside the real "E Comm" data sheet), while Telco is .csv. load_raw_data() was updated to branch on file extension.
Generalized missing-value imputation — Telco's original cleaning logic contained a hardcoded, dataset-specific fix (TotalCharges blank-string coercion, imputed with 0). The e-commerce dataset has genuine NaN values spread across 7 different numerical columns (Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder — each ~4.5–5.5% missing). clean_data() was refactored to apply median imputation across all configured numerical columns generically, rather than a single hardcoded column-specific fix. This still correctly handles Telco's TotalCharges case, though with a minor tradeoff: median imputation is less semantically precise than the original "0 tenure = $0 charges" domain-specific reasoning.
CLI-configurable entrypoints — both data_pipeline.py and train.py now accept a --config argument, and artifact filenames are derived from config["dataset"]["name"] so running the pipeline on one dataset never overwrites the other's saved models.


Results — E-commerce Dataset

ModelPrecisionRecallF1ROC-AUCPR-AUCLogistic Regression0.4410.8470.5800.8860.681Random Forest0.9940.8630.9240.9990.994XGBoost0.9690.9790.9740.9990.991

XGBoost was selected (F1 = 0.974) — a stark contrast to Telco, where Logistic Regression won and all models sat in the 0.55–0.61 F1 range.

Investigating the unusually high performance

An F1 score above 0.97 is unusual enough to warrant suspicion rather than celebration — in real-world ML work, results that look "too good" are more often a sign of data leakage than genuine model quality. Before accepting this result, two checks were run:


Correlation analysis across all 13 numerical features against Churn — no single feature showed a strong linear correlation (Tenure: -0.35, Complain: 0.25, all others below 0.16), ruling out an obvious leaking column.
Duplicate check — zero duplicate rows and zero duplicate CustomerIDs, ruling out train/test contamination via repeated records.


Conclusion: the elevated performance is most likely explained by the e-commerce dataset being genuinely more cleanly separable — a combination of its smaller size (5,630 vs. 7,043 rows) and a more curated feature set — rather than a pipeline defect or leakage. Tree-based models (Random Forest, XGBoost) likely captured non-linear interactions across several moderate-strength features that no single correlation coefficient would reveal.

This distinguishes two very different situations that can look identical from a metrics table alone — "the pipeline is broken" versus "the data is genuinely easier" — and reaching a defensible answer required verification, not just accepting a good-looking number.

Setup

bashgit clone https://github.com/crazyEj/Customer-Church-Prediction-System.git
cd Customer-Church-Prediction-System

py -3.12 -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

Place the Telco Customer Churn CSV at data/raw/telco_churn.csv.

Run the pipeline

bashpython src/data_pipeline.py   # clean data, split, fit preprocessor
python src/train.py           # train & compare 3 models, save best
python src/explain.py         # compute SHAP values, save global explanation plots
python src/predict.py         # example inference on sample data
streamlit run app.py          # launch interactive UI with per-prediction SHAP explanations

Lessons Learned


Git hygiene matters from commit #1. A virtual environment accidentally committed early in this project ballooned the repo to 340MB and got rejected by GitHub's file size limits. Resolved by rewriting git history (git init fresh + .gitignore in place before the first commit) — a good reminder to set up .gitignore before running git add . for the first time, not after.
F1 vs. Recall is a business decision, not just a technical one. F1 was chosen here as a balanced default, but a real deployment would involve discussing with stakeholders whether missing a churner (false negative) is costlier than a false alarm (false positive) — that conversation should happen before finalizing a selection metric, not after.
A high score is a question, not an answer. The 0.974 F1 on the e-commerce dataset was treated as a prompt to investigate, not a result to report — checking for leakage before trusting a metric is a habit worth building early.


Future Work


Deployment — containerize with Docker and deploy the Streamlit app (e.g. Streamlit Community Cloud, Render, or AWS) so the live demo doesn't require a local clone.
Hyperparameter tuning (GridSearchCV/Optuna) on the selected models for both datasets, with tracked experiments (e.g. MLflow) rather than one-off runs.
Automated testing — unit tests for the pipeline's cleaning and imputation logic, particularly around the generalized missing-value handling, to guard against regressions when pointing the pipeline at a third dataset.
Monitoring for drift — track feature and prediction distributions over time to detect when a deployed model's assumptions stop holding.
A third, larger-scale dataset to further stress-test the config-driven architecture beyond two examples.
