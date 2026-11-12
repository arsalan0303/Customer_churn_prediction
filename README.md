
# Customer Churn Prediction — End-to-End Data Science Project

A portfolio-ready machine-learning project covering business understanding, EDA, preprocessing, feature engineering, model comparison, cross-validation, hyperparameter tuning, threshold selection, final evaluation, SHAP interpretation, API deployment, Streamlit UI, and monitoring.

## 1. Dataset

This project is designed for the IBM Telco Customer Churn CSV commonly named:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The CSV is not included because dataset redistribution rights may vary. Do not commit private or restricted customer data.

Expected target: `Churn` (`Yes` / `No`).

## 2. Environment setup

Python 3.10–3.12 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Run the notebook

Open `notebooks/01_end_to_end_churn_project.ipynb` in Jupyter or VS Code.

Run cells from top to bottom. The notebook:
- loads and validates the CSV;
- performs EDA;
- creates train/validation/test splits;
- builds leakage-safe preprocessing and feature engineering;
- compares three classifiers using stratified cross-validation;
- tunes Logistic Regression with GridSearchCV;
- selects a threshold using validation data;
- evaluates once on the held-out test set;
- generates SHAP explanations when SHAP is installed;
- saves the fitted pipeline and metadata under `models/`.

## 4. Train from the command line

After placing the dataset in `data/raw/`:

```bash
python -m src.train
```

The training script saves:
- `models/churn_pipeline.joblib`
- `models/metadata.json`
- `reports/model_evaluation.json`

## 5. Run the FastAPI service

After training:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

The API expects all model input fields. The request schema is defined in `app/main.py`.

## 6. Run the Streamlit app

After training:

```bash
streamlit run app/streamlit_app.py
```

## 7. Run tests

```bash
pytest -q
```

Tests require a trained model for the model smoke test. API schema tests can run independently.

## 8. Important methodology notes

- The test set is kept separate from model selection and threshold selection.
- Imputation, scaling, encoding and feature engineering are inside the pipeline.
- Stratified splitting is used for this non-temporal example. Use chronological splitting for time-dependent prediction and group splitting when multiple records per customer could cross splits.
- The example threshold objective is validation F1. In a business deployment, choose a threshold based on intervention capacity and the costs/benefits of false positives and false negatives.
- SHAP explanations describe model behavior; they do not establish causality.
- Drift alerts should prompt investigation, not automatic retraining.
- This dataset is for learning. It is not evidence of performance on a real telecom company's current population.

## 9. Repository structure

```text
customer-churn-prediction/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── .gitkeep
├── notebooks/
│   └── 01_end_to_end_churn_project.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_api_schema.py
│   └── test_model.py
├── .gitignore
├── requirements.txt
└── README.md
```
