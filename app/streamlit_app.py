
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import sys
import os

# Get project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("Customer Churn Prediction")
st.caption("Educational decision-support prototype. Review predictions before taking action.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Run `python -m src.train` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, encoding="utf-8") as file:
    threshold = float(json.load(file)["threshold"])

with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone = st.selectbox("Phone service", ["Yes", "No"])
    multiple = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0)
    total = st.number_input("Total charges", min_value=0.0, value=840.0)
    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    probability = float(model.predict_proba(customer)[0, 1])
    prediction = int(probability >= threshold)

    st.metric("Estimated churn probability", f"{probability:.1%}")
    st.write(f"Decision threshold: {threshold:.2f}")
    if prediction:
        st.warning("Model classifies this customer as likely to churn.")
    else:
        st.success("Model classifies this customer as likely to stay.")
