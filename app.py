import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import joblib

sys.path.append(str(Path(__file__).parent / "src"))
from data_pipeline import load_config
from predict import load_artifacts, predict_churn, explain_prediction

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Custom styling — Streamlit's defaults look like every other Streamlit app.
# This injects real CSS: a dark, gradient-accented theme with card-style
# panels, closer to a modern SaaS product than a default data-science form.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b0f19 0%, #111827 100%);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: #e5e7eb !important;
    }
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #9ca3af !important;
        font-size: 1.05rem;
    }
    .card {
        background: #1a2234;
        border: 1px solid #2a3450;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #818cf8 !important;
        margin-bottom: 0.75rem;
    }
    .result-high {
        background: linear-gradient(135deg, #3f1d2e, #2a1220);
        border: 1px solid #7f1d3d;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #10302a, #0d2a22);
        border: 1px solid #14532d;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        text-align: center;
    }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af !important;
    }
    .result-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }
    .driver-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid #2a3450;
    }
    .driver-row:last-child { border-bottom: none; }
    .stButton > button {
        background: linear-gradient(90deg, #6366f1, #38bdf8);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 0;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.88; }
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #0f1522 !important;
        border-radius: 8px !important;
        border: 1px solid #2a3450 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_config_and_artifacts():
    config = load_config()
    preprocessor, model = load_artifacts(config)
    X_train, X_test, y_train, y_test = joblib.load("data/processed/train_test_data.pkl")
    return config, preprocessor, model, X_train

config, preprocessor, model, X_train = get_config_and_artifacts()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>Customer Churn Predictor</h1>
    <p>Enter a customer's profile to see their churn risk — and exactly why.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input form, grouped into cards
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Account</div>', unsafe_allow_html=True)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Services</div>', unsafe_allow_html=True)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("Predict Churn Risk", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if predict_clicked:
    input_data = pd.DataFrame([{
        "customerID": "MANUAL-INPUT",
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    result = predict_churn(input_data, config, preprocessor, model)
    prediction = result["Churn_Prediction"].iloc[0]
    probability = float(result["Churn_Probability"].iloc[0])

    st.write("")
    result_col, driver_col = st.columns([1, 1.4])

    with result_col:
        css_class = "result-high" if prediction == "Yes" else "result-low"
        label = "High Churn Risk" if prediction == "Yes" else "Low Churn Risk"
        color = "#f87171" if prediction == "Yes" else "#4ade80"
        st.markdown(f"""
        <div class="{css_class}">
            <div class="result-label">{label}</div>
            <div class="result-value" style="color:{color};">{probability:.0%}</div>
            <div class="result-label">predicted churn probability</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(probability)

    with driver_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Why This Prediction</div>', unsafe_allow_html=True)

        explanations = explain_prediction(input_data, config, preprocessor, model, X_train)
        for exp in explanations:
            up = "increases" in exp["effect"]
            arrow = "▲" if up else "▼"
            arrow_color = "#f87171" if up else "#4ade80"
            feature_clean = exp["feature"].replace("num__", "").replace("cat__", "").replace("_", " ")
            st.markdown(f"""
            <div class="driver-row">
                <span>{feature_clean}</span>
                <span style="color:{arrow_color}; font-weight:700;">{arrow} {abs(exp['shap_value']):.3f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)