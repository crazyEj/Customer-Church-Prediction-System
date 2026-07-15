import streamlit as st
import pandas as pd
import sys
import io
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from data_pipeline import load_config
from predict import load_artifacts, load_training_background, predict_churn, explain_prediction, predict_batch

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

# Load config and artifacts once, cached across reruns for performance.
# Both loaders are dataset-aware (read config["dataset"]["name"]), so
# switching config.yaml to the e-commerce config picks up the right
# preprocessor, model, and SHAP background automatically.
@st.cache_resource
def get_config_and_artifacts():
    config = load_config()
    preprocessor, model = load_artifacts(config)
    X_train = load_training_background(config)
    return config, preprocessor, model, X_train

config, preprocessor, model, X_train = get_config_and_artifacts()

st.title("📊 Customer Churn Prediction")

tab_single, tab_batch = st.tabs(["Single Customer", "Batch Upload"])

# =====================================================================
# Single customer prediction (unchanged from before, just wrapped in a tab)
# =====================================================================
with tab_single:
    st.write("Enter a customer's details below to predict their likelihood of churning.")

    st.divider()

    # --- Demographics ---
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    with col2:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    # --- Account Info ---
    st.subheader("Account Information")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    # --- Services ---
    st.subheader("Services")
    col1, col2 = st.columns(2)
    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    with col2:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # --- Billing ---
    st.subheader("Billing")
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0, step=1.0)

    st.divider()

    if st.button("Predict Churn", type="primary", use_container_width=True):
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
        probability = result["Churn_Probability"].iloc[0]

        st.divider()
        if prediction == "Yes":
            st.error(f"⚠️ **High Churn Risk** — Predicted probability: {probability:.1%}")
        else:
            st.success(f"✅ **Low Churn Risk** — Predicted probability: {probability:.1%}")

        st.progress(float(probability))

        st.divider()
        st.subheader("Why this prediction?")
        explanations = explain_prediction(input_data, config, preprocessor, model, X_train)

        for exp in explanations:
            icon = "🔺" if "increases" in exp["effect"] else "🔻"
            st.write(f"{icon} **{exp['feature']}** ({exp['effect']}, impact: {exp['shap_value']:+.3f})")

# =====================================================================
# Batch prediction — upload a CSV/Excel of customers, score them all,
# and download results (with per-row SHAP driver summaries) as CSV or Excel.
# =====================================================================
with tab_batch:
    st.write(
        "Upload a CSV or Excel file with one row per customer "
        "(same columns as the single-customer form above) to score them all at once."
    )

    uploaded_file = st.file_uploader("Upload customer file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                batch_df = pd.read_excel(uploaded_file)
            else:
                batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            batch_df = None

        if batch_df is not None:
            st.write(f"Loaded **{len(batch_df)}** customers.")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                try:
                    with st.spinner(f"Scoring {len(batch_df)} customers..."):
                        results = predict_batch(batch_df, config, preprocessor, model, X_train)
                except Exception as e:
                    st.error(
                        "Batch scoring failed — check that the uploaded file's columns "
                        f"match the expected schema (same fields as the single-customer form).\n\n"
                        f"**Error:** {e}"
                    )
                else:
                    st.divider()
                    n_high_risk = int((results["Churn_Prediction"] == "Yes").sum())
                    col1, col2 = st.columns(2)
                    col1.metric("Total Customers", len(results))
                    col2.metric("High Risk (Predicted Churn)", n_high_risk)

                    st.dataframe(results, use_container_width=True)

                    # Build both export formats
                    csv_bytes = results.to_csv(index=False).encode("utf-8")

                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        results.to_excel(writer, index=False, sheet_name="Predictions")
                    excel_bytes = excel_buffer.getvalue()

                    st.divider()
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        st.download_button(
                            "⬇️ Download CSV",
                            data=csv_bytes,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with dl_col2:
                        st.download_button(
                            "⬇️ Download Excel",
                            data=excel_bytes,
                            file_name="churn_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )