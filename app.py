import streamlit as st
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin


# Custom Transformer

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["loan_income_ratio"] = (
            X["loan_amount_lakh"]
            /
            (X["annual_income_lakh"] + 1)
        )

        X["financial_stability_score"] = (
            (X["credit_score"] * 0.5)
            +
            (X["savings_balance_lakh"] * 10)
            -
            (X["missed_payments"] * 5)
        )

        X["debt_burden_score"] = (
            X["debt_to_income_ratio"]
            *
            (X["existing_loans"] + 1)
        )

        return X




st.set_page_config(
    page_title="Bank Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)


# Custom CSS

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
}

h1{
    text-align:center;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)


# Load Model

model = joblib.load("model_new.pkl")

# Header


st.title("🏦 Bank Loan Approval Predictor")

st.markdown(
    """
    Predict whether a customer's loan application
    is likely to be approved.
    """
)


# Sidebar Inputs

st.sidebar.header("📋 Customer Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

education_level = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Bachelor", "Master", "PhD"]
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    ["Salaried", "Self-Employed", "Business", "Student"]
)

loan_purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ["Business", "Car", "Education", "Home", "Personal"]
)

annual_income_lakh = st.sidebar.number_input(
    "Annual Income (Lakh)",
    min_value=0.0,
    value=10.0
)

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

years_employed = st.sidebar.number_input(
    "Years Employed",
    min_value=0,
    value=5
)

existing_loans = st.sidebar.number_input(
    "Existing Loans",
    min_value=0,
    value=0
)

debt_to_income_ratio = st.sidebar.number_input(
    "Debt To Income Ratio",
    min_value=0.0,
    value=20.0
)

loan_amount_lakh = st.sidebar.number_input(
    "Loan Amount (Lakh)",
    min_value=0.0,
    value=5.0
)

interest_rate_percent = st.sidebar.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=8.0
)

missed_payments = st.sidebar.number_input(
    "Missed Payments",
    min_value=0,
    value=0
)

savings_balance_lakh = st.sidebar.number_input(
    "Savings Balance (Lakh)",
    min_value=0.0,
    value=5.0
)

# =====================================================
# Input Data
# =====================================================

input_df = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "employment_type": [employment_type],
    "annual_income_lakh": [annual_income_lakh],
    "credit_score": [credit_score],
    "years_employed": [years_employed],
    "existing_loans": [existing_loans],
    "debt_to_income_ratio": [debt_to_income_ratio],
    "loan_amount_lakh": [loan_amount_lakh],
    "interest_rate_percent": [interest_rate_percent],
    "missed_payments": [missed_payments],
    "savings_balance_lakh": [savings_balance_lakh],
    "loan_purpose": [loan_purpose]
})


# Customer Details


display_df = input_df.T.reset_index()
display_df.columns = ["Feature", "Value"]

with st.expander("📋 View Customer Details"):
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

# Prediction Button


if st.button("🔍 Predict Loan Approval"):

    prediction = model.predict(input_df)[0]

    try:
        probability = model.predict_proba(input_df)
        approval_prob = probability[0][1]
    except:
        approval_prob = None

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    if approval_prob is not None:

        st.subheader("📈 Approval Probability")

        st.progress(float(approval_prob))

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Approval Probability",
                f"{approval_prob:.2%}"
            )

        with col2:
            st.metric(
                "Rejection Probability",
                f"{1 - approval_prob:.2%}"
            )

        st.subheader("⚠️ Risk Assessment")

        if approval_prob >= 0.80:
            st.success("🟢 Low Risk Applicant")

        elif approval_prob >= 0.50:
            st.warning("🟡 Medium Risk Applicant")

        else:
            st.error("🔴 High Risk Applicant")