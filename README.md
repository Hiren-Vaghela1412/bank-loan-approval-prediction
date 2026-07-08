# 🏦 Bank Loan Approval Prediction App

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://bank-loan-approval-prediction-fe4kijik6xgrncj9zizwww.streamlit.app/)

A Machine Learning-powered Streamlit web application that predicts whether a customer's loan application is likely to be approved based on demographic, employment, financial, and credit-related information.

## 🚀 Project Overview

This project applies classification algorithms to predict loan approval decisions for banking customers. The application provides real-time predictions through an interactive Streamlit interface.

### Key Features

- Loan Approval Prediction
- Approval Probability Score
- Risk Assessment
- Automated Feature Engineering
- Interactive Streamlit Dashboard
- End-to-End Machine Learning Pipeline

---

## 📊 Dataset Features

### Customer Information

- Age
- Gender
- Marital Status
- Education Level
- Employment Type

### Financial Information

- Annual Income (Lakh)
- Credit Score
- Years Employed
- Existing Loans
- Debt-to-Income Ratio
- Loan Amount (Lakh)
- Interest Rate (%)
- Missed Payments
- Savings Balance (Lakh)
- Loan Purpose

---

## ⚙️ Feature Engineering

Three custom features were created to improve model performance:

### Loan Income Ratio

```python
loan_amount_lakh / (annual_income_lakh + 1)
```

### Financial Stability Score

```python
(credit_score * 0.5)
+
(savings_balance_lakh * 10)
-
(missed_payments * 5)
```

### Debt Burden Score

```python
debt_to_income_ratio * (existing_loans + 1)
```

---

## 🤖 Models Evaluated

The following machine learning algorithms were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------:|
| Logistic Regression | 99.37% | 1.000 | 0.974 | 0.987 |
| Gaussian Naive Bayes | 94.28% | 0.996 | 0.765 | 0.865 |
| K-Nearest Neighbors | 82.22% | 0.711 | 0.440 | 0.543 |
| Bernoulli Naive Bayes | 76.09% | 0.667 | 0.012 | 0.024 |
| Multinomial Naive Bayes | 75.94% | 0.000 | 0.000 | 0.000 |

### Best Performing Model

🏆 **Logistic Regression** achieved the highest overall performance across all evaluation metrics and was selected as the final production model.

---

## 🔄 Machine Learning Pipeline

### Data Preprocessing

- Missing Value Imputation
- Min-Max Scaling
- Ordinal Encoding
- Categorical Encoding
- Feature Engineering

### Model Training

- Logistic Regression
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- K-Nearest Neighbors

### Model Selection

Models were compared using:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## 📸 Application Preview

Add your Streamlit screenshot here:

```text
screenshots/app.png
```

---

## 📂 Project Structure

```text
bank-loan-approval-prediction-streamlit/
│
├── app.py
├── model.pkl
├── requirements.txt
├── Bank_Loan_Classification.ipynb
├── README.md
```

---

## 📈 Application Output

The application provides:

- Loan Approval Prediction
- Approval Probability
- Rejection Probability
- Risk Assessment

### Risk Levels

🟢 Low Risk Applicant

🟡 Medium Risk Applicant

🔴 High Risk Applicant

---

## 👨‍💻 Author

**Hiren Vaghela**

Aspiring Data Scientist | Machine Learning Enthusiast | Power BI Developer

GitHub: https://github.com/your-github-username
