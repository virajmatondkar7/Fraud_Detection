import streamlit as st
import joblib
import pandas as pd

model = joblib.load("fraud_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Fraud Detection App")

# User Inputs
step = st.number_input("Step", value=1)
amount = st.number_input("Amount", value=1000.0)
oldbalanceOrg = st.number_input("Old Balance Origin", value=10000.0)
newbalanceOrig = st.number_input("New Balance Origin", value=9000.0)
oldbalanceDest = st.number_input("Old Balance Destination", value=0.0)
newbalanceDest = st.number_input("New Balance Destination", value=1000.0)

# Dropdown for transaction type
transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "PAYMENT", "TRANSFER"]
)

# Create input dictionary
input_data = {
    "step": step,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
}

# Add one-hot encoding manually
for col in model_columns:
    if col.startswith("type_"):
        input_data[col] = 1 if col == f"type_{transaction_type}" else 0

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Transaction (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Genuine Transaction (Probability: {probability:.2f})")

