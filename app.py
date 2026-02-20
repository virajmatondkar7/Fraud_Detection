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

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "PAYMENT", "TRANSFER"]
)

if st.button("Predict"):

    # Create empty dataframe with ALL model columns
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    # Fill numeric values
    input_df.at[0, "step"] = step
    input_df.at[0, "amount"] = amount
    input_df.at[0, "oldbalanceOrg"] = oldbalanceOrg
    input_df.at[0, "newbalanceOrig"] = newbalanceOrig
    input_df.at[0, "oldbalanceDest"] = oldbalanceDest
    input_df.at[0, "newbalanceDest"] = newbalanceDest

    # Set correct type column to 1
    type_column = f"type_{transaction_type}"
    if type_column in input_df.columns:
        input_df.at[0, type_column] = 1

    # Ensure correct column order
    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Transaction (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Genuine Transaction (Probability: {probability:.2f})")


