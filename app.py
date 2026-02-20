import streamlit as st
import joblib
import pandas as pd

# Load trained model and column names
model = joblib.load("fraud_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Fraud Detection App")

st.write("Enter transaction details below:")

input_data = {}

# Dynamically create input fields for each column
for col in model_columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud Transaction (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Genuine Transaction (Probability: {probability:.2f})")
