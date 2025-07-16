import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# Load the trained model
model = joblib.load('fraud_detection_model.pkl')
st.set_page_config(page_title="Bank Fraud Detector", layout="wide")

# -----------------------------
st.title("🔍 Bank Transaction Fraud Detection")

tab1, tab2, tab3 = st.tabs(["💸 Prediction", "📊 Fraud Chart", "🎙️ Voice Input"])

with tab1:
    st.subheader("Enter Transaction Details")
    
    # Inputs
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    ttype = st.selectbox("Transaction Type", options=[0, 1])  # 0: Debit, 1: Credit
    location = st.number_input("Location Code", value=1)
    channel = st.selectbox("Channel", options=[0, 1])  # 0: Online, 1: POS
    age = st.number_input("Customer Age", min_value=0, value=30)
    occupation = st.number_input("Occupation Code", value=1)
    duration = st.number_input("Transaction Duration (seconds)", value=10)
    attempts = st.number_input("Login Attempts", value=1)
    balance = st.number_input("Account Balance", value=5000.0)
    hour = st.slider("Transaction Hour", 0, 23, 14)
    day = st.slider("Transaction Day", 1, 31, 15)
    days_prev = st.number_input("Days Since Previous Transaction", value=1)

    if st.button("🚀 Predict Fraud"):
        input_data = np.array([[amount, ttype, location, channel, age, occupation,
                                duration, attempts, balance, hour, day, days_prev]])
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")
        st.info(f"Model Confidence: {proba:.2%}")

with tab2:
    st.subheader("📊 View Sample Fraud Distribution Chart")
    try:
        df = pd.read_csv("bank_transactions_data_2.csv")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='isFraud', palette='Set2')
        st.pyplot(fig)
    except Exception:
        st.warning("Dataset with 'isFraud' column not found or failed to load.")

# -----------------------------
st.divider()
st.header("📂 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        new_data = pd.read_csv(uploaded_file)
        predictions = model.predict(new_data)
        new_data['Predicted_isFraud'] = predictions
        st.write(new_data.head())

        st.download_button("📥 Download Results", new_data.to_csv(index=False).encode(),
                           file_name='fraud_predictions.csv')
    except Exception as e:
        st.error(f"Error processing file: {e}")
