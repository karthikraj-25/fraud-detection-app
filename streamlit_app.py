import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import speech_recognition as sr
import os

# -----------------------------
# Load the trained model
model = joblib.load('fraud_detection_model.pkl')
st.set_page_config(page_title="Bank Fraud Detector", layout="wide")

# -----------------------------
# Voice-to-text using SpeechRecognition
def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio)
            st.success(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("😕 Could not understand audio.")
        except sr.RequestError:
            st.error("⚠️ Could not connect to Google API.")
        except sr.WaitTimeoutError:
            st.error("⌛ Listening timed out.")
    return ""

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

with tab3:
    st.subheader("🎙️ Voice Assistant (Experimental)")
    st.caption("Ask a question like: _What is SMOTE?_ or _Why is this fraud?_")

    q = st.text_input("Type your question or use voice below:")

    if st.button("🎤 Use Voice Input"):
        voice_input = voice_to_text()
        if voice_input:
            q = voice_input

    # Dummy response generator
    if q:
        if "smote" in q.lower():
            st.info("SMOTE is used to balance imbalanced datasets by generating synthetic minority samples.")
        elif "why" in q.lower() and "fraud" in q.lower():
            st.info("Fraud is flagged when patterns look abnormal (e.g., large amount, late night, new location).")
        else:
            st.warning("🤔 Sorry, I couldn't understand your question.")

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
