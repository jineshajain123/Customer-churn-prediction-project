import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
    
st.header('Customer Churn Prediction ML Model')

# Collect User Inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=600.0)

# Categorical Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
phone_service = st.selectbox("Has Phone Service?", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# 🔹 Encode Categorical Features (Apply the same encoding used during training)
gender_encoded = 1 if gender == "Male" else 0
senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
partner_encoded = 1 if partner == "Yes" else 0
dependents_encoded = 1 if dependents == "Yes" else 0
phone_service_encoded = 1 if phone_service == "Yes" else 0

multiple_lines_mapping = {"No": 0, "Yes": 1, "No phone service": 2}
internet_service_mapping = {"DSL": 0, "Fiber optic": 1, "No": 2}
online_security_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
online_backup_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
device_protection_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
tech_support_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
streaming_tv_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
streaming_movies_mapping = {"No": 0, "Yes": 1, "No internet service": 2}
contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
payment_method_mapping = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}

# Apply mappings
multiple_lines_encoded = multiple_lines_mapping[multiple_lines]
internet_service_encoded = internet_service_mapping[internet_service]
online_security_encoded = online_security_mapping[online_security]
online_backup_encoded = online_backup_mapping[online_backup]
device_protection_encoded = device_protection_mapping[device_protection]
tech_support_encoded = tech_support_mapping[tech_support]
streaming_tv_encoded = streaming_tv_mapping[streaming_tv]
streaming_movies_encoded = streaming_movies_mapping[streaming_movies]
contract_encoded = contract_mapping[contract]
payment_method_encoded = payment_method_mapping[payment_method]

# Create input array with numerical features only
input_data = np.array([[
    tenure, monthly_charges, total_charges, gender_encoded, senior_citizen_encoded, 
    partner_encoded, dependents_encoded, phone_service_encoded, multiple_lines_encoded, 
    internet_service_encoded, online_security_encoded, online_backup_encoded, 
    device_protection_encoded, tech_support_encoded, streaming_tv_encoded, 
    streaming_movies_encoded, contract_encoded, paperless_billing_encoded, 
    payment_method_encoded
]])

# Predict only if button is clicked
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    churn_result = "Yes" if prediction == 1 else "No"
    st.write(f"### Churn Prediction: {churn_result}")
