import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the trained models and scaler
xgboost_model = joblib.load("xgboost_model.joblib")
mlp_model = load_model("mlp_model.keras")
scaler = joblib.load("scaler.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Function to preprocess input
def preprocess_input(longitude, latitude, date, time, cross_street, area_id, reporting_district_no, part_1_2, premise_code, victim_age, victim_sex, victim_descent, premise_description, area_name, weapon_used_code, weapon_description, status):
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    time_obj = datetime.strptime(time, '%H:%M')
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    hour = time_obj.hour

    # Create input with exactly 19 features (Same as training)
    input_data = pd.DataFrame({
        'Longitude': [longitude],
        'Latitude': [latitude],
        'Hour': [hour],
        'Day': [day],
        'Month': [month],
        'Year': [year],
        'Cross_Street': [cross_street],
        'Area_ID': [area_id],
        'Reporting_District_no': [reporting_district_no],
        'Part 1-2': [part_1_2],
        'Premise_Code': [premise_code],
        'Victim_Age': [victim_age],
        'Victim_Sex': [victim_sex],
        'Victim_Descent': [victim_descent],
        'Premise_Description': [premise_description],
        'Area_Name': [area_name],
        'Weapon_Used_Code': [weapon_used_code],
        'Weapon_Description': [weapon_description],
        'Status': [status]
    })

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

# Function to predict crime category
def predict_crime_category(input_data_scaled):
    # Predict probabilities using XGBoost model
    sample_prob_xgb = xgboost_model.predict_proba(input_data_scaled)

    # Predict probabilities using MLP model
    sample_prob_mlp = mlp_model.predict(input_data_scaled)

    # Combine the probabilities by averaging
    combined_prob = (sample_prob_xgb + sample_prob_mlp) / 2

    # Determine the final prediction based on the highest average probability
    combined_prediction = np.argmax(combined_prob, axis=1)
    predicted_crime_category = label_encoders['Crime_Category'].inverse_transform(combined_prediction)

    return predicted_crime_category[0]

# Streamlit UI
st.title("Crime Classification Model Deployment")
st.write("Enter details below to predict the crime category:")

# User Inputs
longitude = st.number_input("Enter Longitude:", format="%.6f")
latitude = st.number_input("Enter Latitude:", format="%.6f")
date = st.date_input("Enter Date:").strftime('%Y-%m-%d')
time = st.time_input("Enter Time:").strftime('%H:%M')
cross_street = st.number_input("Enter Cross Street:", min_value=0)
area_id = st.number_input("Enter Area ID:", min_value=0)
reporting_district_no = st.number_input("Enter Reporting District No:", min_value=0)
part_1_2 = st.number_input("Enter Part 1-2:", min_value=0)
premise_code = st.number_input("Enter Premise Code:", min_value=0)
victim_age = st.number_input("Enter Victim Age:", min_value=0)
victim_sex = st.number_input("Enter Victim Sex:", min_value=0)
victim_descent = st.number_input("Enter Victim Descent:", min_value=0)
premise_description = st.number_input("Enter Premise Description:", min_value=0)
area_name = st.number_input("Enter Area Name:", min_value=0)
weapon_used_code = st.number_input("Enter Weapon Used Code:", min_value=0)
weapon_description = st.number_input("Enter Weapon Description:", min_value=0)
status = st.number_input("Enter Status:", min_value=0)

# Predict button
if st.button("Predict"):
    input_data_scaled = preprocess_input(longitude, latitude, date, time, cross_street, area_id, reporting_district_no, part_1_2, premise_code, victim_age, victim_sex, victim_descent, premise_description, area_name, weapon_used_code, weapon_description, status)
    result = predict_crime_category(input_data_scaled)
    st.write(f"Predicted Crime Category: {result}")