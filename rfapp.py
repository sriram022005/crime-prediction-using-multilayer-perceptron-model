import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the trained models and scaler
rf_model = joblib.load("random_forest_model.joblib")
mlp_model = load_model("mlp_modelrf.keras")
scaler = joblib.load("scalerrf.joblib")
label_encoders = joblib.load("label_encodersrf.joblib")

# Function to preprocess input
def preprocess_input(longitude, latitude, time):
    time_obj = datetime.strptime(time, '%H:%M')
    hour = time_obj.hour

    # Create input with exactly 3 features (Same as training)
    input_data = pd.DataFrame({
        'Longitude': [longitude],
        'Latitude': [latitude],
        'Hour': [hour]
    })

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

# Function to predict crime category
def predict_crime_category(input_data_scaled):
    
    sample_prob_rf = rf_model.predict_proba(input_data_scaled)

   
    sample_prob_mlp = mlp_model.predict(input_data_scaled)

   
    combined_prob = (sample_prob_rf + sample_prob_mlp) / 2

    
    combined_prediction = np.argmax(combined_prob, axis=1)
    predicted_crime_category = label_encoders['Crime_Category'].inverse_transform(combined_prediction)

    return predicted_crime_category[0]


st.title("Crime Classification Model Deployment")
st.write("Enter details below to predict the crime category:")


longitude = st.number_input("Enter Longitude:", format="%.6f")
latitude = st.number_input("Enter Latitude:", format="%.6f")
time = st.time_input("Enter Time:").strftime('%H:%M')

if st.button("Predict"):
    input_data_scaled = preprocess_input(longitude, latitude, time)
    result = predict_crime_category(input_data_scaled)
    st.write(f"Predicted Crime Category: {result}")