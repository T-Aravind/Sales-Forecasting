import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your model and features
model = joblib.load('sales_model.pkl')
features = joblib.load('feature_names.pkl')

st.title("💰 Sales Prediction Dashboard")
st.write("Enter the details below to get an instant forecast.")

# Create an input for every feature the model expects
input_data = {}
for col in features:
    input_data[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict Revenue"):
    # Convert inputs to DataFrame
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    st.success(f"Predicted Sales: ${prediction:,.2f}")