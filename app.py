import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and metadata
model = joblib.load('house_model.pkl')
model_columns = joblib.load('model_columns.pkl')
locations = joblib.load('locations.pkl')

st.title("🏡 Ahmedabad House Price Predictor")
st.write("Calculate the estimated price of properties in the Ahmedabad area.")

# Input fields
bhk = st.number_input("Select BHK", min_value=1, max_value=10, value=2)
area = st.number_input("Total Area (sq. ft.)", min_value=300, max_value=10000, value=1000)
location = st.selectbox("Select Location", locations)

if st.button("💰 Predict Market Price"):
    # Create empty dataframe matching training format
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Fill numeric features
    input_df.at[0, 'bhk'] = bhk
    input_df.at[0, 'area_sqft'] = area
    
    # Fill location encoding
    loc_col = f'location_{location}'
    if loc_col in model_columns:
        input_df.at[0, loc_col] = 1
        
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    if prediction >= 100:
        st.success(f"### Estimated Price: ₹{prediction/100:.2f} Crore")
    else:
        st.success(f"### Estimated Price: ₹{prediction:.2f} Lakhs")