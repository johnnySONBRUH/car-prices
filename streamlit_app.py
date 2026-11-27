import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and the label encoders
data_dict = joblib.load('predictor.pkl')
model = data_dict['model']
encoders = data_dict['encoders']

def predict_price(name, year, km_driven, fuel, seller_type, transmission, owner, seats, max_power, mileage_unit, mileage, engine):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'seats': [seats],
        'max_power (in bph)': [max_power],
        'Mileage Unit': [mileage_unit],
        'Mileage': [mileage],
        'Engine (CC)': [engine]
    })
    
    # Transform categorical features using the specific encoder for each column
    try:
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
    except Exception as e:
        return f"Error transforming data: {e}"

    pred = model.predict(input_data)
    return pred[0]

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Selling Price Predictor")
st.write("Enter car details to predict the selling price.")

# Define input fields based on the dataset
name = st.selectbox('Car Name', ['Maruti', 'Hyundai', 'Toyota', 'Ford', 'Mahindra', 'Tata', 'Volkswagen', 'Renault', 'Chevrolet', 'Honda'])
year = st.number_input('Year', min_value=1990, max_value=2026, value=2015)
km_driven = st.number_input('KM Driven', min_value=0, value=50000)
fuel = st.selectbox('Fuel', ['Diesel', 'Petrol', 'CNG', 'LPG'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
seats = st.selectbox('Seats', [4, 5, 7, 8])
max_power = st.number_input('Max Power (bhp)', min_value=0.0, value=70.0)
mileage_unit = st.selectbox('Mileage Unit', ['kmpl', 'km/kg'])
mileage = st.number_input('Mileage', min_value=0.0, value=20.0)
engine = st.number_input('Engine (CC)', min_value=500, value=1200)

if st.button("Predict Price", type="primary"):
    result = predict_price(name, year, km_driven, fuel, seller_type, transmission, owner, seats, max_power, mileage_unit, mileage, engine)
    if isinstance(result, str):
        st.error(result)
    else:
        st.success(f"Predicted Selling Price: ₹{result:,.2f}")
    
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>Adapted for Car Selling Price Prediction</p>
</div>
""", unsafe_allow_html=True)

