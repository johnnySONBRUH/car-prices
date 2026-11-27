import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and the label encoder
data_dict = joblib.load('predictor.pkl')
model = data_dict['model']
le = data_dict['enc']

# Note: The training method used in main.py is problematic (reusing the same label encoder).
# To make this app work, we have to assume the 'enc' object in predictor.pkl 
# somehow holds all the necessary mappings. In a real scenario, this should be fixed
# by training with separate encoders for each categorical column.

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
    
    # Transform categorical features
    # This will likely fail or produce wrong results due to the training method
    try:
        input_data['name'] = le.transform(input_data['name'])
        input_data['fuel'] = le.transform(input_data['fuel'])
        input_data['seller_type'] = le.transform(input_data['seller_type'])
        input_data['transmission'] = le.transform(input_data['transmission'])
        input_data['owner'] = le.transform(input_data['owner'])
        input_data['Mileage Unit'] = le.transform(input_data['Mileage Unit'])
    except Exception as e:
        return f"Error transforming data: {e}"

    pred = model.predict(input_data)
    return pred[0]

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Selling Price Predictor")
st.write("Enter car details to predict the selling price.")

# Define input fields based on the dataset
name = st.text_input('Car Name', 'Maruti')
year = st.number_input('Year', min_value=1990, max_value=2026, value=2015)
km_driven = st.number_input('KM Driven', min_value=0, value=50000)
fuel = st.selectbox('Fuel', ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
seats = st.number_input('Seats', min_value=2, max_value=10, value=5)
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

