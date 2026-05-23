import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "delivery_delay_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"

# Load saved model and model columns
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Page setup
st.set_page_config(
    page_title="SmartLogistics Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 SmartLogistics Delivery Delay Predictor")

st.write(
    "This app predicts whether a delivery is likely to be delayed using "
    "distance, package weight, traffic level, weather condition, vehicle type, "
    "delivery hour, and driver experience."
)

st.divider()

# User inputs
distance_km = st.number_input(
    "Distance in kilometers",
    min_value=1.0,
    max_value=100.0,
    value=10.0
)

package_weight_kg = st.number_input(
    "Package weight in kilograms",
    min_value=0.5,
    max_value=50.0,
    value=5.0
)

traffic_level = st.selectbox(
    "Traffic level",
    ["Low", "Medium", "High"]
)

weather_condition = st.selectbox(
    "Weather condition",
    ["Clear", "Rainy", "Foggy", "Stormy"]
)

vehicle_type = st.selectbox(
    "Vehicle type",
    ["Bike", "Car", "Van", "Truck"]
)

delivery_hour = st.slider(
    "Delivery hour",
    min_value=6,
    max_value=22,
    value=12
)

driver_experience_years = st.slider(
    "Driver experience in years",
    min_value=0,
    max_value=20,
    value=3
)

# Feature engineering functions
def categorize_distance(distance):
    if distance <= 5:
        return "Short"
    elif distance <= 20:
        return "Medium"
    else:
        return "Long"


def experience_level(years):
    if years < 2:
        return "Beginner"
    elif years <= 5:
        return "Intermediate"
    else:
        return "Experienced"


# Create extra features
is_peak_hour = 1 if delivery_hour in [7, 8, 9, 16, 17, 18] else 0
heavy_package = 1 if package_weight_kg > 20 else 0
distance_category = categorize_distance(distance_km)
driver_experience_level = experience_level(driver_experience_years)

# Put user input into a DataFrame
input_data = pd.DataFrame({
    "distance_km": [distance_km],
    "package_weight_kg": [package_weight_kg],
    "traffic_level": [traffic_level],
    "weather_condition": [weather_condition],
    "vehicle_type": [vehicle_type],
    "delivery_hour": [delivery_hour],
    "driver_experience_years": [driver_experience_years],
    "is_peak_hour": [is_peak_hour],
    "heavy_package": [heavy_package],
    "distance_category": [distance_category],
    "driver_experience_level": [driver_experience_level]
})

# Convert text values into numbers
input_encoded = pd.get_dummies(input_data)

# Match the same columns used during training
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

st.divider()

# Prediction button
if st.button("Predict Delivery Delay"):
    prediction = model.predict(input_encoded)[0]
    prediction_probability = model.predict_proba(input_encoded)[0]

    if prediction == 1:
        st.error("Prediction: This delivery is likely to be delayed.")
    else:
        st.success("Prediction: This delivery is not likely to be delayed.")

    st.write("### Prediction Confidence")
    st.write(f"Not delayed: {round(prediction_probability[0] * 100, 2)}%")
    st.write(f"Delayed: {round(prediction_probability[1] * 100, 2)}%")

    st.write("### Input Summary")
    st.dataframe(input_data)