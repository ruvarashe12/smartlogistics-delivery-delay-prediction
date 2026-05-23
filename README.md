# SmartLogistics Delivery Delay Prediction

SmartLogistics Delivery Delay Prediction is a beginner-friendly data science and machine learning project that predicts whether a delivery is likely to be delayed based on delivery distance, package weight, traffic level, weather condition, vehicle type, delivery hour, and driver experience.

This project was created to solve a real-world logistics problem: delivery delays. In logistics and transport, delays can increase costs, reduce customer satisfaction, and make operations less reliable. This project uses data analysis and machine learning to identify delay patterns and predict delivery risk.

---

## Project Problem

Delivery companies often face delays caused by traffic, weather, long distances, heavy packages, and driver-related factors.

The goal of this project is to answer:

**Can we predict whether a delivery will be delayed using delivery-related information?**

---

## Project Features

- Synthetic delivery dataset creation
- Data cleaning and understanding
- Exploratory data analysis
- Feature engineering
- Machine learning model training
- Random Forest classification model
- Model accuracy evaluation
- Streamlit web app for live predictions

---

## Dataset Columns

The dataset includes:

- `delivery_id`
- `distance_km`
- `package_weight_kg`
- `vehicle_type`
- `traffic_level`
- `weather_condition`
- `driver_experience_years`
- `delivery_hour`
- `delivery_time_minutes`
- `delayed`

The target column is:

```text
delayed