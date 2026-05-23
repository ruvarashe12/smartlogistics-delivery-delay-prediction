import pandas as pd
import numpy as np
import random

# This makes sure we get the same random data each time we run the file
np.random.seed(42)
random.seed(42)

# Number of deliveries we want to create
number_of_deliveries = 1000

# Possible values for some columns
vehicle_types = ["Bike", "Van", "Truck"]
traffic_levels = ["Low", "Medium", "High"]
weather_conditions = ["Clear", "Rainy", "Foggy", "Stormy"]

# Create the dataset
data = {
    "delivery_id": [f"DLV{i}" for i in range(1, number_of_deliveries + 1)],

    "distance_km": np.random.randint(1, 31, number_of_deliveries),

    "package_weight_kg": np.random.randint(1, 21, number_of_deliveries),

    "vehicle_type": np.random.choice(vehicle_types, number_of_deliveries),

    "traffic_level": np.random.choice(traffic_levels, number_of_deliveries),

    "weather_condition": np.random.choice(weather_conditions, number_of_deliveries),

    "driver_experience_years": np.random.randint(0, 11, number_of_deliveries),

    "delivery_hour": np.random.randint(6, 23, number_of_deliveries),
}

df = pd.DataFrame(data)

# Create delivery time based on real-world logic
df["delivery_time_minutes"] = (
    15
    + df["distance_km"] * 2
    + df["package_weight_kg"] * 1
    + df["traffic_level"].map({
        "Low": 0,
        "Medium": 10,
        "High": 25
    })
    + df["weather_condition"].map({
        "Clear": 0,
        "Rainy": 10,
        "Foggy": 15,
        "Stormy": 25
    })
)

# Add peak hour delay
df["delivery_time_minutes"] = df.apply(
    lambda row: row["delivery_time_minutes"] + 15
    if row["delivery_hour"] in [7, 8, 16, 17, 18]
    else row["delivery_time_minutes"],
    axis=1
)

# Experienced drivers reduce delivery time slightly
df["delivery_time_minutes"] = df["delivery_time_minutes"] - (df["driver_experience_years"] * 1.5)

# Add small random noise so the data is not too perfect
df["delivery_time_minutes"] = df["delivery_time_minutes"] + np.random.randint(-5, 6, number_of_deliveries)

# Make sure delivery time is never below 10 minutes
df["delivery_time_minutes"] = df["delivery_time_minutes"].apply(lambda x: max(10, round(x, 1)))

# Create delay status
# 1 means delayed
# 0 means on time
df["delay_status"] = df["delivery_time_minutes"].apply(lambda x: 1 if x > 70 else 0)

# Add a few missing values on purpose so we can learn data cleaning later
df.loc[5, "traffic_level"] = np.nan
df.loc[20, "weather_condition"] = np.nan
df.loc[50, "distance_km"] = np.nan

# Add a duplicate row on purpose so we can remove it later
df = pd.concat([df, df.iloc[[10]]], ignore_index=True)

# Save the dataset inside the data folder
df.to_csv("data/delivery_data.csv", index=False)

print("Dataset created successfully!")
print("File saved as: data/delivery_data.csv")
print(df.head()) 