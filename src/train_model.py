import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "delivery_data.csv"
MODEL_PATH = BASE_DIR / "models" / "delivery_delay_model.pkl"
COLUMNS_PATH = BASE_DIR / "models" / "model_columns.pkl"


df = pd.read_csv(DATA_PATH)

# Create delayed column if it does not already exist
if "delayed" not in df.columns:
    df["delayed"] = df["delivery_time_minutes"].apply(lambda x: 1 if x > 90 else 0)

# Feature engineering
df["is_peak_hour"] = df["delivery_hour"].apply(
    lambda x: 1 if x in [7, 8, 9, 16, 17, 18] else 0
)

df["heavy_package"] = df["package_weight_kg"].apply(
    lambda x: 1 if x > 20 else 0
)

def categorize_distance(distance):
    if distance <= 5:
        return "Short"
    elif distance <= 20:
        return "Medium"
    else:
        return "Long"

df["distance_category"] = df["distance_km"].apply(categorize_distance)

def experience_level(years):
    if years < 2:
        return "Beginner"
    elif years <= 5:
        return "Intermediate"
    else:
        return "Experienced"

df["driver_experience_level"] = df["driver_experience_years"].apply(experience_level)


features = [
    "distance_km",
    "package_weight_kg",
    "traffic_level",
    "weather_condition",
    "vehicle_type",
    "delivery_hour",
    "driver_experience_years",
    "is_peak_hour",
    "heavy_package",
    "distance_category",
    "driver_experience_level"
]

X = df[features]
y = df["delayed"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model training completed.")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(X.columns.tolist(), COLUMNS_PATH)

print("\nModel saved successfully.")
print(f"Model path: {MODEL_PATH}")
print(f"Columns path: {COLUMNS_PATH}")