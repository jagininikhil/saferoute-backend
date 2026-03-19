import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

FILE_PATH = "US_Accidents_March23.csv"

print("Loading dataset...")
df = pd.read_csv(FILE_PATH, nrows=300000)

features = [
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Weather_Condition",
    "Junction",
    "Traffic_Signal"
]

target = "Severity"

df = df[features + [target]].dropna()

print("Encoding weather...")
encoder = LabelEncoder()
df["Weather_Condition"] = encoder.fit_transform(df["Weather_Condition"])

rename = {
    "Temperature(F)": "temperature",
    "Humidity(%)": "humidity",
    "Visibility(mi)": "visibility",
    "Wind_Speed(mph)": "wind_speed",
    "Weather_Condition": "weather",
    "Junction": "junction",
    "Traffic_Signal": "traffic_signal"
}

df.rename(columns=rename, inplace=True)

feature_names = list(rename.values())

X = df[feature_names]
y = df[target]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

joblib.dump(model, "accident_model.pkl")
joblib.dump(feature_names, "model_features.pkl")
joblib.dump(encoder, "weather_encoder.pkl")

print("Model saved successfully")