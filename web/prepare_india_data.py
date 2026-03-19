import pandas as pd
import numpy as np
import os

print("=================================================================")
print("  SafeRoute AI — India + US Combined Model")
print("=================================================================")

# ---------------------------------------------------------
# Create synthetic India accident dataset
# ---------------------------------------------------------

def create_india_synthetic():

    print("\n🇮🇳 Preparing India accident dataset...")

    n = 50000

    temperature = np.random.normal(32, 5, n)
    humidity = np.random.normal(70, 10, n)
    visibility = np.random.uniform(1, 10, n)
    wind_speed = np.random.uniform(1, 15, n)

    weather = np.random.choice([0,1,2,3], n)

    junction = np.random.choice([0,1], n, p=[0.7,0.3])
    traffic_signal = np.random.choice([0,1], n, p=[0.6,0.4])

    # ---------------------------------------------------------
    # Hour distribution
    # ---------------------------------------------------------

    probs = [

        0.01,0.01,0.01,0.01,0.01,0.02,   # 0-5
        0.04,0.05,0.06,0.07,0.07,0.07,   # 6-11
        0.06,0.06,0.05,0.05,0.05,0.05,   # 12-17
        0.07,0.07,0.06,0.04,0.03,0.02    # 18-23
    ]

    probs = np.array(probs)
    probs = probs / probs.sum()   # normalize

    hour = np.random.choice(range(24), n, p=probs)

    severity = np.random.choice([1,2,3,4], n, p=[0.4,0.3,0.2,0.1])

    df = pd.DataFrame({
        "temperature":temperature,
        "humidity":humidity,
        "visibility":visibility,
        "wind_speed":wind_speed,
        "weather":weather,
        "junction":junction,
        "traffic_signal":traffic_signal,
        "hour":hour,
        "severity":severity
    })

    print("India dataset created:", len(df))

    return df


# ---------------------------------------------------------
# Load US dataset
# ---------------------------------------------------------

def load_us_dataset():

    print("\n🇺🇸 Loading US accident dataset...")

    path = "US_Accidents_March23.csv"

    if not os.path.exists(path):

        print("US dataset not found. Skipping US data.")

        return pd.DataFrame()

    df = pd.read_csv(path, nrows=100000)

    df = df.rename(columns={
        "Temperature(F)":"temperature",
        "Humidity(%)":"humidity",
        "Visibility(mi)":"visibility",
        "Wind_Speed(mph)":"wind_speed",
        "Severity":"severity"
    })

    cols = [
        "temperature",
        "humidity",
        "visibility",
        "wind_speed",
        "severity"
    ]

    df = df[cols]

    df["weather"] = 0
    df["junction"] = 0
    df["traffic_signal"] = 0
    df["hour"] = np.random.randint(0,24,len(df))

    print("US dataset loaded:", len(df))

    return df


# ---------------------------------------------------------
# Combine datasets
# ---------------------------------------------------------

def combine_datasets():

    df_india = create_india_synthetic()

    df_us = load_us_dataset()

    if len(df_us) > 0:

        df = pd.concat([df_india, df_us], ignore_index=True)

    else:

        df = df_india

    print("\nFinal dataset size:", len(df))

    return df


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

df = combine_datasets()

df.to_csv("final_accident_dataset.csv", index=False)

print("\nDataset saved as: final_accident_dataset.csv")

print("\n✅ Data preparation complete")