import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")

df = pd.read_csv(
"/Users/nikhil/Downloads/sem4/new/US_Accidents_March23.csv",
usecols=[
"Severity",
"Temperature(F)",
"Humidity(%)",
"Visibility(mi)",
"Wind_Speed(mph)",
"Weather_Condition",
"Junction",
"Traffic_Signal"
],
nrows=100000
)

df = df.rename(columns={
"Temperature(F)":"temperature",
"Humidity(%)":"humidity",
"Visibility(mi)":"visibility",
"Wind_Speed(mph)":"wind_speed",
"Weather_Condition":"weather",
"Traffic_Signal":"traffic_signal"
})

df = df.dropna()

encoder = LabelEncoder()
df["weather"] = encoder.fit_transform(df["weather"])

df["Junction"] = df["Junction"].astype(int)
df["traffic_signal"] = df["traffic_signal"].astype(int)

df.to_csv("processed_data.csv", index=False)

print("Preprocessing complete")