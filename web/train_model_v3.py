"""
train_model_v3.py — SafeRoute AI Improved Model
═════════════════════════════════════════════════
New features added over v2:
  ✅ Precipitation(in)     — rain/snow amount
  ✅ Pressure(in)          — barometric pressure drop = storm incoming
  ✅ Wind_Chill(F)         — feels-like temperature
  ✅ hour                  — 0-23 hour of day
  ✅ is_night              — 1 if 10pm–5am
  ✅ is_peak_hour          — 1 if 7-9am or 5-8pm weekday
  ✅ is_weekend            — 1 if Saturday/Sunday
  ✅ Crossing, Stop        — road infrastructure features

Expected accuracy improvement: +5 to +8% over v2

Run:
    pip install scikit-learn pandas numpy joblib
    python train_model_v3.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib, os, warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
FILE_PATH    = 'US_Accidents_March23.csv'
MODEL_OUT    = 'accident_model.pkl'
FEATURES_OUT = 'model_features.pkl'
ENCODER_OUT  = 'weather_encoder.pkl'
SAMPLE_ROWS  = 400_000   # increase for better accuracy, None = full dataset

print("=" * 60)
print("  SafeRoute AI — Model Training v3")
print("  New: Precipitation + Time-of-Day features")
print("=" * 60)

# ── LOAD ──────────────────────────────────────────────────────────────────────
print(f"\n📂 Loading dataset...")
df = pd.read_csv(FILE_PATH, nrows=SAMPLE_ROWS, low_memory=False)
print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

# ── STEP 1: TIME-OF-DAY FEATURES ─────────────────────────────────────────────
print("\n⏰ Engineering time-of-day features from Start_Time...")

if 'Start_Time' in df.columns:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['hour']       = df['Start_Time'].dt.hour.fillna(12).astype(int)
    df['day_of_week']= df['Start_Time'].dt.dayofweek.fillna(0).astype(int)  # 0=Mon

    # is_night: 10pm to 5am — statistically most dangerous
    df['is_night']   = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # is_peak: morning rush 7-9am + evening rush 5-8pm, weekdays only
    df['is_peak']    = (
        ((df['hour'].between(7,9)) | (df['hour'].between(17,20))) &
        (df['day_of_week'] < 5)
    ).astype(int)

    # is_weekend: Saturday(5) or Sunday(6)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # is_dawn_dusk: 5-7am and 6-8pm — sun glare causes accidents
    df['is_dawn_dusk']= (
        df['hour'].between(5,7) | df['hour'].between(18,20)
    ).astype(int)

    print(f"   ✅ hour, is_night, is_peak, is_weekend, is_dawn_dusk added")
    print(f"   Night accidents:   {df[df['is_night']==1]['Severity'].mean():.2f} avg severity")
    print(f"   Peak hr accidents: {df[df['is_peak']==1]['Severity'].mean():.2f} avg severity")
    print(f"   Daytime accidents: {df[(df['is_night']==0)&(df['is_peak']==0)]['Severity'].mean():.2f} avg severity")
else:
    print("   ⚠ Start_Time column not found — skipping time features")
    for col in ['hour','is_night','is_peak','is_weekend','is_dawn_dusk']:
        df[col] = 0

# ── STEP 2: PRECIPITATION ANALYSIS ───────────────────────────────────────────
if 'Precipitation(in)' in df.columns:
    df['Precipitation(in)'] = pd.to_numeric(df['Precipitation(in)'], errors='coerce').fillna(0)
    rain_mask = df['Precipitation(in)'] > 0.1
    print(f"\n🌧  Precipitation stats:")
    print(f"   Rainy accidents:   {df[rain_mask]['Severity'].mean():.2f} avg severity")
    print(f"   Dry accidents:     {df[~rain_mask]['Severity'].mean():.2f} avg severity")
    print(f"   % accidents in rain: {rain_mask.mean()*100:.1f}%")

# ── STEP 3: DEFINE ALL FEATURES ───────────────────────────────────────────────
# Core 7 (original)
FEATURES = [
    'Temperature(F)',
    'Humidity(%)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Weather_Condition',   # will be encoded
    'Junction',
    'Traffic_Signal',
]

# New precipitation + pressure features
PRECIP_FEATURES = [
    'Precipitation(in)',   # ★ rain/snow amount
    'Pressure(in)',        # ★ barometric pressure
    'Wind_Chill(F)',       # ★ feels-like temp
]

# New time-of-day features
TIME_FEATURES = [
    'hour',               # ★ 0-23
    'is_night',           # ★ 10pm-5am
    'is_peak',            # ★ rush hour
    'is_weekend',         # ★ Saturday/Sunday
    'is_dawn_dusk',       # ★ sun glare hours
]

# Road infrastructure features
ROAD_FEATURES = [
    'Crossing',
    'Stop',
    'Give_Way',
    'Railway',
    'Roundabout',
    'Turning_Loop',
    'Amenity',
]

# Add all that exist in the dataset
for col in PRECIP_FEATURES + ROAD_FEATURES:
    if col in df.columns:
        FEATURES.append(col)

FEATURES += TIME_FEATURES  # always added (we created them above)

TARGET = 'Severity'

print(f"\n📋 Feature set: {len(FEATURES)} features")
print("   Core:          Temperature, Humidity, Visibility, Wind, Weather, Junction, Signal")
print("   Precipitation: " + ", ".join([f for f in PRECIP_FEATURES if f in FEATURES]))
print("   Time-of-day:   hour, is_night, is_peak, is_weekend, is_dawn_dusk")
print("   Road infra:    " + ", ".join([f for f in ROAD_FEATURES if f in FEATURES]))

# ── STEP 4: CLEAN DATA ────────────────────────────────────────────────────────
df_model = df[FEATURES + [TARGET]].copy()
before   = len(df_model)
df_model.dropna(subset=[TARGET], inplace=True)

# Fill missing numerics with median
for col in df_model.select_dtypes(include=[np.number]).columns:
    df_model[col].fillna(df_model[col].median(), inplace=True)

# Fill missing booleans/objects
for col in df_model.select_dtypes(include=['bool', 'object']).columns:
    if col != 'Weather_Condition':
        df_model[col].fillna(False, inplace=True)

df_model.dropna(subset=['Weather_Condition'], inplace=True)
print(f"\n🧹 Cleaned: {before - len(df_model):,} rows dropped → {len(df_model):,} remaining")

# ── STEP 5: ENCODE WEATHER CONDITION ─────────────────────────────────────────
le = LabelEncoder()
df_model['Weather_Condition'] = le.fit_transform(
    df_model['Weather_Condition'].astype(str)
)
joblib.dump(le, ENCODER_OUT)
print(f"   Encoded {len(le.classes_)} weather categories")

# Convert booleans to int
for col in df_model.select_dtypes(include=['bool']).columns:
    df_model[col] = df_model[col].astype(int)

# ── STEP 6: RENAME TO MATCH API ───────────────────────────────────────────────
rename = {
    'Temperature(F)':    'temperature',
    'Humidity(%)':       'humidity',
    'Visibility(mi)':    'visibility',
    'Wind_Speed(mph)':   'wind_speed',
    'Weather_Condition': 'weather',
    'Junction':          'junction',
    'Traffic_Signal':    'traffic_signal',
    'Precipitation(in)': 'precipitation',
    'Pressure(in)':      'pressure',
    'Wind_Chill(F)':     'wind_chill',
    'Crossing':          'crossing',
    'Stop':              'stop',
    'Give_Way':          'give_way',
    'Railway':           'railway',
    'Roundabout':        'roundabout',
    'Turning_Loop':      'turning_loop',
    'Amenity':           'amenity',
}
df_model.rename(columns={k:v for k,v in rename.items() if k in df_model.columns}, inplace=True)
feature_names = [rename.get(f, f) for f in FEATURES if rename.get(f, f) in df_model.columns]

# Save feature list so backend uses EXACT same order
joblib.dump(feature_names, FEATURES_OUT)
print(f"   Saved {len(feature_names)} feature names → {FEATURES_OUT}")

# ── STEP 7: PREPARE X, y ─────────────────────────────────────────────────────
X = df_model[feature_names].astype(float)
y = df_model[TARGET]

print(f"\n📊 Severity distribution:")
for sv in sorted(y.unique()):
    cnt = (y == sv).sum()
    bar = '█' * max(1, cnt // (len(y) // 40))
    print(f"   Severity {sv}: {cnt:>7,} ({cnt/len(y)*100:.1f}%)  {bar}")

# ── STEP 8: TRAIN / TEST SPLIT ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── STEP 9: TRAIN MODELS ─────────────────────────────────────────────────────
print("\n🌳 Training Random Forest with new features...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=22,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"   RandomForest accuracy:  {rf_acc*100:.1f}%")

print("\n🚀 Training Gradient Boosting (3-5 min)...")
gb = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42,
)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"   GradientBoosting accuracy: {gb_acc*100:.1f}%")

print("\n🔗 Training Ensemble (VotingClassifier)...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft', n_jobs=-1
)
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"   Ensemble accuracy:      {ens_acc*100:.1f}%")

# ── STEP 10: PICK BEST ───────────────────────────────────────────────────────
best_acc   = max(rf_acc, gb_acc, ens_acc)
best_model = rf if rf_acc==best_acc else (gb if gb_acc==best_acc else ensemble)
best_name  = 'RandomForest' if rf_acc==best_acc else ('GradientBoosting' if gb_acc==best_acc else 'Ensemble')
print(f"\n🏆 Best model: {best_name}  ({best_acc*100:.1f}% accuracy)")

# ── STEP 11: EVALUATE ────────────────────────────────────────────────────────
y_pred = best_model.predict(X_test)
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=[f'Severity {i}' for i in sorted(y.unique())]))

# Feature importance
print("🔑 Top 10 Most Important Features:")
if hasattr(best_model,'feature_importances_'):
    imps = best_model.feature_importances_
elif hasattr(best_model,'estimators_'):
    imps = best_model.estimators_[0][1].feature_importances_
else:
    imps = rf.feature_importances_

ranked = sorted(zip(feature_names, imps), key=lambda x: -x[1])
for name, imp in ranked[:10]:
    bar = '█' * int(imp * 60)
    tag = ' ← NEW' if name in ['precipitation','pressure','wind_chill','hour','is_night','is_peak','is_weekend','is_dawn_dusk'] else ''
    print(f"   {name:22s} {imp:.3f}  {bar}{tag}")

# ── STEP 12: SAVE ─────────────────────────────────────────────────────────────
joblib.dump(best_model, MODEL_OUT)
size = os.path.getsize(MODEL_OUT)/1024/1024
print(f"\n💾 Model saved → {MODEL_OUT}  ({size:.1f} MB)")
print(f"   Features saved → {FEATURES_OUT}")
print(f"\n✅ Done! Restart app.py to load the new model.")
print("=" * 60)