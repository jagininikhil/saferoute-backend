"""
evaluate_and_improve.py — SafeRoute AI FIXED
═════════════════════════════════════════════
Fix: loads model_features.pkl to match exact training columns.
     Adds missing columns filled with 0 instead of crashing.

Run:  python evaluate_and_improve_fixed.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib, os, warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))

FILE_PATH    = os.path.join(BASE, 'US_Accidents_March23.csv')
MODEL_PATH   = os.path.join(BASE, 'accident_model.pkl')
FEATURES_PKL = os.path.join(BASE, 'model_features.pkl')
ENCODER_PKL  = os.path.join(BASE, 'weather_encoder.pkl')
SAMPLE_ROWS  = 300_000

print("=" * 65)
print("  SafeRoute AI — Model Evaluation & Improvement (Fixed)")
print("=" * 65)

# ── LOAD DATASET ──────────────────────────────────────────────────────────────
print(f"\n📂 Loading {SAMPLE_ROWS} rows...")
df = pd.read_csv(FILE_PATH, nrows=SAMPLE_ROWS, low_memory=False)
print(f"   Loaded {len(df):,} rows")

# ── TIME FEATURES ─────────────────────────────────────────────────────────────
if 'Start_Time' in df.columns:
    df['Start_Time']  = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['hour']        = df['Start_Time'].dt.hour.fillna(12).astype(int)
    df['day_of_week'] = df['Start_Time'].dt.dayofweek.fillna(0).astype(int)
    df['is_night']    = ((df['hour']>=22)|(df['hour']<=5)).astype(int)
    df['is_peak']     = (((df['hour'].between(7,9))|(df['hour'].between(17,20)))&(df['day_of_week']<5)).astype(int)
    df['is_weekend']  = (df['day_of_week']>=5).astype(int)
    df['is_dawn_dusk']= (df['hour'].between(5,7)|df['hour'].between(18,20)).astype(int)
else:
    for c in ['hour','is_night','is_peak','is_weekend','is_dawn_dusk']:
        df[c]=0

# ── LOAD MODEL + SAVED FEATURE LIST ──────────────────────────────────────────
print("\n🔍 Loading model and feature list...")
existing_model = None
saved_features = None

try:
    existing_model = joblib.load(MODEL_PATH)
    print(f"   ✅ Model: {type(existing_model).__name__}")
except Exception as e:
    print(f"   ⚠  Model load failed: {e}")

try:
    saved_features = joblib.load(FEATURES_PKL)
    print(f"   ✅ Saved features ({len(saved_features)}): {saved_features}")
except:
    saved_features = ['temperature','humidity','visibility','wind_speed',
                      'weather','junction','traffic_signal']
    print(f"   ⚠  model_features.pkl not found — using 7 base features")

# ── RENAME + CLEAN ────────────────────────────────────────────────────────────
rename = {
    'Temperature(F)':'temperature','Humidity(%)':'humidity',
    'Visibility(mi)':'visibility','Wind_Speed(mph)':'wind_speed',
    'Weather_Condition':'weather','Junction':'junction',
    'Traffic_Signal':'traffic_signal','Precipitation(in)':'precipitation',
    'Pressure(in)':'pressure','Wind_Chill(F)':'wind_chill',
    'Crossing':'crossing','Stop':'stop','Give_Way':'give_way',
    'Railway':'railway','Roundabout':'roundabout',
    'Turning_Loop':'turning_loop','Amenity':'amenity',
}

avail = [c for c in list(rename.keys())+['hour','is_night','is_peak','is_weekend','is_dawn_dusk'] if c in df.columns]
df2   = df[avail+['Severity']].copy()
df2.dropna(subset=['Severity'], inplace=True)

for col in df2.select_dtypes(include=[np.number]).columns:
    df2[col].fillna(df2[col].median(), inplace=True)
for col in df2.select_dtypes(include=['bool','object']).columns:
    if col!='Weather_Condition': df2[col].fillna(False, inplace=True)
df2.dropna(subset=['Weather_Condition'], inplace=True)

# Encode weather
le = LabelEncoder()
try:
    le = joblib.load(ENCODER_PKL)
    known = set(le.classes_)
    df2['Weather_Condition'] = le.transform(
        df2['Weather_Condition'].astype(str).map(lambda x: x if x in known else le.classes_[0]))
    print("   ✅ Encoder loaded")
except:
    df2['Weather_Condition'] = le.fit_transform(df2['Weather_Condition'].astype(str))
    print("   ℹ  Encoder rebuilt")

for col in df2.select_dtypes(include=['bool']).columns:
    df2[col] = df2[col].astype(int)

df2.rename(columns={k:v for k,v in rename.items() if k in df2.columns}, inplace=True)

# ── KEY FIX: add any columns the model expects but are missing ────────────────
print(f"\n🔧 Aligning to saved feature list ({len(saved_features)} features)...")
for feat in saved_features:
    if feat not in df2.columns:
        df2[feat] = 0
        print(f"   ➕ Added missing '{feat}' = 0")

TARGET = 'Severity'
X = df2[saved_features].astype(float)
y = df2[TARGET]

print(f"\n📊 Class distribution:")
for sv in sorted(y.unique()):
    cnt=(y==sv).sum()
    bar='█'*max(1,cnt//(len(y)//40))
    print(f"   Severity {sv}: {cnt:>7,} ({cnt/len(y)*100:.1f}%)  {bar}")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f"\n✂  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── STEP 1: EVALUATE EXISTING ─────────────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 1 — Existing model performance")
print("─"*65)

old_acc = 0
if existing_model:
    y_pred  = existing_model.predict(X_test)
    old_acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {old_acc*100:.1f}%\n")
    print(classification_report(y_test, y_pred,
          target_names=[f'Severity {i}' for i in sorted(y.unique())]))
    # Feature importance
    imps = None
    if hasattr(existing_model,'feature_importances_'):
        imps = existing_model.feature_importances_
    elif hasattr(existing_model,'estimators_'):
        try: imps = existing_model.estimators_[0][1].feature_importances_
        except: pass
    if imps is not None:
        print("🔑 Feature Importance (top 10):")
        for name,imp in sorted(zip(saved_features,imps),key=lambda x:-x[1])[:10]:
            print(f"   {name:22s} {imp:.3f}  {'█'*int(imp*60)}")

# ── STEP 2: RETRAIN WITH ALL FEATURES ────────────────────────────────────────
print("\n" + "─"*65)
print("STEP 2 — Retraining with all available features")
print("─"*65)

# Use every available feature
all_feats = list(dict.fromkeys(
    [rename.get(c,c) for c in avail if rename.get(c,c) in df2.columns] +
    [c for c in ['hour','is_night','is_peak','is_weekend','is_dawn_dusk'] if c in df2.columns]
))
print(f"   {len(all_feats)} features: {all_feats}")

X2       = df2[all_feats].astype(float)
X2_train = X2.loc[X_train.index]
X2_test  = X2.loc[X_test.index]

print("\n🌳 RandomForest...")
rf = RandomForestClassifier(n_estimators=200,max_depth=22,min_samples_leaf=4,
     max_features='sqrt',class_weight='balanced',n_jobs=-1,random_state=42)
rf.fit(X2_train,y_train)
rf_acc = accuracy_score(y_test,rf.predict(X2_test))
print(f"   {rf_acc*100:.1f}%  (was {old_acc*100:.1f}%,  +{(rf_acc-old_acc)*100:.1f}%)")

print("\n🚀 GradientBoosting (3-5 min)...")
gb = GradientBoostingClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,
     subsample=0.8,min_samples_leaf=10,random_state=42)
gb.fit(X2_train,y_train)
gb_acc = accuracy_score(y_test,gb.predict(X2_test))
print(f"   {gb_acc*100:.1f}%")

print("\n🔗 Ensemble...")
ens = VotingClassifier(estimators=[('rf',rf),('gb',gb)],voting='soft',n_jobs=-1)
ens.fit(X2_train,y_train)
ens_acc = accuracy_score(y_test,ens.predict(X2_test))
print(f"   {ens_acc*100:.1f}%")

best_acc   = max(rf_acc,gb_acc,ens_acc)
best_model = rf if rf_acc==best_acc else (gb if gb_acc==best_acc else ens)
best_name  = 'RandomForest' if rf_acc==best_acc else ('GradientBoosting' if gb_acc==best_acc else 'Ensemble')
print(f"\n🏆 Best: {best_name}  {best_acc*100:.1f}%  (+{(best_acc-old_acc)*100:.1f}% improvement)")
print(f"\n{classification_report(y_test,best_model.predict(X2_test),target_names=[f'Severity {i}' for i in sorted(y.unique())])}")

print("🔑 New feature importance (top 12):")
ni = best_model.feature_importances_ if hasattr(best_model,'feature_importances_') else best_model.estimators_[0][1].feature_importances_
for name,imp in sorted(zip(all_feats,ni),key=lambda x:-x[1])[:12]:
    tag=' ← NEW' if name in ['precipitation','pressure','wind_chill','hour','is_night','is_peak','is_weekend','is_dawn_dusk'] else ''
    print(f"   {name:22s} {imp:.3f}  {'█'*int(imp*60)}{tag}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
print("\n" + "─"*65)
joblib.dump(best_model, MODEL_PATH)
joblib.dump(all_feats,  FEATURES_PKL)
if not os.path.exists(ENCODER_PKL):
    joblib.dump(le, ENCODER_PKL)
size=os.path.getsize(MODEL_PATH)/1024/1024
print(f"💾 Saved: accident_model.pkl ({size:.1f}MB)  |  model_features.pkl ({len(all_feats)} features)")
print(f"\n✅ Done! Run: python app.py")
print("=" * 65)