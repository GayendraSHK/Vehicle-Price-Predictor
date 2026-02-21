import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# FEATURE ENCODING & TRAIN/TEST SPLIT

print("=" * 60)
print("FEATURE ENCODING & TRAIN/TEST SPLIT")
print("=" * 60)

# Load preprocessed data
filepath = input("Enter preprocessed CSV file path: ").strip()

df = pd.read_csv(filepath, encoding='utf-8-sig')
print(f"\nLoaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# Define target and features
TARGET = 'price'
CATEGORICAL_COLS = ['make', 'model', 'gear', 'fuel_type', 'options', 'location']
NUMERIC_COLS = ['yom', 'mileage', 'engine_cc']

print(f"\nTarget: {TARGET}")
print(f"Categorical features: {CATEGORICAL_COLS}")
print(f"Numeric features: {NUMERIC_COLS}")

# Label Encode categorical features
print(f"\n Label Encoding ")
label_encoders = {}

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes encoded")

# Save label encoders for later use (in frontend)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"\nSaved label encoders to: label_encoders.pkl")

# Separate features and target
X = df[CATEGORICAL_COLS + NUMERIC_COLS]
y = df[TARGET]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature dtypes:")
print(X.dtypes)

# Train/Test Split (80/20)
print(f"\n Train/Test Split ")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify class distribution
print(f"\n Target Distribution ")
print(f"  Train mean price: Rs {y_train.mean():,.0f}")
print(f"  Test mean price:  Rs {y_test.mean():,.0f}")
print(f"  Train median:     Rs {y_train.median():,.0f}")
print(f"  Test median:      Rs {y_test.median():,.0f}")

# Save to CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False, header=['price'])
y_test.to_csv('y_test.csv', index=False, header=['price'])

print(f"\nSaved files:")
print(f"  X_train.csv: {X_train.shape}")
print(f"  X_test.csv:  {X_test.shape}")
print(f"  y_train.csv: {y_train.shape}")
print(f"  y_test.csv:  {y_test.shape}")

# Summary
print(f"\n{'=' * 60}")
print(f"READY FOR XGBOOST TRAINING")
print(f"{'=' * 60}")
print(f"Run: python xgboost_vehicle_model.py")
