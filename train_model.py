# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# # 1. Load the dataset
# import csv

# with open("data/insurance_claims.csv", "r", encoding="ISO-8859-1", errors="replace") as f:
#     sample = f.read(2048)
#     f.seek(0)
#     dialect = csv.Sniffer().sniff(sample)
#     df = pd.read_csv(f, delimiter=dialect.delimiter, on_bad_lines='skip')

df = pd.read_csv("data/insurance_claims.csv", encoding="ISO-8859-1", on_bad_lines='skip')
print("📌 Available columns:", df.columns.tolist())



# 2. Drop unnecessary columns
df = df.drop(['policy_number', 'policy_bind_date', 'incident_location',
              'incident_date', 'insured_zip', 'auto_make', 'auto_model', 'auto_year'], axis=1, errors='ignore')

# 3. Drop rows with missing target label
print("📌 Available columns:", df.columns.tolist())
df = df.dropna(subset=['fraud_reported'])

# 4. Convert target column to binary
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

# 5. Encode all categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# 6. Separate features and target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# 7. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Save model to file
os.makedirs('model', exist_ok=True)
with open('model/fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 10. Save encoders
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Model and encoders saved successfully!")
