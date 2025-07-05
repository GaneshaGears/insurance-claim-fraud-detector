import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("model/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.set_page_config(page_title="Insurance Claim Fraud Detector", layout="centered")

st.title("🚨 Insurance Claim Fraud Detector")
st.markdown("Enter claim details below to check if it's **fraudulent** or **legitimate**.")

# ----------- Categorical Inputs -------------
user_input = {}

col1, col2 = st.columns(2)

st.subheader("📝 Claim Details (Localized for India)")

# Mappings: India-friendly display → model-compatible values
state_map = {
    "Karnataka (KA)": "OH",
    "Maharashtra (MH)": "IL",
    "Delhi (DL)": "IN"
}

city_map = {
    "Bengaluru": "Columbus",
    "Mumbai": "Arlington",
    "New Delhi": "Riverwood"
}

relationship_map = {
    "Self": "husband",
    "Child": "own-child",
    "Partner (Unmarried)": "unmarried",
    "Spouse": "wife"
}

hobby_map = {
    "Reading": "reading",
    "Chess": "chess",
    "CrossFit/Gym": "cross-fit",
    "Movies": "movies",
    "Trekking": "kayaking"
}

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    policy_state_display = st.selectbox("Policy Issued State", list(state_map.keys()))
    user_input['policy_state'] = state_map[policy_state_display]

    user_input['insured_sex'] = st.selectbox("Gender", ['MALE', 'FEMALE'])

    user_input['insured_education_level'] = st.selectbox(
        "Education Level",
        ['High School', 'College', 'JD', 'MD', 'PhD']
    )

    user_input['insured_occupation'] = st.selectbox(
        "Occupation",
        ['craft-repair', 'machine-op-inspct', 'sales', 'tech-support', 'exec-managerial']
    )

    user_input['incident_type'] = st.selectbox(
        "Type of Incident",
        ['Single Vehicle Collision', 'Vehicle Theft', 'Multi-vehicle Collision', 'Parked Car']
    )

    user_input['collision_type'] = st.selectbox(
        "Collision Type",
        ['Rear Collision', 'Side Collision', 'Front Collision']
    )

    user_input['incident_severity'] = st.selectbox(
        "Damage Severity",
        ['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage']
    )

with col2:
    relationship_display = st.selectbox("Relationship to Policyholder", list(relationship_map.keys()))
    user_input['insured_relationship'] = relationship_map[relationship_display]

    hobby_display = st.selectbox("Hobby", list(hobby_map.keys()))
    user_input['insured_hobbies'] = hobby_map[hobby_display]

    user_input['authorities_contacted'] = st.selectbox(
        "Authorities Contacted",
        ['Police', 'Fire', 'Other', 'None']
    )

    incident_city_display = st.selectbox("Incident City", list(city_map.keys()))
    user_input['incident_city'] = city_map[incident_city_display]

    # For simplicity, incident_state = policy_state
    user_input['incident_state'] = user_input['policy_state']

    user_input['property_damage'] = st.radio("Was Property Damaged?", ['YES', 'NO'])

    user_input['police_report_available'] = st.radio("Police Report Available?", ['YES', 'NO'])



# ----------- Numeric Inputs -------------
st.subheader("Numeric Claim Details")

numeric_fields = {
    'months_as_customer': (0, 300),
    'age': (18, 100),
    'policy_deductable': (0, 3000),
    'policy_annual_premium': (0, 150000),
    'umbrella_limit': (-1000000, 1000000),
    'capital-gains': (0, 100000),
    'capital-loss': (0, 100000),
    'incident_hour_of_the_day': (0, 23),
    'number_of_vehicles_involved': (1, 5),
    'bodily_injuries': (0, 5),
    'witnesses': (0, 10),
    'total_claim_amount': (0, 100000),
    'injury_claim': (0, 50000),
    'property_claim': (0, 50000),
    'vehicle_claim': (0, 50000),
}

# Use ranges for input
for col in numeric_fields:
    min_val, max_val = numeric_fields.get(col, (0, 10000))
    user_input[col] = st.number_input(
        col.replace("_", " ").title(),
        min_value=min_val,
        max_value=max_val,
        value=min_val
    )

#Predict
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    try:
        # 🔐 Encode categorical fields first
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])
            else:
                # If the categorical column is missing, use first class from encoder
                input_df[col] = le.transform([le.classes_[0]])

        # 🧠 Ensure all expected features are present
        expected_cols = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(encoders.keys()) + list(numeric_fields.keys())
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0  # Add default for missing numeric fields

        # 🧩 Arrange columns in model order
        input_df = input_df[expected_cols]

        # 🔮 Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][prediction]

        # ✅ Display result
        result_text = "✅ Legitimate Claim" if prediction == 0 else "🚨 Fraudulent Claim"
        st.subheader("Prediction Result:")
        st.success(f"{result_text} (Confidence: {probability:.2%})")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.write("🔍 Debug Input:", input_df)



import io

st.markdown("---")
st.header("🧾 Sample CSV for Testing")

# Create sample CSV in memory
sample_data = {
    'policy_state': ['OH'],
    'policy_csl': ['250/500'],
    'insured_sex': ['MALE'],
    'insured_education_level': ['High School'],
    'insured_occupation': ['craft-repair'],
    'insured_hobbies': ['reading'],
    'insured_relationship': ['husband'],
    'incident_type': ['Single Vehicle Collision'],
    'collision_type': ['Rear Collision'],
    'incident_severity': ['Major Damage'],
    'authorities_contacted': ['Police'],
    'incident_state': ['NY'],
    'incident_city': ['Columbus'],
    'property_damage': ['YES'],
    'police_report_available': ['YES'],
    'months_as_customer': [45],
    'age': [34],
    'policy_deductable': [500],
    'policy_annual_premium': [1200.50],
    'umbrella_limit': [0],
    'capital-gains': [0],
    'capital-loss': [0],
    'incident_hour_of_the_day': [13],
    'number_of_vehicles_involved': [1],
    'bodily_injuries': [0],
    'witnesses': [2],
    'total_claim_amount': [10000],
    'injury_claim': [5000],
    'property_claim': [3000],
    'vehicle_claim': [2000]
}

df_sample = pd.DataFrame(sample_data)
csv_bytes = df_sample.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Sample CSV",
    data=csv_bytes,
    file_name="sample_insurance_claim.csv",
    mime="text/csv"
)
