# 🛡️ Insurance Claim Fraud Detector

An intelligent fraud detection web app built with machine learning and Streamlit.  
It predicts whether an insurance claim is likely to be **fraudulent or legitimate**, based on real-world data patterns.

---

## 🔍 Problem Statement

Insurance fraud is a growing concern that causes billions in losses worldwide.  
This project aims to build a **lightweight, deployable fraud detection tool** that helps insurance analysts:

- Assess incoming claims quickly
- Flag high-risk (potentially fraudulent) claims
- Improve claim validation efficiency

---

## 🚀 Features

✅ Simple & intuitive web interface (built with Streamlit)  
✅ Works on individual claims (manual entry)  
✅ Optionally localized for Indian context (cities, states, relationships)  
✅ Confidence score with each prediction  
✅ Built on real-world data using scikit-learn  
✅ Fully deployable via Streamlit Cloud  

---

## 🛠️ Tech Stack

| Component     | Technology |
|---------------|------------|
| Frontend UI   | Streamlit  |
| ML Framework  | Scikit-learn |
| Data Handling | Pandas     |
| Model Storage | Pickle     |
| Deployment    | Streamlit Cloud |

---

## 📁 Project Structure
insurance-claim-fraud-detector/
├── app.py # Streamlit frontend app
├── train_model.py # Script to train and save model
├── requirements.txt # Python dependencies
├── models/
│ ├── fraud_model.pkl
│ └── label_encoders.pkl
├── data/
│ └── insurance_claims.csv
└── README.md # This file
