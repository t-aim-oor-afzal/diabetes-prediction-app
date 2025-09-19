import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient information below to predict diabetes outcome.")

# Input form
#preg = st.number_input("Pregnancies", 0, 20, 1)
#glucose = st.number_input("Glucose", 0, 200, 120)
#bp = st.number_input("Blood Pressure", 0, 150, 70)
#skin = st.number_input("Skin Thickness", 0, 100, 20)
#insulin = st.number_input("Insulin", 0, 900, 80)
#bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
#dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
#age = st.number_input("Age", 1, 120, 30)

preg = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 0, 200, 120)
bp = st.slider("Blood Pressure", 0, 150, 70)
skin = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 80)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 1, 120, 30)


# Collect input
features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High risk of Diabetes (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of Diabetes (Probability: {prob:.2f})")