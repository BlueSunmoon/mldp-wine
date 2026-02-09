import joblib
import streamlit as st
import numpy as np
import pandas as pd

#Trained model
model = joblib.load("model.pkl")

feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "total sulfur dioxide", "density", "alcohol", "sugar_acid_ratio"]

st.title("White Wine Quality Prediction")
st.write("Predict whether a white wine is 'Good' (quality >= 7) or 'Average'")

st.sidebar.header("Wine Features")
input_data = {}
for col in feature_names:
    input_data[col] = st.sidebar.number_input(col, min_value = 0.0, value = 1.0)

input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict"):
    proba = model.predict_proba(input_df)[0,1]
    prediction = (proba > 0.5).astype(int)
    st.subheader("Result")
    if prediction == 1:
        st.success(f"This White Wine is predicted to be 'Good' with probability {proba:.2f}")
    else:
        st.warning(f"This White Wine is predicted to be 'Average' with probability {proba:.2f}")



