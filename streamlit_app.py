import joblib
import streamlit as st
from sklearn.metrics import precision_score, classification_report
import pandas as pd

#Trained model
model = joblib.load("model.pkl")

feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "total sulfur dioxide", "density", "alcohol", "sugar_acid_ratio"]

st.title("White Wine Quality Prediction")
st.write("Predict whether a white wine is 'Good' (quality >= 7) or 'Average'")

st.sidebar.header("Wine Features")
input_data = {}
for col in feature_names:
    if col == "sugar_acid_ratio":
        continue
    input_data[col] = st.sidebar.number_input(col, min_value = 0.0, value = 1.0)

input_data['sugar_acid_ratio'] = input_data['residual sugar'] / (input_data['volatile acidity'] + 1e-6)

input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict"):
    proba = model.predict_proba(input_df)[0,1]
    prediction = (proba > 0.5).astype(int)
    st.subheader("Result")
    if prediction == 1:
        st.success(f"This White Wine is predicted to be 'Good' with probability {proba:.2f}")
    else:
        st.warning(f"This White Wine is predicted to be 'Average' with probability {proba:.2f}")

st.header("Evaluate Model on Dataset")

if st.button("Run Evaluation on Dataset"):
    data = pd.read_csv("winequality-white.csv", sep=";")

    data['sugar_acid_ratio'] = data['residual sugar'] / (data['volatile acidity'] + 1e-6)
    dup_count = data.duplicated().sum()
    st.write(f"Number of duplicate rows: {dup_count}")

    unique_data = data.drop_duplicates()
    X = unique_data[feature_names]
    y_true = (unique_data['quality'] >= 7).astype(int)
    y_pred = model.predict(X)

    prec = precision_score(y_true, y_pred)
    st.metric("Precision on dataset", f"{prec:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred))

