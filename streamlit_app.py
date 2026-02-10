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

st.header("Batch Predict")
st.write("Upload CSV file with white wine features to get predictions")

template_data = {col: [0.0] for col in feature_names}
template_df = pd.DataFrame(template_data)

st.download_button(
    label = "Download CSV Template",
    data = template_df.to_csv(index=False).encode("utf-8"),
    file_name = "white_wine_features_template.csv",
    mime = "text/csv"
)

uploaded_csv = st.file_uploader("Choose CSV file", type="csv")
if uploaded_csv is not None:
    input_df = pd.read_csv(uploaded_csv)
    st.write("Preview of uploaded file")
    st.dataframe(input_df.head())

    if "sugar_acid_ratio" not in input_df.columns:
        if "residual sugar" in input_df.columns and "volatile acidity" in input_df.columns:
            input_df["sugar_acid_ratio"] = input_df["residual sugar"] / (input_df["volatile acidity"] + 1e-6)
        else:
            st.error("CSV must contain 'residual sugar' and 'volatile acidity' columns to compute sugar_acid_ratio.") 

    missing_columns = [col for col in feature_names if col not in input_df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
    else:
        # Reorder columns to match training columns
        input_df = input_df[feature_names]

        probas = model.predict_proba(input_df)

        input_df["Probability Good"] = probas[:, 1]
        input_df["Probability Average"] = probas[:, 0]

        threshold = 0.3
        preds = (input_df["Probability Good"] > threshold).astype(int)

        input_df["Predicted Quality"] = preds
        

        st.write("Predicted Probabilities & Qualities:")
        st.dataframe(input_df)

        csv_output = input_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label = "Download predictions as CSV",
            data = csv_output,
            file_name = "predicted_white_wine_quality.csv",
            mime = "text/csv"
        )


    