import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join("models", "student_model.pkl")

@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["pipeline"], data["numeric_features"], data["categorical_features"]

def generate_advice(predicted_score, input_data):
    score = float(predicted_score)

    advice_list = []

    if score < 10:
        advice_list.append(
            "Your predicted final score is low. Focus on understanding core concepts and seek extra help from teachers or friends."
        )
    elif score < 14:
        advice_list.append(
            "Your predicted score is average. Regular practice and revising previous tests can significantly improve your performance."
        )
    else:
        advice_list.append(
            "Your predicted score is good. Keep your current study habits and try solving more challenging problems to aim for excellence."
        )

    if "studytime" in input_data:
        study_time = input_data["studytime"]
        if study_time <= 2:
            advice_list.append(
                "Increase your weekly study time. Try scheduling at least 1 extra hour of focused study per week."
            )

    if "failures" in input_data:
        failures = input_data["failures"]
        if failures >= 1:
            advice_list.append(
                "You have had past failures. Review what went wrong and discuss your strategy with a teacher or mentor."
            )

    if "absences" in input_data:
        absences = input_data["absences"]
        if absences > 5:
            advice_list.append(
                "High absences can reduce your performance. Try to attend classes more regularly."
            )

    if not advice_list:
        advice_list.append("Keep a consistent study schedule and maintain a healthy balance between school and rest.")

    return advice_list

def main():
    st.title("Student Performance Improvement Predictor")
    st.write("Predict final exam score (G3) and get simple improvement suggestions.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please run train_model.py first.")
        return

    pipeline, numeric_features, categorical_features = load_model()

    st.sidebar.header("Input Features")

    input_data = {}

    # Minimal, but representative subset of features for UI
    # You can extend these with more if needed
    input_data["age"] = st.sidebar.slider("Age", min_value=15, max_value=22, value=17)
    input_data["absences"] = st.sidebar.slider("Number of absences", 0, 30, 2)
    input_data["G1"] = st.sidebar.slider("First period grade (G1)", 0, 20, 10)
    input_data["G2"] = st.sidebar.slider("Second period grade (G2)", 0, 20, 10)
    input_data["studytime"] = st.sidebar.slider("Weekly study time (1=low, 4=high)", 1, 4, 2)
    input_data["failures"] = st.sidebar.slider("Past class failures (0-4)", 0, 4, 0)

    # Some common categorical features
    input_data["sex"] = st.sidebar.selectbox("Sex", ["F", "M"])
    input_data["schoolsup"] = st.sidebar.selectbox("Extra educational support (schoolsup)", ["yes", "no"])
    input_data["famsup"] = st.sidebar.selectbox("Family educational support (famsup)", ["yes", "no"])
    input_data["internet"] = st.sidebar.selectbox("Internet access at home", ["yes", "no"])

    if st.button("Predict Final Score"):
        df_input = pd.DataFrame([input_data])

        predicted_score = pipeline.predict(df_input)[0]

        st.subheader("Predicted Final Score (G3)")
        st.write(f"{predicted_score:.2f} / 20")

        st.subheader("Improvement Suggestions")
        advice_list = generate_advice(predicted_score, input_data)
        for a in advice_list:
            st.write("- " + a)

if __name__ == "__main__":
    main()


