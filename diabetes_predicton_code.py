import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import streamlit as st
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/gopiashokan/dataset/main/diabetes_prediction_dataset.csv")

# Preprocessing using Ordinal Encoder
enc = OrdinalEncoder()
df["smoking_history"] = enc.fit_transform(df[["smoking_history"]])
df["gender"] = enc.fit_transform(df[["gender"]])

# Define Independent and Dependent Variables
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Scale numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# Streamlit app
st.set_page_config(page_title="Diabetes Prediction", page_icon=":dna:")
st.markdown("<h1 style='text-align: center;'>Diabetes Prediction</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.text_input("Age")
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        ["Never", "Current", "Former", "Ever", "Not Current", "No Info"],
    )
    bmi = st.text_input("BMI")
    hba1c_level = st.text_input("HbA1c Level")
    blood_glucose_level = st.text_input("Blood Glucose Level")

st.write("")
st.write("")
submit = st.button("Submit")
st.write("")

# Gender and Smoking History Mapping
gender_dict = {"Female": 0.0, "Male": 1.0, "Other": 2.0}
hypertension_dict = {"No": 0, "Yes": 1}
heart_disease_dict = {"No": 0, "Yes": 1}
smoking_history_dict = {
    "Never": 4.0,
    "No Info": 0.0,
    "Current": 1.0,
    "Former": 3.0,
    "Ever": 2.0,
    "Not Current": 5.0,
}

if submit:
    try:
        # Input Validation
        age = float(age) if age else None
        bmi = float(bmi) if bmi else None
        hba1c_level = float(hba1c_level) if hba1c_level else None
        blood_glucose_level = float(blood_glucose_level) if blood_glucose_level else None

        if None in [age, bmi, hba1c_level, blood_glucose_level]:
            st.warning("Please fill in all required information with valid values.")
        else:
            user_data = np.array(
                [
                    [
                        gender_dict[gender],
                        age,
                        hypertension_dict[hypertension],
                        heart_disease_dict[heart_disease],
                        smoking_history_dict[smoking_history],
                        bmi,
                        hba1c_level,
                        blood_glucose_level,
                    ]
                ]
            )
            user_data_scaled = scaler.transform(user_data)
            test_result = model.predict(user_data_scaled)

            if test_result[0] == 0:
                st.success("Diabetes Result: Negative")
                st.balloons()
            else:
                st.error("Diabetes Result: Positive (Please Consult with a Doctor)")

    except ValueError:
        st.warning("Please enter numeric values for Age, BMI, HbA1c Level, and Blood Glucose Level.")