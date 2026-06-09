import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn."
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

monthly_charges = st.slider(
    "Monthly Charges",
    0,
    150,
    70
)

total_charges = st.number_input(
    "Total Charges",
    value=1000
)

input_df = pd.DataFrame(
    0,
    index=[0],
    columns=feature_columns
)

input_df["tenure"] = tenure
input_df["MonthlyCharges"] = monthly_charges
input_df["TotalCharges"] = total_charges

if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error(
            "⚠ Customer is likely to churn"
        )
    else:
        st.success(
            "✅ Customer is likely to stay"
        )