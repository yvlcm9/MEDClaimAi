import streamlit as st
import pickle as pkl
import pandas as pd

# ===========================================
# Page Configuration
# ===========================================

st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="🏥",
    layout="centered"
)

# ===========================================
# Load Trained Model
# ===========================================

try:
    model = pkl.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("model.pkl not found. Please place model.pkl in the project folder.")
    st.stop()

# ===========================================
# Title
# ===========================================

st.title("🏥 Insurance Premium Prediction System")

st.write(
    "Predict medical insurance charges using Machine Learning."
)

st.divider()

# ===========================================
# User Inputs
# ===========================================

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Gender",
    ("male", "female")
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ("yes", "no")
)

region = st.selectbox(
    "Region",
    (
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    )
)

st.divider()

# ===========================================
# Prediction
# ===========================================

if st.button("Predict Insurance Charges"):

    data = [[
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    ]]

    columns = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region"
    ]

    df = pd.DataFrame(data, columns=columns)

    prediction = model.predict(df)

    st.success(
        f"Predicted Insurance Charges: ₹ {prediction[0]:,.2f}"
    )

    st.subheader("Input Summary")

    st.dataframe(df)