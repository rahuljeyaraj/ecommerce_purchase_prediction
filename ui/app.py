import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("E-commerce Purchase Prediction")
st.write("Enter session behavior features to predict purchase likelihood.")

# Inputs
cart = st.number_input("Cart count", min_value=0, value=1, step=1)
view = st.number_input("View count", min_value=0, value=5, step=1)
session_duration_sec = st.number_input("Session duration (seconds)", min_value=0.0, value=120.0, step=1.0)
unique_products = st.number_input("Unique products viewed", min_value=0, value=3, step=1)
avg_price = st.number_input("Average price", min_value=0.0, value=500.0, step=1.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "cart": cart,
        "view": view,
        "session_duration_sec": session_duration_sec,
        "unique_products": unique_products,
        "avg_price": avg_price
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.write(f"**Purchase Prediction:** {int(prediction)}")
    st.write(f"**Purchase Probability:** {probability:.4f}")

    if prediction == 1:
        st.success("This session is likely to result in a purchase.")
    else:
        st.warning("This session is less likely to result in a purchase.")