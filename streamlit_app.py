import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    low  = joblib.load("model_low.pkl")
    med  = joblib.load("model_med.pkl")
    high = joblib.load("model_high.pkl")
    return low, med, high

model_low, model_med, model_high = load_models()

st.title("🥕 Capsicum Modal Price Forecast")
st.write("Enter your feature values in the sidebar to get price forecasts.")

st.sidebar.header("Input Features")
def user_input_features():
    data = {
        "Year":        st.sidebar.slider("Year", 1990, 2030, 2025),
        "Month":       st.sidebar.slider("Month", 1, 12, 9),
        "Arrivals":    st.sidebar.number_input("Arrivals (tonnes)", 0.0),
        "tempmax":     st.sidebar.number_input("Max Temperature (°F)", 0.0),
        "tempmin":     st.sidebar.number_input("Min Temperature (°F)", 0.0),
        "humidity":    st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0),
        "precip":      st.sidebar.number_input("Precip (inches)", 0.0),
        "solarenergy": st.sidebar.number_input("Solar Energy (kWh/m²)", 0.0),
        "uvindex":     st.sidebar.number_input("UV Index", 0.0),
    }
    return pd.DataFrame([data])

input_df = user_input_features()

low_pred  = model_low.predict(input_df)[0]
med_pred  = model_med.predict(input_df)[0]
high_pred = model_high.predict(input_df)[0]

st.subheader("Forecasted Price (Rs./Quintal)")
st.metric("10th Percentile", f"{low_pred:,.2f}")
st.metric("Median",          f"{med_pred:,.2f}")
st.metric("90th Percentile", f"{high_pred:,.2f}")
