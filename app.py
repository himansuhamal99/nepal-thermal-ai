import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration & Performance Optimization
st.set_page_config(page_title="Nepal Thermal AI", page_icon="🇳🇵", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('nepal_thermal_model.pkl')

model = load_model()

# HEADER 
st.title("Nepal Thermal Anomaly Predictor")
st.markdown("### Agricultural Decision Support System | Bagmati Province")
st.write("---")

# INPUT SECTION
with st.sidebar:
    st.header("📍 Observation Site")
    zone = st.selectbox("Ecological Zone", [0, 1, 2], 
                        format_func=lambda x: ["Terai (Lowland)", "Hills", "Mountains"][x])
    st.info("The model adjusts thresholds based on altitude-specific climate norms.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Temperature Data")
    t_mean = st.number_input("Yesterday's Mean Temp (°C)", value=22.0, help="Avg of Max/Min from last 24h.")
    # Explain why we use Yesterday's Temp
    st.caption("Primary Predictor: Captures **Thermal Inertia**.")

with col2:
    st.subheader("🌬️ Atmospheric State")
    hum = st.number_input("Current Humidity (%)", value=55.0)
    wind_kmh = st.number_input("Wind Speed (km/h)", value=12.0)
    pres = st.number_input("Surface Pressure (hPa)", value=1013.0)

# INFERENCE ENGINE & ADVISORY LOGIC
if st.button("RUN RISK ASSESSMENT", use_container_width=True):
    # Standardizing Units
    wind_ms = wind_kmh / 3.6
    features = ['Latitude', 'Longitude', 'Temp_2m', 'RH_2m', 'Pressure', 'WindSpeed_10m', 'DTR', 'Eco_Zone', 'Temp_Lag_1']
    data = pd.DataFrame([[27.7, 85.3, t_mean, hum, pres, wind_ms, 8.5, zone, t_mean]], columns=features)
    
    prob = model.predict_proba(data)[0][1]
    
    # RESULTS
    st.write("---")
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Risk Probability", f"{prob:.2%}")
        if prob > 0.5:
            st.error("🚨 HIGH RISK ALERT")
        else:
            st.success("✅ STABLE CONDITIONS")

    with res_col2:
        st.subheader("📝 Deployment Advisory")
        if prob > 0.5:
            st.warning("**Immediate Actions Required:**")
            st.markdown("""
            * **Agriculture:** Initiate early morning irrigation to combat soil moisture loss.
            * **Livestock:** Ensure adequate shade and ventilation in poultry and cattle sheds.
            * **Public Health:** Issue community alerts regarding peak heat hours (12:00 PM - 4:00 PM).
            * **Monitoring:** Increase frequency of atmospheric pressure checks.
            """)
        else:
            st.info("**Routine Operations:**")
            st.markdown("""
            * Continue standard seasonal crop cycles.
            * Maintain normal irrigation schedules.
            * No significant thermal anomaly predicted for the next 24-48 hours.
            """)

# --- TECHNICAL FOOTER ---
st.write("---")
with st.expander("🔬 View Model Architecture "):
    st.write("""
    - **Model:** Random Forest Ensemble (100+ Trees)
    - **F1-Score:** 0.889 | **Recall:** 0.861
    - **Feature Engineering:** Integrated 24h temporal lag and SI unit standardization.
    - **Data Source:** NASA POWER Daily Atmospheric Archive (Bagmati Region).
    """)
