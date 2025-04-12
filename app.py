import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Load Model & Assets ---
@st.cache_resource  # Cache to avoid reloading
def load_model_assets():
    try:
        model = joblib.load("model/predictor.pkl")
        scaler = joblib.load("model/scaler.pkl")
        feature_columns = joblib.load("model/features.pkl")
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, scaler, feature_columns = load_model_assets()

# --- Input Options ---
CROP_TYPES = ["Cotton", "Carrot", "Sugarcane", "Tomato", "Soybean", "Rice", "Maize", "Wheat", "Potato", "Barley"]
IRRIGATION_TYPES = ["Sprinkler", "Manual", "Flood", "Rain-fed", "Drip"]
SOIL_TYPES = ["Loamy", "Peaty", "Silty", "Clay", "Sandy"]
SEASONS = ["Kharif", "Zaid", "Rabi"]

# --- Streamlit UI ---
st.title("🌾 Agricultural Yield Predictor")
st.markdown("Predict crop yield based on farm conditions")

with st.form("input_form"):
    st.header("Farm Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        farm_area = st.number_input("Farm Area (acres)", min_value=0.0, value=100.0)
        fertilizer = st.number_input("Fertilizer Used (tons)", min_value=0.0, value=5.0)
    
    with col2:
        pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, value=2.0)
        water_usage = st.number_input("Water Usage (cubic meters)", min_value=0.0, value=50000.0)
    
    crop_type = st.selectbox("Crop Type", CROP_TYPES)
    irrigation_type = st.selectbox("Irrigation Type", IRRIGATION_TYPES)
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    season = st.selectbox("Season", SEASONS)
    
    submitted = st.form_submit_button("Predict Yield")

# --- Prediction Logic ---
if submitted:
    # Create input dictionary
    input_data = {
        "Farm_Area(acres)": farm_area,
        "Fertilizer_Used(tons)": fertilizer,
        "Pesticide_Used(kg)": pesticide,
        "Water_Usage(cubic meters)": water_usage,
    }

    # Add one-hot encoded features
    for crop in CROP_TYPES:
        input_data[f"Crop_Type_{crop}"] = 1 if crop == crop_type else 0
    for irrig in IRRIGATION_TYPES:
        input_data[f"Irrigation_Type_{irrig}"] = 1 if irrig == irrigation_type else 0
    for soil in SOIL_TYPES:
        input_data[f"Soil_Type_{soil}"] = 1 if soil == soil_type else 0
    for s in SEASONS:
        input_data[f"Season_{s}"] = 1 if s == season else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure correct column order (same as training)
    input_df = input_df[feature_columns]

    # Scale numerical features
    numerical_cols = [
        "Farm_Area(acres)",
        "Fertilizer_Used(tons)",
        "Pesticide_Used(kg)",
        "Water_Usage(cubic meters)"
    ]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    predicted_yield = model.predict(input_df)[0]

    # Display results
    st.success(f"**Predicted Yield:** {predicted_yield:.2f} tons")

    # Show feature importance
    st.subheader("🔍 Top Factors Affecting Yield")
    feature_importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(5)

    for _, row in feature_importance.iterrows():
        feature = row["Feature"]
        importance = row["Importance"]

        # Make feature names readable
        if feature.startswith("Crop_Type_"):
            readable = f"🌱 Crop: {feature.replace('Crop_Type_', '')}"
        elif feature.startswith("Irrigation_Type_"):
            readable = f"💧 Irrigation: {feature.replace('Irrigation_Type_', '')}"
        elif feature.startswith("Soil_Type_"):
            readable = f"🌍 Soil: {feature.replace('Soil_Type_', '')}"
        elif feature.startswith("Season_"):
            readable = f"⏳ Season: {feature.replace('Season_', '')}"
        else:
            readable = feature.replace("(", " (").replace("_", " ")

        st.write(f"- **{readable}** (Impact: {importance:.1%})")