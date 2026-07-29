import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Log Classifier", layout="wide")
st.title("Well Log Classifier")
st.markdown("Classify lithology and estimate porosity from well log data.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("lithology", "lithology_classifier.pkl"), ("porosity", "porosity_estimator.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
gamma_ray_api = st.sidebar.slider("Gamma Ray Api", 0, 200, 100)
resistivity_ohm_m = st.sidebar.slider("Resistivity Ohm M", 0, 100, 50)
neutron_porosity_pct = st.sidebar.slider("Neutron Porosity Pct", 0, 45, 22)
density_porosity_pct = st.sidebar.slider("Density Porosity Pct", 0, 45, 22)
sonic_us_ft = st.sidebar.slider("Sonic Us Ft", 40, 140, 90)
caliper_in = st.sidebar.slider("Caliper In", 6, 16, 11)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[gamma_ray_api, resistivity_ohm_m, neutron_porosity_pct, density_porosity_pct, sonic_us_ft, caliper_in]])
        m = models["lithology"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Lithology", result if isinstance(result, str) else f"{result:.4f}")
        m = models["porosity"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Porosity", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")

