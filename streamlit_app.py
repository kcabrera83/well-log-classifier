import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Log Classifier", layout="wide")
st.title("Well Log Classifier")
st.markdown("Classify lithology and estimate porosity from well logs.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'lithology': joblib.load(d / 'lithology_classifier.pkl'), 'porosity': joblib.load(d / 'porosity_estimator.pkl')}

st.sidebar.header("Input Parameters")
gamma_ray = st.sidebar.slider('Gamma Ray', 0, 200, 100)
resistivity = st.sidebar.slider('Resistivity', 0, 100, 50)
neutron_porosity = st.sidebar.slider('Neutron Porosity', 0, 45, 22)
density_porosity = st.sidebar.slider('Density Porosity', 0, 45, 22)
sonic = st.sidebar.slider('Sonic', 40, 140, 90)
caliper = st.sidebar.slider('Caliper', 6, 16, 11)

if st.sidebar.button("Run"):
    try:
        x = np.array([[gamma_ray, resistivity, neutron_porosity, density_porosity, sonic, caliper]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))