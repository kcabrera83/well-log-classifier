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
gamma_ray = st.sidebar.slider('gamma ray', 0, 200, 100)
resistivity = st.sidebar.slider('resistivity', 0, 100, 50)
neutron_porosity = st.sidebar.slider('neutron porosity', 0, 45, 22)
density_porosity = st.sidebar.slider('density porosity', 0, 45, 22)
sonic = st.sidebar.slider('sonic', 40, 140, 90)
caliper = st.sidebar.slider('caliper', 6, 16, 11)

if st.sidebar.button("Run"):
    try:
        x = np.array([[gamma_ray, resistivity, neutron_porosity, density_porosity, sonic, caliper]])
        cols = st.columns(2)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                cols[i].metric(k.title(), m['label_encoder'].inverse_transform(p)[0])
            else:
                cols[i].metric(k.title(), f'{p[0]:.2f}')
    except Exception as e:
        st.error(f'Error: {e}')
