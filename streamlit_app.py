import streamlit as st, joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path; import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Well Log Classifier", layout="wide")
st.title("Well Log Classifier")

class Engine:
    def __init__(self):
        p = Path(__file__).parent / 'outputs' / 'models'
        self.lithology = joblib.load(p / 'lithology_classifier.pkl')
        self.porosity = joblib.load(p / 'porosity_estimator.pkl')
    def run(self, name, X):
        m = getattr(self, name)
        if isinstance(m, dict):
            x = m['scaler'].transform(X)
            r = m['model'].predict(x)
            if 'label_encoder' in m:
                return m['label_encoder'].inverse_transform(r)[0]
            return float(r[0])
        return float(m.predict(X)[0])

eng = Engine()

with st.sidebar:
    st.header('Inputs')
    gamma = st.slider('Gamma', 0, 200, 100)
    resistivity = st.slider('Resistivity', 0, 100, 50)
    neutron = st.slider('Neutron', 0, 45, 22)
    density = st.slider('Density', 0, 45, 22)
    sonic = st.slider('Sonic', 40, 140, 90)
    caliper = st.slider('Caliper', 6, 16, 11)
    go = st.button('Predict', type='primary', use_container_width=True)

if go:
    x = np.array([[gamma, resistivity, neutron, density, sonic, caliper]])
    out = {}
    out['lithology'] = eng.run('lithology', x)
    out['porosity'] = eng.run('porosity', x)
    cols = st.columns(len(out))
    for i, (k, v) in enumerate(out.items()):
        cols[i].metric(k.title(), str(v) if isinstance(v, str) else f'{v:.2f}')
    nums = [v for v in out.values() if isinstance(v, (int, float))]
    if nums:
        fig, ax = plt.subplots(figsize=(6,2))
        names = [k.title() for k, v in out.items() if isinstance(v, (int, float))]
        colors = ['#2E86AB','#A23B72','#F18F01']
        bars = ax.bar(names, nums, color=colors[:len(names)])
        ax.axhline(y=sum(nums)/len(nums), color='gray', ls='--', alpha=0.5)
        for bar, val in zip(bars, nums):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*0.9, f'{val:.1f}', ha='center', va='top', color='white', fontweight='bold')
        st.pyplot(fig)