import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import numpy as np
import lightgbm as lgb
import scipy.sparse

st.set_page_config(layout="wide")

# Load model and scaler
model = lgb.Booster(model_file="shot_model_lgbm.txt")
scaler = joblib.load("scaler.pkl")

court_length = 50
court_width = 47

st.title("🏀 Shot Predictor (Simplified)")
st.write("Drag the player and adjust defender distance to see shot make probability.")

# Court sliders
x = st.slider("Player X Position (feet from left sideline)", 0.0, float(court_width), 25.0, 0.5)
y = st.slider("Player Y Position (feet from baseline)", 0.0, float(court_length), 20.0, 0.5)
def_dist = st.slider("Defender Distance (feet)", 0.0, 15.0, 3.0)

# Compute shot distance to hoop
shot_dist = round(((x - 25)**2 + y**2)**0.5, 2)

# Draw court
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, court_width)
ax.set_ylim(0, court_length)
ax.set_aspect('equal')
ax.add_patch(patches.Circle((25, 0), 1.5, fill=False, color="orange"))
ax.plot(x, y, 'bo', markersize=12, label="Player")
ax.add_patch(patches.Circle((x, y), def_dist, fill=False, color="red", linestyle="--", label="Defender Range"))
ax.legend(loc="upper right")
ax.axis("off")
st.pyplot(fig)

# Predict
X = np.array([[shot_dist, def_dist]])
X_scaled = scaler.transform(X)
prob = model.predict(X_scaled)
# Ensure prob is a flat numpy array, even if sparse
try:
    import scipy.sparse
    if scipy.sparse.issparse(prob):
        prob = scipy.sparse.csr_matrix(prob).toarray().flatten()
    else:
        prob = np.array(prob).flatten()
except Exception:
    prob = np.array(prob).flatten()
st.subheader(f"📈 Predicted Make Probability: **{round(prob[0] * 100, 2)}%**")
