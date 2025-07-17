import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import numpy as np

st.set_page_config(layout="wide")
model = joblib.load("shot_model.pkl")

court_length = 50  # feet
court_width = 47
basket_x, basket_y = 25, 0

# Input: current player location
x = st.slider("Current X (feet from sideline)", 0.0, float(court_width), 25.0, 0.5)
y = st.slider("Current Y (feet from baseline)", 0.0, float(court_length), 20.0, 0.5)

# Context settings
def_dist = st.slider("Defender Distance (feet)", 0.0, 10.0, 3.0)
play_type = st.radio("Choose your play type:", ["Drive", "3-Pointer"])

# Search space params
radius_from_player = 5 if play_type == "Drive" else 10
min_shot_dist = 0 if play_type == "Drive" else 20
max_shot_dist = 5 if play_type == "Drive" else 30

# Grid search to find best location
best_prob = -1
best_coords = (x, y)

for dx in np.linspace(-radius_from_player, radius_from_player, 20):
    for dy in np.linspace(-radius_from_player, radius_from_player, 20):
        new_x = x + dx
        new_y = y + dy

        # Must be in court
        if not (0 <= new_x <= court_width and 0 <= new_y <= court_length):
            continue

        # Shot distance to hoop
        dist = ((new_x - basket_x) ** 2 + (new_y - basket_y) ** 2) ** 0.5

        # Skip if not in valid range
        if not (min_shot_dist <= dist <= max_shot_dist):
            continue

        X = np.array([[dist, def_dist]])
        prob = model.predict_proba(X)[0][1]

        if prob > best_prob:
            best_prob = prob
            best_coords = (new_x, new_y)

# Visualize
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, court_width)
ax.set_ylim(0, court_length)
ax.set_aspect('equal')
ax.add_patch(patches.Circle((basket_x, basket_y), 1.5, fill=False, color="orange", linewidth=2))
ax.plot(x, y, 'bo', markersize=10, label="You")
ax.plot(best_coords[0], best_coords[1], 'go', markersize=10, label="Best Move")
ax.plot([x, best_coords[0]], [y, best_coords[1]], 'k--', alpha=0.4)
ax.add_patch(patches.Circle((x, y), radius_from_player, fill=False, linestyle="--", color="gray"))
ax.legend()
ax.set_title("Court View")
ax.axis("off")
st.pyplot(fig)

# Output
st.subheader(f"📍 Recommended Move: ({round(best_coords[0],1)} ft, {round(best_coords[1],1)} ft)")
st.subheader(f"📈 New Shot Probability: **{round(best_prob * 100, 2)}%**")
