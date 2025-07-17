# predictor/predict.py

import joblib
import numpy as np

# Load the model
model = joblib.load("shot_model.pkl")

def predict_shot(features: list):
    """
    Predict shot make probability based on input features.
    
    Args:
        features (list): A list of feature values in the same order as training:
                         [LOCATION, W, FINAL_MARGIN, SHOT_NUMBER, PERIOD, DRIBBLES,
                          SHOT_DIST, CLOSE_DEF_DIST, FGM, PTS]

    Returns:
        probability of shot being made (float between 0 and 1)
    """
    proba = model.predict_proba([features])[0][1]
    return proba
