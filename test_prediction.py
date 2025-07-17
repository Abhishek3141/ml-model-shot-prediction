# test_prediction.py

from predictor.predict import predict_shot

# Example input: [LOCATION, W, FINAL_MARGIN, SHOT_NUMBER, PERIOD, DRIBBLES, SHOT_DIST, CLOSE_DEF_DIST, FGM, PTS]
sample_input = [1, 1, 5, 10, 2, 1, 15, 2.5, 0, 2]

prob = predict_shot(sample_input)
print(f"🔮 Shot make probability: {prob * 100:.2f}%")
