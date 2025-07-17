import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("shot_logs.csv")

# Drop rows with missing values in relevant columns
df = df[["SHOT_DIST", "CLOSE_DEF_DIST", "FGM"]].dropna()

# Features and target
X = df[["SHOT_DIST", "CLOSE_DEF_DIST"]]
y = df["FGM"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Train
pipe.fit(X_train, y_train)

# Save
joblib.dump(pipe, "shot_model.pkl")
print("✅ Model trained and saved as shot_model.pkl")
