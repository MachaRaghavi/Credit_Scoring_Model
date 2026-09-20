import os
import joblib
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("Loading credit dataset...")

# Load German Credit dataset
data = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True
)

df = data.frame

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# Select financial features
features = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents"
]

X = df[features].copy()

# 1 = Good credit
# 0 = Higher credit risk
y = (df["class"] == "good").astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Machine Learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training model...")

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n----- MODEL PERFORMANCE -----")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
joblib.dump(model, "models/credit_model.pkl")

print("\nModel saved successfully!")
print("Location: models/credit_model.pkl")
