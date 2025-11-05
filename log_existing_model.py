"""
Log an existing Random Tree model (random_tree.pkl) to MLflow
using the Pima Indians Diabetes dataset for evaluation.

Run with:
    python log_existing_model.py
Then start MLflow UI:
    mlflow ui
"""

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Load your pre-trained model ---
model_path = "modele_random_forest_diabete.pkl"
print(f"📦 Loading model from: {model_path}")
model = joblib.load(model_path)

# --- Load dataset (Pima Indians Diabetes from OpenML) ---
print("📊 Loading diabetes dataset...")
diabetes = fetch_openml(name="diabetes", version=1, as_frame=True)
X = diabetes.data
y = diabetes.target

# --- Ensure feature names match the model ---
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    # Keep only matching columns
    X = X[model_features]
    print(f"✅ Using model feature names: {model_features}")
else:
    print("⚠️ Model does not store feature names; using all dataset columns.")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Evaluate model ---
print("🔍 Evaluating model...")
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Accuracy: {acc:.4f}")

# --- Log to MLflow ---
print("🧾 Logging to MLflow...")
with mlflow.start_run(run_name="loaded_random_tree"):
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    print("🎉 Model logged successfully to MLflow")

print("\nTo view MLflow UI, run:\n  mlflow ui\nand open http://localhost:5000 in your browser.")
