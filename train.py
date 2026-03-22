import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

os.makedirs("models", exist_ok=True)

X, y = load_iris(return_X_y=True)
target_names = load_iris().target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained. Accuracy: {accuracy:.2f}")
print(f"Classes: {target_names}")
print(f"Saved to models/model.pkl")

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
