# model.py
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

class DiabetesModel:
    def __init__(self):
        # Load Pima Indians dataset
        df = pd.read_csv("diabetes.csv")  
        self.feature_names = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        X = df[self.feature_names]
        y = df["Outcome"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        # Save model
        joblib.dump(self.model, "models/gb_diabetes_model.pkl")

    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
