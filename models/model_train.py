# train_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train_and_save_model():
    df = pd.read_csv("diabetes.csv")
    df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(model, "gb_diabetes_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    train_and_save_model()

