import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "student-mat.csv")
MODEL_PATH = os.path.join("models", "student_model.pkl")
FEATURE_IMPORTANCE_PLOT = os.path.join("models", "feature_importance.png")

def load_data():
    df = pd.read_csv(DATA_PATH, sep=';')
    return df

def build_preprocessing_and_model(df):
    # Target: final grade G3
    selected_features = ["G1", "G2", "absences", "studytime"]
    X = df[selected_features]
    y = df["G3"]


    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
        ("num", numeric_transformer, numeric_features)
        ]
    )


    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return X, y, pipeline, numeric_features, categorical_features

def train_and_evaluate():
    df = load_data()
    X, y, pipeline, numeric_features, categorical_features = build_preprocessing_and_model(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Evaluation metrics on test set:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")

    # Feature importance
    model = pipeline.named_steps["model"]

    feature_names = numeric_features
    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.barh(range(len(sorted_importances)), sorted_importances[::-1], align='center')
    plt.yticks(range(len(sorted_importances)), sorted_feature_names[::-1])
    plt.xlabel("Relative Importance")
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PLOT)
    plt.close()

    joblib.dump(
        {
            "pipeline": pipeline,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_names": feature_names,
        },
        MODEL_PATH
    )
    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PLOT}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train_and_evaluate()
