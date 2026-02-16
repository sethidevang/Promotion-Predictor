"""
Train promotion prediction model for the Flask app (from hrdata.ipynb).
Requires WA_Fn-UseC_-HR-Employee-Attrition.csv in this folder.
Run once, then start the app.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    print("Loading data...")
    df = pd.read_csv(csv_path)

    df["Promoted"] = (df["YearsSinceLastPromotion"] == 0).astype(int)
    X = df.drop(columns=["Promoted", "YearsSinceLastPromotion", "EmployeeNumber"])
    y = df["Promoted"]
    feature_columns = list(X.columns)

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("Training...")
    pipeline.fit(X_train, y_train)

    pipe_path = os.path.join(base_dir, "pipeline.pkl")
    cols_path = os.path.join(base_dir, "feature_columns.pkl")
    joblib.dump(pipeline, pipe_path)
    joblib.dump(feature_columns, cols_path)
    print(f"Pipeline saved to {pipe_path}")
    print(f"Feature columns ({len(feature_columns)}) saved to {cols_path}")


if __name__ == "__main__":
    main()
