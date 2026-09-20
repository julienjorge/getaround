import time

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPERIMENT_NAME = "getaround-pricing"
SELECTED_MODEL = "random_forest_depth_10"

if __name__ == "__main__":
    print("training model...")
    start_time = time.time()

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()

    pricing = pd.read_csv("data/raw/get_around_pricing_project.csv", index_col=0)
    pricing = pricing[(pricing["mileage"] >= 0) & (pricing["mileage"] <= 500_000) & (pricing["engine_power"] > 0)]

    X = pricing.drop(columns="rental_price_per_day")
    y = pricing["rental_price_per_day"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical = ["model_key", "fuel", "paint_color", "car_type"]
    numeric = ["mileage", "engine_power"]
    boolean = [col for col in X.columns if col not in categorical + numeric]

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
        ("boolean", "passthrough", boolean),
    ])

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
        "random_forest_depth_10": RandomForestRegressor(max_depth=10, random_state=42),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model)])
            pipeline.fit(X_train, y_train)

            r2_test = r2_score(y_test, pipeline.predict(X_test))
            mae_test = mean_absolute_error(y_test, pipeline.predict(X_test))
            mlflow.log_metric("r2_test", r2_test)
            mlflow.log_metric("mae_test", mae_test)
            print(f"{name}: r2_test={r2_test:.3f} mae_test={mae_test:.2f}")

            if name == SELECTED_MODEL:
                joblib.dump(pipeline, "api/model.joblib")

    print("...Done!")
    print(f"---Total training time: {time.time() - start_time:.1f} s")
