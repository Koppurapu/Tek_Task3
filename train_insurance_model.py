import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bmi_bins = [0, 18.5, 25, 30, np.inf]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["bmi_category"] = pd.cut(df["bmi"], bins=bmi_bins, labels=bmi_labels, right=False)

    age_bins = [0, 30, 45, 60, np.inf]
    age_labels = ["YoungAdult", "Adult", "MiddleAged", "Senior"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)

    smoker_flag = (df["smoker"] == "yes").astype(int)
    df["smoker_bmi_interaction"] = smoker_flag * df["bmi"]

    return df


def iqr_clip_outliers(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    clipped_df = df.copy()
    bounds = {}

    for col in columns:
        q1 = clipped_df[col].quantile(0.25)
        q3 = clipped_df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            # Skip clipping when spread is zero to preserve informative values.
            bounds[col] = {
                "lower": float(clipped_df[col].min()),
                "upper": float(clipped_df[col].max()),
            }
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped_df[col] = clipped_df[col].clip(lower=lower, upper=upper)
        bounds[col] = {"lower": float(lower), "upper": float(upper)}

    return clipped_df, bounds


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(x_test)
    mse = mean_squared_error(y_test, preds)
    return {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, preds)),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "insurance.csv"

    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    # IQR outlier handling by clipping numeric feature tails.
    numeric_cols_for_outliers = ["age", "bmi", "children", "smoker_bmi_interaction", "charges"]
    df, iqr_bounds = iqr_clip_outliers(df, numeric_cols_for_outliers)

    y = df["charges"]
    x = df.drop(columns=["charges"])

    numeric_features = ["age", "bmi", "children", "smoker_bmi_interaction"]
    categorical_features = ["sex", "smoker", "region", "bmi_category", "age_group"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    fitted_pipelines = {}
    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        fitted_pipelines[name] = pipeline
        results[name] = evaluate_model(pipeline, x_test, y_test)

    best_model_name = min(results, key=lambda m: results[m]["RMSE"])
    best_model = fitted_pipelines[best_model_name]

    model_path = base_dir / "final_insurance_model.joblib"
    report_path = base_dir / "model_report.json"

    joblib.dump(best_model, model_path)

    report = {
        "dataset": str(data_path.name),
        "outlier_handling": {
            "method": "IQR clipping",
            "columns": numeric_cols_for_outliers,
            "bounds": iqr_bounds,
        },
        "metrics": results,
        "best_model": best_model_name,
        "artifacts": {
            "model": str(model_path.name),
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Model comparison (lower MAE/RMSE is better, higher R2 is better):")
    for model_name, metric in results.items():
        print(
            f"- {model_name}: MAE={metric['MAE']:.2f}, RMSE={metric['RMSE']:.2f}, R2={metric['R2']:.4f}"
        )

    print(f"\nSelected final model: {best_model_name}")
    print(f"Saved model artifact: {model_path}")
    print(f"Saved metrics report: {report_path}")


if __name__ == "__main__":
    main()
