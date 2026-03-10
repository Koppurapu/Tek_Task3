import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_insurance_model import add_engineered_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_insurance_model.joblib"
REPORT_PATH = BASE_DIR / "model_report.json"


def load_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    with REPORT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def clip_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def prepare_single_input(
    age: int,
    sex: str,
    bmi: float,
    children: int,
    smoker: str,
    region: str,
    report: dict,
) -> pd.DataFrame:
    row = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
    }

    df = pd.DataFrame([row])
    df = add_engineered_features(df)

    bounds = report.get("outlier_handling", {}).get("bounds", {})
    for feature in ["age", "bmi", "children", "smoker_bmi_interaction"]:
        if feature in bounds:
            low = float(bounds[feature]["lower"])
            high = float(bounds[feature]["upper"])
            df[feature] = df[feature].apply(lambda x: clip_value(float(x), low, high))

    return df


def show_metrics(report: dict) -> None:
    metrics = report.get("metrics", {})
    if not metrics:
        st.warning("No metrics found. Run train_insurance_model.py first.")
        return

    rows = []
    for model_name, vals in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "MAE": vals["MAE"],
                "RMSE": vals["RMSE"],
                "R2": vals["R2"],
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values(by="RMSE", ascending=True)
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Quick Visual Comparison")
    chart_df = metrics_df.set_index("Model")[["MAE", "RMSE", "R2"]]
    st.bar_chart(chart_df)


def show_outlier_info(report: dict) -> None:
    outlier = report.get("outlier_handling", {})
    bounds = outlier.get("bounds", {})

    if not bounds:
        st.info("Outlier bounds not found in report.")
        return

    rows = []
    for feature, vals in bounds.items():
        rows.append(
            {
                "Feature": feature,
                "Lower Bound": vals["lower"],
                "Upper Bound": vals["upper"],
            }
        )

    st.subheader("IQR Outlier Handling")
    st.caption("These are clipping bounds learned from the training data.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")
    st.title("Insurance Charges Prediction App")
    st.write("Predict insurance charges with interaction features and compare model performance.")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run train_insurance_model.py first.")
        return

    model = joblib.load(MODEL_PATH)
    report = load_report()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("User Inputs")
        age = st.slider("Age", min_value=18, max_value=64, value=30)
        sex = st.selectbox("Sex", options=["female", "male"])
        bmi = st.slider("BMI", min_value=10.0, max_value=55.0, value=27.5, step=0.1)
        children = st.slider("Children", min_value=0, max_value=5, value=0)
        smoker = st.selectbox("Smoker", options=["no", "yes"])
        region = st.selectbox(
            "Region",
            options=["northeast", "northwest", "southeast", "southwest"],
        )

        if st.button("Predict Charges", type="primary"):
            input_df = prepare_single_input(age, sex, bmi, children, smoker, region, report)
            pred = float(model.predict(input_df)[0])

            st.success(f"Estimated Insurance Charges: ${pred:,.2f}")

            st.markdown("### Generated Features")
            preview_cols = ["bmi_category", "age_group", "smoker_bmi_interaction"]
            st.dataframe(input_df[preview_cols], use_container_width=True)

    with col2:
        st.subheader("Model Evaluation")
        best_model = report.get("best_model", "Unknown")
        st.info(f"Best deployed model: {best_model}")
        show_metrics(report)
        show_outlier_info(report)


if __name__ == "__main__":
    main()
