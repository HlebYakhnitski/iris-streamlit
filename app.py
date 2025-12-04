import os
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

@st.cache_resource
def load_model():
    # Try common locations for the trained pipeline
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "model.joblib"),  # app/model.joblib
        os.path.join(os.getcwd(), "app", "model.joblib"),         # cwd/app/model.joblib
        os.path.join(os.getcwd(), "model.joblib"),                # cwd/model.joblib
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return load(p), p
    raise FileNotFoundError("model.joblib not found in: " + ", ".join(candidate_paths))

@st.cache_resource
def get_metadata():
    iris = load_iris()
    feature_names = iris.feature_names  # ['sepal length (cm)', ...]
    target_names = iris.target_names    # ['setosa', 'versicolor', 'virginica']
    X = iris.data
    means = X.mean(axis=0)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    return feature_names, target_names, means, mins, maxs

def main():
    st.title("🌸 Iris Species Predictor")
    st.write("Enter the four Iris flower measurements and click **Predict** to see the species.")

    feature_names, target_names, means, mins, maxs = get_metadata()

    # Sidebar: model info
    with st.sidebar:
        st.header("Model")
        try:
            model, model_path = load_model()
            st.success(f"Loaded: `{os.path.relpath(model_path, os.getcwd())}`")
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()
        st.caption("The model is a scikit-learn pipeline (StandardScaler + LogisticRegression).")

    # Inputs
    st.subheader("Measurements (in centimeters)")
    cols = st.columns(2)
    inputs = []
    for i, name in enumerate(feature_names):
        with cols[i % 2]:
            val = st.number_input(
                label=name.replace(" (cm)", ""),
                min_value=float(mins[i]),
                max_value=float(maxs[i]),
                value=float(np.round(means[i], 2)),
                step=0.1,
                format="%.2f",
                help=f"Range: {mins[i]:.1f}–{maxs[i]:.1f} cm",
            )
            inputs.append(val)

    if st.button("Predict", type="primary"):
        X = np.array(inputs).reshape(1, -1)
        pred_idx = int(model.predict(X)[0])
        proba = model.predict_proba(X).flatten()

        pred_label = target_names[pred_idx]
        st.success(f"Predicted species: **{pred_label.title()}**")  # Title-case for display

        # Show probabilities
        proba_df = pd.DataFrame({
            "species": [n.title() for n in target_names],
            "probability": proba
        }).set_index("species")

        st.subheader("Class probabilities")
        st.bar_chart(proba_df)

        with st.expander("Raw prediction output"):
            st.json({
                "input": dict(zip(feature_names, inputs)),
                "predicted_index": pred_idx,
                "predicted_label": pred_label,
                "proba": {name: float(p) for name, p in zip(target_names, proba)}
            })

    st.divider()
    st.caption("Tip: run `streamlit run app/app.py` and open http://localhost:8501 in your browser.")

def predict(features):
    """
    Функция для тестов и для простого использования из кода.

    features: iterable из 4 чисел [sepal length, sepal width, petal length, petal width]
    возвращает: int class_id (0, 1 или 2)
    """
    model, _ = load_model()
    X = np.array(features).reshape(1, -1)
    return int(model.predict(X)[0])


if __name__ == "__main__":
    main()
