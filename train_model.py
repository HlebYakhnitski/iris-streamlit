import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

import joblib
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("file:./mlruns")

# =====================
# MLflow experiment
# =====================
mlflow.set_experiment("iris-model-zoo")


# =====================
# Data loading
# =====================
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# Model Zoo
# =====================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            probability=True,
            random_state=42
        ))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])
}

# =====================
# Tracking best model
# =====================
best_f1 = -1.0
best_model = None
best_model_name = None
best_run_id = None
best_metrics = {}

if __name__ == "__main__":
    print("MLflow training script started")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # -----------------
            # Metadata
            # -----------------
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("version", "v1.0.0")

            # -----------------
            # Train
            # -----------------
            model.fit(X_train, y_train)

            # -----------------
            # Predict
            # -----------------
            y_pred = model.predict(X_test)

            try:
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            except Exception:
                y_proba = None
                roc_auc = None

            # -----------------
            # Metrics
            # -----------------
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_metric("recall_macro", recall)

            if roc_auc is not None:
                mlflow.log_metric("roc_auc_ovr", roc_auc)

            # -----------------
            # Artifacts
            # -----------------
            report = classification_report(y_test, y_pred)
            report_path = f"classification_report_{model_name}.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(4, 4))
            plt.imshow(cm)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.colorbar()
            plt.tight_layout()

            cm_path = f"confusion_matrix_{model_name}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # -----------------
            # Log model
            # -----------------
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id

            # -----------------
            # Best model selection
            # -----------------
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name
                best_run_id = run_id
                best_metrics = {
                    "accuracy": round(accuracy, 4),
                    "f1_macro": round(f1, 4),
                }
    # =====================
    # Save best model locally
    # =====================
    Path("app").mkdir(exist_ok=True)

    joblib.dump(best_model, "app/model.joblib")

    meta = {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "mlflow_run_id": best_run_id,
        "version": "v1.0.0"
    }

    with open("app/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Best model: {best_model_name} | F1-macro: {best_metrics['f1_macro']}")

    # =====================
    # Register model in MLflow Model Registry
    # =====================
    model_uri = f"runs:/{best_run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name="IrisModel"
    )


