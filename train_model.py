from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)

print("\n=== Hold-out evaluation ===")
print("accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("log_loss:", round(log_loss(y_test, y_proba), 4))
print("\nclassification_report:\n", classification_report(y_test, y_pred, digits=4))
print("confusion_matrix:\n", confusion_matrix(y_test, y_pred))

print("\n=== Stratified 5-fold CV (accuracy) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=None)
print("folds:", np.round(cv_scores, 4))
print("mean:", round(cv_scores.mean(), 4), "std:", round(cv_scores.std(), 4))

print("\n=== Multiclass ROC-AUC (OvR) ===")
print("roc_auc_ovr:", round(roc_auc_score(y_test, y_proba, multi_class="ovr"), 4))

Path("app").mkdir(exist_ok=True)
from joblib import dump
dump(pipe, "app/model.joblib")
