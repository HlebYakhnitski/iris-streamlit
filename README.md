# iris-streamlit

A small end-to-end ML service: train several classifiers, track the runs, pick the best one, and serve it behind a web UI — packaged in Docker and shipped by CI.

The dataset is the classic Iris set. It is deliberately simple, because the point of the project is the pipeline around the model, not the model itself.

## What it does

**Training** (`train_model.py`)

- Compares four classifiers — Logistic Regression, Random Forest, SVC and k-Nearest Neighbors — each inside a `StandardScaler` pipeline.
- Logs every run to **MLflow** (`iris-model-zoo` experiment): parameters, metrics and artifacts.
- Scores each model on accuracy, precision, recall, F1 and ROC-AUC, and saves a confusion matrix per model.
- Exports the winning pipeline to `app/model.joblib`.

**Serving** (`app.py`)

- A **Streamlit** UI with sliders for the four flower measurements.
- Loads the trained pipeline and returns the predicted species with class probabilities.
- Resolves the model path across local, container and CI layouts, so the same code runs everywhere.

## Quick start

```bash
pip install -r requirements.txt
python train_model.py        # trains the models, writes app/model.joblib
streamlit run app.py         # opens http://localhost:8501
```

Training must run first — the app expects `app/model.joblib` to exist.

To inspect the experiment runs:

```bash
mlflow ui                    # reads ./mlruns, opens http://localhost:5000
```

## Docker

```bash
docker build -t iris-streamlit .
docker run -p 8501:8501 iris-streamlit
```

The image is `python:3.11-slim`, exposes `8501` and starts Streamlit bound to `0.0.0.0`.

## Tests

```bash
pytest
```

`tests/test_app.py` checks that `predict()` returns a valid class id for a known input. The test imports the app module directly, so the trained model has to be in place first.

## CI

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. Sets up Python 3.11 and installs dependencies
2. Trains the model, so the artifact exists in the runner
3. Runs `pytest`
4. Builds the Docker image and pushes it to GitHub Container Registry

That ordering is the interesting part: the model is a build artifact, not something committed to the repository, so CI reproduces it from source on every run.

## Layout

```
app.py                     Streamlit UI and prediction entry point
train_model.py             model comparison, MLflow logging, export
requirements.txt
Dockerfile                 python:3.11-slim, serves on 8501
tests/test_app.py
.github/workflows/ci.yml   test, build, push to GHCR
```

## Notes

This is a learning project built to practise the full path from training to a deployed, tested, containerised service. The model is trivial on purpose; the structure around it is the part worth reading.
