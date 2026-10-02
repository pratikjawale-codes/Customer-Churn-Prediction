# Telecom Customer Churn Prediction

A demonstration project that generates synthetic telecom customer data, trains and compares churn classifiers, and serves predictions through a Flask API and a pearl-inspired web interface.

> **Data note:** The included generator creates synthetic records and synthetic churn labels. This is not a production-validated model, and the example retention actions and predictions are not evidence of real-world accuracy, revenue, or churn reduction.

## Project flow

1. Generate reproducible synthetic customer records.
2. Split into training, validation, and untouched test sets before fitting preprocessing.
3. Fit preprocessing on training data, tune selected models with cross-validation, and choose the model by validation ROC-AUC.
4. Refit the selected model and preprocessor on training plus validation data; report final metrics once on the test set.
5. Save model, preprocessor, and metrics under `artifacts/`, then serve `/` and `/predict`.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
# .venv\Scripts\activate.bat
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python src/app.py
```

Open `http://localhost:5000` for the prediction page. The pipeline creates the dataset and model artifacts before the page can make predictions.

### API

- `GET /health` — model readiness.
- `POST /predict` — one customer object or a non-empty list of customer objects. Supply the customer/service/billing fields shown in the web form. Returns churn class, estimated probability, risk band, and a suggested action.

Run the test suite with:

```bash
python -m pytest
```

The Jenkins pipeline runs tests and builds a Docker image; it does not push or deploy the image to a cloud target.

## Docker

The image trains the synthetic-data pipeline once during the build; containers then start the API directly on port 5000:

```bash
docker build -t telecom-churn-api .
docker run --rm -p 5000:5000 telecom-churn-api
```

## Main files

- `data/generate_data.py` — synthetic data generation.
- `src/data_processing.py` — feature engineering and train-fitted preprocessing.
- `src/train.py` — model selection, final holdout evaluation, and artifact export.
- `src/app.py` — Flask web page, health check, and prediction API.
- `src/templates/` and `src/static/` — responsive pearl-themed interface.
- `tests/test_pipeline.py` — data, preprocessing, and API tests.
