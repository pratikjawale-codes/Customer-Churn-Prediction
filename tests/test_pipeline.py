import importlib

import pytest

from data.generate_data import generate_telecom_churn_data
from src.data_processing import ChurnDataPreprocessor

app_module = importlib.import_module("src.app")


@pytest.fixture
def sample_data():
    return generate_telecom_churn_data(n_samples=200, seed=42)


def test_data_generation_is_reproducible_and_has_both_classes(sample_data):
    assert len(sample_data) == 200
    assert "customerID" in sample_data.columns
    assert set(sample_data["Churn"]) == {"Yes", "No"}
    assert sample_data.equals(generate_telecom_churn_data(n_samples=200, seed=42))


def test_preprocessor_fits_and_preserves_rows(sample_data):
    train_data = sample_data.iloc[:150]
    test_data = sample_data.iloc[150:].drop(columns=["Churn"])

    preprocessor = ChurnDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_data)
    X_test = preprocessor.transform(test_data)
    duplicated = preprocessor.transform(test_data.iloc[[0, 0]])

    assert X_train.shape[0] == len(train_data)
    assert len(y_train) == len(train_data)
    assert X_test.shape == (len(test_data), X_train.shape[1])
    assert duplicated.shape[0] == 2
    assert len(preprocessor.feature_names) > 10


def test_web_page_and_uninitialized_health(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "preprocessor", None)
    monkeypatch.setattr(app_module, "model_path", str(tmp_path / "missing-model.pkl"))
    monkeypatch.setattr(app_module, "preprocessor_path", str(tmp_path / "missing-preprocessor.pkl"))

    client = app_module.app.test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"Customer profile" in page.data

    health = client.get("/health")
    assert health.get_json()["status"] == "uninitialized"


def test_flask_prediction_flow(sample_data, tmp_path, monkeypatch):
    from src.train import train_and_evaluate_models

    data_path = tmp_path / "sample.csv"
    artifacts_dir = tmp_path / "artifacts"
    sample_data.to_csv(data_path, index=False)

    monkeypatch.setattr(app_module, "model", None)
    monkeypatch.setattr(app_module, "preprocessor", None)
    monkeypatch.setattr(app_module, "artifacts_dir", str(artifacts_dir))
    monkeypatch.setattr(app_module, "model_path", str(artifacts_dir / "best_model.pkl"))
    monkeypatch.setattr(app_module, "preprocessor_path", str(artifacts_dir / "preprocessor.pkl"))

    best_model, results = train_and_evaluate_models(str(data_path), str(artifacts_dir))
    app_module.load_artifacts()
    client = app_module.app.test_client()

    health = client.get("/health")
    assert health.get_json()["status"] == "healthy"
    assert results["test_metrics"]["roc_auc"] >= 0
    assert best_model in results

    records = sample_data.iloc[:2].to_dict(orient="records")
    prediction = client.post("/predict", json=records)
    assert prediction.status_code == 200
    payload = prediction.get_json()
    assert payload["status"] == "success"
    assert payload["predictions_count"] == 2
    assert "churn_prediction" in payload["predictions"][0]
    assert "churn_probability" in payload["predictions"][0]
    assert "recommended_action" in payload["predictions"][0]

    invalid = client.post("/predict", json={"tenure": 6})
    assert invalid.status_code == 400
    assert "missing_fields" in invalid.get_json()
