import os
import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def model_path():
    return "models/model.joblib"


@pytest.fixture
def test_data_path():
    return "data/processed/test.csv"


def test_model_artifact_exists(model_path):
    assert os.path.exists(model_path), f"Trained model not found at {model_path}"


def test_model_artifact_structure(model_path):
    artifact = joblib.load(model_path)
    assert isinstance(artifact, dict), "Model artifact should be a dictionary"
    assert "model" in artifact, "Model key missing from artifact"
    assert "feature_names" in artifact, "feature_names missing from artifact"
    assert "target_names" in artifact, "target_names missing from artifact"
    assert artifact["target_names"] == ["Setosa", "Versicolor", "Virginica"]


def test_model_accuracy_threshold(model_path, test_data_path):
    artifact = joblib.load(model_path)
    clf = artifact["model"]
    feature_cols = artifact["feature_names"]

    df_test = pd.read_csv(test_data_path)
    X_test = df_test[feature_cols]
    y_test = df_test["target"]

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Random forest on Iris should comfortably exceed 90% accuracy
    assert acc >= 0.90, f"Model accuracy {acc:.4f} is below the 0.90 threshold"


def test_known_setosa_prediction(model_path):
    artifact = joblib.load(model_path)
    clf = artifact["model"]

    # Canonical Setosa sample (small petals)
    sample = pd.DataFrame([{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }])

    pred = clf.predict(sample)[0]
    assert pred == 0, f"Expected class 0 (Setosa), got {pred}"
    assert artifact["target_names"][pred] == "Setosa"
