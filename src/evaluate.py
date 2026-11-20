import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(params_path="params.yaml"):
    params = load_params(params_path)
    data_params = params["data"]
    train_params = params["train"]
    eval_params = params["evaluate"]

    test_path = data_params["test_path"]
    model_path = train_params["model_path"]
    metrics_path = eval_params["metrics_path"]

    print(f"Loading test data from {test_path}...")
    df_test = pd.read_csv(test_path)

    print(f"Loading model from {model_path}...")
    model_artifact = joblib.load(model_path)
    clf = model_artifact["model"]
    feature_cols = model_artifact["feature_names"]

    X_test = df_test[feature_cols]
    y_test = df_test["target"]

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    print("Evaluation Results:")
    print(json.dumps(metrics, indent=2))

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    mlflow.set_experiment("Iris_Flower_Classification")
    with mlflow.start_run(run_name="RandomForest_Evaluation"):
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1_score": f1,
        })


if __name__ == "__main__":
    evaluate_model()
