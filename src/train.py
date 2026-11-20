import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def train_model(params_path="params.yaml"):
    params = load_params(params_path)
    train_params = params["train"]
    data_params = params["data"]

    train_path = data_params["train_path"]
    model_path = train_params["model_path"]

    print(f"Reading training data from {train_path}...")
    df_train = pd.read_csv(train_path)

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_train = df_train[feature_cols]
    y_train = df_train["target"]

    n_estimators = train_params["n_estimators"]
    max_depth = train_params["max_depth"]
    random_state = train_params["random_state"]

    mlflow.set_experiment("Iris_Flower_Classification")

    with mlflow.start_run(run_name="RandomForest_Train"):
        print("Training Random Forest Classifier...")
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)

        # Training predictions & metric
        y_train_pred = clf.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        print(f"Train Accuracy: {train_acc:.4f}")

        # MLflow logging
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("train_accuracy", train_acc)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Store metadata along with model
        model_artifact = {
            "model": clf,
            "feature_names": feature_cols,
            "target_names": ["Setosa", "Versicolor", "Virginica"],
            "classes": [0, 1, 2],
        }
        joblib.dump(model_artifact, model_path)
        print(f"Model saved to {model_path}")

        mlflow.log_artifact(model_path, artifact_path="model_artifact")
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            serialization_format="cloudpickle",
        )


if __name__ == "__main__":
    train_model()
