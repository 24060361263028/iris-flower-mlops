import os
import yaml
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(params_path="params.yaml"):
    params = load_params(params_path)
    data_params = params["data"]

    print("Loading Iris dataset from scikit-learn...")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Rename columns to clean snake_case
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    # Map target integer to species names: Setosa, Versicolor, Virginica
    target_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    df["species"] = df["target"].map(target_names)

    # Save raw dataset
    raw_path = data_params["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset ({len(df)} rows) to {raw_path}")

    # Train / test split (stratified by target to maintain 50/50/50 balance)
    train_df, test_df = train_test_split(
        df,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
        stratify=df["target"],
    )

    # Save processed splits
    train_path = data_params["train_path"]
    test_path = data_params["test_path"]
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train set ({len(train_df)} rows) to {train_path}")
    print(f"Saved test set ({len(test_df)} rows) to {test_path}")


if __name__ == "__main__":
    prepare_data()
