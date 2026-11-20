import os
import pandas as pd
import pytest


@pytest.fixture
def raw_data_path():
    return "data/raw/iris.csv"


@pytest.fixture
def train_data_path():
    return "data/processed/train.csv"


@pytest.fixture
def test_data_path():
    return "data/processed/test.csv"


def test_raw_dataset_exists_and_shape(raw_data_path):
    assert os.path.exists(raw_data_path), f"File {raw_data_path} not found"
    df = pd.read_csv(raw_data_path)

    # The Iris dataset contains exactly 150 observations
    assert len(df) == 150, f"Expected 150 rows, got {len(df)}"

    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "target", "species"}
    assert expected_cols.issubset(set(df.columns)), f"Missing columns in {df.columns}"


def test_species_distribution_in_raw(raw_data_path):
    df = pd.read_csv(raw_data_path)
    counts = df["species"].value_counts().to_dict()

    # Problem statement requirement: 50 observations for each of the three species
    assert counts.get("Setosa") == 50, f"Expected 50 Setosa, got {counts.get('Setosa')}"
    assert counts.get("Versicolor") == 50, f"Expected 50 Versicolor, got {counts.get('Versicolor')}"
    assert counts.get("Virginica") == 50, f"Expected 50 Virginica, got {counts.get('Virginica')}"


def test_no_null_values(raw_data_path, train_data_path, test_data_path):
    for path in [raw_data_path, train_data_path, test_data_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            assert df.isnull().sum().sum() == 0, f"Found missing values in {path}"


def test_split_proportions(train_data_path, test_data_path):
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    assert len(train_df) == 120, f"Expected 120 train samples (80%), got {len(train_df)}"
    assert len(test_df) == 30, f"Expected 30 test samples (20%), got {len(test_df)}"
