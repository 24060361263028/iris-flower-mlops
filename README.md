# Iris Flower Classification using MLOps

An end-to-end, production-ready Machine Learning Operations (MLOps) pipeline for classifying Iris flower species using Random Forest and modern engineering best practices.

---

## 📌 Problem Statement

- **Objective**: Build a machine learning system that identifies the species of an Iris flower using its sepal and petal measurements.
- **Input**: Four numeric measurements in centimeters:
  - `sepal_length` (cm)
  - `sepal_width` (cm)
  - `petal_length` (cm)
  - `petal_width` (cm)
- **Output**: One of three species:
  - **Setosa**
  - **Versicolor**
  - **Virginica**
- **Dataset**: Iris dataset with 150 observations (50 for each of the three species) from `scikit-learn`.
- **ML Algorithm**: Random Forest Classifier.

---

## 🛠️ MLOps Stack & Tools

| Tool | Purpose |
|---|---|
| **Git** | Source code, configuration, and pipeline definition version control. |
| **DVC (Data Version Control)** | Data pipeline orchestration, artifact tracking, dependency hashing, and metrics comparison. |
| **scikit-learn** | Built-in dataset extraction, model training (Random Forest), and evaluation. |
| **MLflow** | Experiment tracking, hyperparameter logging, model artifact management, and metric visualization. |
| **pytest** | Automated test suite verifying data integrity, model performance thresholds, and REST API contracts. |
| **FastAPI** | High-performance asynchronous REST API for single and batch predictions with Swagger UI documentation. |
| **Docker & Compose** | Containerization for environment parity and zero-dependency production deployment. |
| **GitHub Actions** | Automated Continuous Integration & Continuous Deployment (CI/CD) workflow. |

---

## 📁 Repository Structure

```text
d:\mloops\
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD pipeline
├── app/
│   ├── __init__.py
│   └── main.py                # FastAPI inference application
├── data/
│   ├── raw/                   # Raw iris dataset (150 rows, 50 per class)
│   │   └── iris.csv
│   └── processed/             # Stratified train (80%) and test (20%) splits
│       ├── train.csv
│       └── test.csv
├── metrics/
│   └── metrics.json           # Accuracy, precision, recall, F1-score, confusion matrix
├── models/
│   └── model.joblib           # Serialized Random Forest model artifact
├── src/
│   ├── __init__.py
│   ├── prepare.py             # Data extraction and train/test splitting
│   ├── train.py               # Model training and MLflow tracking
│   └── evaluate.py            # Evaluation metrics calculation and logging
├── tests/
│   ├── __init__.py
│   ├── test_data.py           # Tests for data integrity and class distributions
│   ├── test_model.py          # Tests for accuracy threshold (>90%) and predictions
│   └── test_api.py            # Tests for FastAPI endpoints and input validation
├── .dockerignore
├── .gitignore
├── Dockerfile                 # Production container image definition
├── docker-compose.yml         # Container deployment configuration
├── dvc.lock                   # DVC stage hashes and pipeline state
├── dvc.yaml                   # DVC pipeline specification
├── params.yaml                # Centralized hyperparameters & data paths
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies if setting up on a new machine:
```bash
pip install -r requirements.txt
```

---

### 2. DVC Pipeline Execution & Data Versioning

Execute the full reproducible pipeline (`prepare` ➔ `train` ➔ `evaluate`):
```bash
dvc repro
```

Inspect model evaluation metrics:
```bash
dvc metrics show
```

Output:
```text
Path                  accuracy    f1_score    precision    recall
metrics\metrics.json  0.9667      0.9666      0.9697       0.9667
```

---

### 3. MLflow Experiment Tracking

Start the local MLflow dashboard to visualize runs, metrics, and model artifacts:
```bash
mlflow ui
```
Open your browser at `http://127.0.0.1:5000` to inspect the `Iris_Flower_Classification` experiment.

---

### 4. Running the Test Suite (pytest)

Run all unit and integration tests:
```bash
pytest -v
```

This validates:
1. `test_data.py`:
   - Raw Iris dataset contains 150 rows.
   - Exactly 50 observations for each class (Setosa, Versicolor, Virginica).
   - Zero missing or null values.
   - Train (120 rows) and test (30 rows) split proportions.
2. `test_model.py`:
   - Model artifact exists and contains metadata.
   - Test accuracy exceeds the 90% production threshold (achieved ~96.7%).
   - Known samples classify accurately.
3. `test_api.py`:
   - Health check and info endpoints.
   - Single prediction schema and confidence score.
   - Batch prediction processing.
   - Pydantic validation on invalid inputs (e.g. negative values).

---

### 5. Serving the REST API (FastAPI)

Launch the FastAPI application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation (Swagger UI):
- Open `http://127.0.0.1:8000/docs`

#### Sample Prediction Request (curl):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

#### Sample Response:
```json
{
  "species": "Setosa",
  "class_id": 0,
  "probabilities": {
    "Setosa": 1.0,
    "Versicolor": 0.0,
    "Virginica": 0.0
  },
  "confidence": 1.0,
  "input_features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
}
```

---

### 6. Docker Containerization

#### Build the image:
```bash
docker build -t iris-mlops-api:latest .
```

#### Run the container:
```bash
docker run -d -p 8000:8000 --name iris-container iris-mlops-api:latest
```

#### Using Docker Compose:
```bash
docker compose up -d
```

---

### 7. Continuous Integration / Continuous Deployment (CI/CD)

The `.github/workflows/ci-cd.yml` workflow triggers on push / pull request to `main` or `master` and executes:
1. Python environment configuration and dependency caching.
2. Automated reproduction of the DVC pipeline (`dvc repro`).
3. Execution of the complete `pytest` validation suite.
4. Container image build (`docker build`).
5. Live container smoke testing via health check probe.
