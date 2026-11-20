import os
import time
from typing import List, Dict
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# App initialization
app = FastAPI(
    title="Iris Flower Classification API",
    description="MLOps production service to classify Iris species based on sepal & petal measurements.",
    version="1.0.0",
)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
model_artifact = None


def get_model():
    global model_artifact
    if model_artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Train the model first via 'dvc repro' or 'python src/train.py'."
            )
        model_artifact = joblib.load(MODEL_PATH)
    return model_artifact


# Pydantic Schemas
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=20.0, description="Sepal length in centimeters", json_schema_extra={"example": 5.1})
    sepal_width: float = Field(..., ge=0.0, le=20.0, description="Sepal width in centimeters", json_schema_extra={"example": 3.5})
    petal_length: float = Field(..., ge=0.0, le=20.0, description="Petal length in centimeters", json_schema_extra={"example": 1.4})
    petal_width: float = Field(..., ge=0.0, le=20.0, description="Petal width in centimeters", json_schema_extra={"example": 0.2})


class IrisBatchRequest(BaseModel):
    samples: List[IrisFeatures]


class PredictionResponse(BaseModel):
    species: str
    class_id: int
    probabilities: Dict[str, float]
    confidence: float
    input_features: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    latency_ms: float


@app.get("/", tags=["General"])
def root():
    return {
        "project": "Iris Flower Classification using MLOps",
        "description": "Identifies species (Setosa, Versicolor, Virginica) using 4 numeric measurements.",
        "algorithm": "Random Forest Classifier",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
        },
    }


@app.get("/health", tags=["Monitoring"])
def health():
    try:
        art = get_model()
        model_loaded = art is not None
    except Exception:
        model_loaded = False

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
    }


@app.get("/info", tags=["Metadata"])
def get_info():
    art = get_model()
    clf = art["model"]
    return {
        "algorithm": "Random Forest Classifier",
        "feature_names": art["feature_names"],
        "target_names": art["target_names"],
        "classes": art["classes"],
        "hyperparameters": {
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "random_state": clf.random_state,
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: IrisFeatures):
    try:
        art = get_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {str(e)}",
        )

    clf = art["model"]
    target_names = art["target_names"]

    data = pd.DataFrame([{
        "sepal_length": features.sepal_length,
        "sepal_width": features.sepal_width,
        "petal_length": features.petal_length,
        "petal_width": features.petal_width,
    }])

    pred_idx = int(clf.predict(data)[0])
    species_name = target_names[pred_idx]
    probs = clf.predict_proba(data)[0]

    prob_dict = {
        target_names[i]: round(float(probs[i]), 4)
        for i in range(len(target_names))
    }

    return PredictionResponse(
        species=species_name,
        class_id=pred_idx,
        probabilities=prob_dict,
        confidence=round(float(max(probs)), 4),
        input_features=features.model_dump(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: IrisBatchRequest):
    start_time = time.time()
    try:
        art = get_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {str(e)}",
        )

    if not batch.samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch list of samples is empty",
        )

    clf = art["model"]
    target_names = art["target_names"]

    df = pd.DataFrame([s.model_dump() for s in batch.samples])
    preds = clf.predict(df)
    probs_matrix = clf.predict_proba(df)

    results = []
    for i, s in enumerate(batch.samples):
        pred_idx = int(preds[i])
        species_name = target_names[pred_idx]
        probs = probs_matrix[i]
        prob_dict = {
            target_names[j]: round(float(probs[j]), 4)
            for j in range(len(target_names))
        }
        results.append(
            PredictionResponse(
                species=species_name,
                class_id=pred_idx,
                probabilities=prob_dict,
                confidence=round(float(max(probs)), 4),
                input_features=s.model_dump(),
            )
        )

    latency = round((time.time() - start_time) * 1000, 2)

    return BatchPredictionResponse(
        predictions=results,
        count=len(results),
        latency_ms=latency,
    )
