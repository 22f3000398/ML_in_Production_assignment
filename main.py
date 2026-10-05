from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import pickle
import os
import logging
import json
import time

from datetime import datetime, timezone


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "california_housing_model.pkl"
)


app = FastAPI(
    title="California Housing Price Prediction API",
    description="A simple FastAPI prediction API for California housing prices.",
    version="1.0.0",
)




logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger("ml_api")


def log_prediction_event(
    request_body: dict,
    prediction: dict,
    latency_ms: float
):
    """Log a single prediction event as structured JSON."""

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": request_body,
        "output": prediction,
        "latency_ms": round(latency_ms, 2),
    }

    logger.info(json.dumps(event))




@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    logger.info(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
    }))

    return response


# --- Load model once when API starts ------------------------------

try:

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:

    model = None


# --- Request format ----------------------------------------------

class PredictionRequest(BaseModel):

    MedInc: float = Field(..., ge=0)
    HouseAge: float = Field(..., ge=0)
    AveRooms: float = Field(..., ge=0)
    AveBedrms: float = Field(..., ge=0)
    Population: float = Field(..., ge=0)
    AveOccup: float = Field(..., ge=0)
    Latitude: float = Field(..., ge=-90, le=90)
    Longitude: float = Field(..., ge=-180, le=180)


# --- Response format ---------------------------------------------

class PredictionResponse(BaseModel):

    prediction: float
    prediction_dollars: str


# --- Home page ---------------------------------------------------

@app.get("/")
def root():

    return FileResponse(
        os.path.join(
            os.path.dirname(__file__),
            "static",
            "index.html"
        )
    )


# --- Health check ------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# --- Prediction --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Make sure california_housing_model.pkl exists."
            )
        )

    # Start prediction timer
    start = time.perf_counter()

    # Get request data
    request_body = request.model_dump()

    # IMPORTANT:
    # Same feature order used during model training
    features = [[
        request.MedInc,
        request.HouseAge,
        request.AveRooms,
        request.AveBedrms,
        request.Population,
        request.AveOccup,
        request.Latitude,
        request.Longitude
    ]]

    # Make prediction
    prediction = float(
        model.predict(features)[0]
    )

    # Convert from $100,000 units to dollars
    prediction_dollars = prediction * 100000

    result = {
        "prediction": prediction,
        "prediction_dollars": f"${prediction_dollars:,.2f}"
    }

    # Calculate prediction latency
    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    # Log prediction
    log_prediction_event(
        request_body,
        result,
        latency_ms
    )

    return PredictionResponse(
        prediction=prediction,
        prediction_dollars=f"${prediction_dollars:,.2f}"
    )