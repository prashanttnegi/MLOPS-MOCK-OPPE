from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json
import joblib
import numpy as np
import pandas as pd

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# FastAPI app
app = FastAPI()
model=None

# Dummy model function
def dummy_model(features: dict):
    global model # Explicitly declare model as global to access the loaded model
    if model is None:
        # Raise exception if model didn;t load successfully
        raise HTTPException(status_code=500, detail="Model not loaded yet.")
    time.sleep(0.1)  # Simulate compute
    # Conversion from dict to list
    input = pd.DataFrame([features])
    prediction = model.predict(input)[0]
    return {"predicted class": prediction}

# Input schema
class Input(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Simulated flags, normally these would be set by various parts of the code
# e.g. if model load is taking time due to weights being large, 
#  then is_ready would be False until the model is loaded.
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    global model
    import time
    try:
        model = joblib.load("model.joblib")
        logger.info(json.dumps({"event": "model_loaded", "status": "success"}))
    except Exception as e:
        logger.error(json.dumps({"event": "model_loading_failed", "error": str(e)}))
        # It's good practice to set is_ready to False if model loading fails
        app_state["is_ready"] = False
        app_state["is_alive"] = False # Or even set alive to False if the service is unusable
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    time.sleep(2)  # simulate work, normally this would be model loading
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.post("/predict")
async def predict(input: Input, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = input.dict()
            result = dummy_model(input_data)
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "latency_ms": latency,
                "status": "success"
            }))
            return result

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
