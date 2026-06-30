"""
deployment/app.py
FastAPI service that loads the fine-tuned DistilBERT sentiment model (Model V2)
and exposes prediction endpoints.

Endpoints:
    GET  /         -> basic info
    GET  /health   -> health check
    POST /predict  -> sentiment prediction for input text

Run locally:
    uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/model_v2")
FALLBACK_MODEL = "distilbert-base-uncased"  # used if a fine-tuned model isn't present yet

app = FastAPI(
    title="IMDB Sentiment Classifier API",
    description="Serves predictions from a fine-tuned DistilBERT sentiment model (Model V2).",
    version="1.0.0",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    raw_score_negative: float
    raw_score_positive: float


@app.on_event("startup")
def load_model():
    global tokenizer, model
    model_path = MODEL_DIR if os.path.isdir(MODEL_DIR) else FALLBACK_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")


@app.get("/")
def root():
    return {
        "message": "IMDB Sentiment Classifier API",
        "endpoints": ["/", "/health", "/predict"],
        "model": MODEL_DIR if os.path.isdir(MODEL_DIR) else FALLBACK_MODEL,
    }


@app.get("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return {"status": status}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty")

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    label = "positive" if probs[1] > probs[0] else "negative"
    confidence = max(probs)

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        raw_score_negative=round(probs[0], 4),
        raw_score_positive=round(probs[1], 4),
    )
