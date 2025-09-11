from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io

app = FastAPI(title="IDS Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

artifact = joblib.load("./models/model.pkl")
pipeline = artifact["pipeline"]
threshold = artifact["threshold"]
attack_idx = artifact["attack_idx"]


@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}


@app.post("/predict-json")
def predict_json(record: dict):
    df = pd.DataFrame([record])
    proba = pipeline.predict_proba(df)[0][attack_idx]
    label = "attack" if proba >= threshold else "normal"
    return {"label": label, "confidence": float(proba)}


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    results = []
    for _, row in df.iterrows():
        proba = pipeline.predict_proba(pd.DataFrame([row]))[0][attack_idx]
        label = "attack" if proba >= threshold else "normal"
        results.append({"label": label, "confidence": float(proba)})
    return {"results": results}



