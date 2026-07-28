"""
Car Price API — serves the real trained XGBoost model.
Run:  uvicorn main:app --reload --port 8000   (from the api/ folder)
Endpoints:
  GET  /health          -> status + metrics
  GET  /meta            -> dropdown options (makes, bodies, colors, ...)
  GET  /recent?limit=50 -> recent market sales for the dashboard
  POST /predict         -> predict selling price for one car
"""
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "outputs")

# ---- load artifacts ------------------------------------------------------
model = xgb.XGBRegressor()
model.load_model(os.path.join(OUT, "xgb_price_no_mmr.json"))
with open(os.path.join(OUT, "model_meta.json")) as f:
    META = json.load(f)
with open(os.path.join(OUT, "cat_vocab.json")) as f:
    VOCAB = json.load(f)
with open(os.path.join(OUT, "recent_sales.json")) as f:
    RECENT = json.load(f)

CATS = META["cats"]
NUMS = META["nums"]

app = FastAPI(title="Car Price Intelligence API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Car(BaseModel):
    year: int = 2015
    make: str = "Kia"
    model: str = "Sorento"
    trim: str = "LX"
    body: str = "SUV"
    transmission: str = "automatic"
    state: str = "ca"
    condition: float = 30
    odometer: float = 30000
    color: str = "white"
    interior: str = "black"
    sale_year: int = 2015
    sale_month: int = 6


@app.get("/health")
def health():
    return {"status": "ok", "rows_trained_on": META["n_rows"], "metrics": META["metrics"]["no_mmr"]}


@app.get("/meta")
def meta():
    return {
        "options": META["vocab_top"],
        "year_range": META["year_range"],
        "metrics": META["metrics"],
        "importance": META["importance"],
    }


@app.get("/recent")
def recent(limit: int = 50):
    return RECENT[:limit]


def _row(car: Car) -> pd.DataFrame:
    car_age = car.sale_year - car.year
    if car_age < 0 or car_age > 40:
        car_age = np.nan
    data = {
        "make": car.make, "model": car.model, "trim": car.trim, "body": car.body,
        "transmission": car.transmission, "state": car.state, "color": car.color,
        "interior": car.interior,
        "car_age": car_age, "odometer": car.odometer, "condition": car.condition,
        "sale_month": car.sale_month, "year": car.year,
    }
    df = pd.DataFrame([data])
    # rebuild categorical dtype with the SAME categories used in training
    for col in CATS:
        df[col] = pd.Categorical(df[col].astype(str), categories=VOCAB[col])
    for col in NUMS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[CATS + NUMS]


@app.post("/predict")
def predict(car: Car):
    X = _row(car)
    price = float(model.predict(X)[0])
    mae = META["metrics"]["no_mmr"]["mae"]
    return {
        "predicted_price": round(price, 2),
        "range_low": round(price - mae, 2),
        "range_high": round(price + mae, 2),
        "confidence_note": f"typical error +/- ${mae:,.0f}",
    }
