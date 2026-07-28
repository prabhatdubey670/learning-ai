# 🚗 Car Price Intelligence

Predict used-car auction prices with **XGBoost**, explore the model in a narrated
Jupyter notebook, and price cars in real time from a **Next.js dashboard** backed by a
**FastAPI** service that serves the actual trained model.

> **Headline:** Guessing the average price is **~$7,058 off** per car.
> This XGBoost model, using **only car specs** (no MMR leakage), predicts within
> **~$1,081 on average** and explains **96.4% of price variation** (R² = 0.964).

---

## 📊 Results

| Model | MAE (avg $ error) | RMSE | R² |
|---|---|---|---|
| Baseline (predict the mean) | $7,058 | $9,639 | 0.00 |
| **XGBoost — specs only** ⭐ *(deployed)* | **$1,081** | $1,836 | **0.964** |
| XGBoost — specs + MMR | $897 | $1,488 | 0.976 |

Trained on **553,412** cleaned auction records. MMR (the wholesale benchmark) only
improves accuracy by ~$184 — proof the model doesn't *need* it and works on any car.

---

## 🗂️ Project structure

```
├── car_price_prediction.ipynb   # The narrated notebook (EDA → features → model → conclusion story)
├── xgb_price.py                 # Training pipeline — produces everything in outputs/
├── car_price_analysis.py        # Earlier PyTorch (embeddings) experiment, kept for reference
├── car_prices_challenge.md      # The original data-science challenge brief
├── api/                         # FastAPI service that serves the trained model
│   ├── main.py                  #   GET /health /meta /recent  ·  POST /predict
│   └── requirements.txt
├── dashboard/                   # Next.js real-time pricing dashboard
│   ├── app/                     #   KPI tiles, predictor, charts, live sales feed
│   ├── components/
│   └── lib/api.ts
└── outputs/                     # Trained models + metadata + charts (committed)
    ├── xgb_price_no_mmr.json     #   the deployed model
    ├── model_meta.json           #   metrics, feature importance, dropdown options
    ├── cat_vocab.json            #   category vocab for inference
    └── recent_sales.json         #   recent sales for the dashboard feed
```

> **The dataset (`car_prices.csv`, ~84 MB) is not committed** — it's too large for git.
> Grab the "Used Car Auction Prices" CSV (Kaggle) and place it at
> `car_prices.csv/car_prices.csv` to re-run the notebook or `xgb_price.py`.
> The dashboard + API run **without** it, from the artifacts in `outputs/`.

---

## 🚀 Run it

### 1. Backend (FastAPI — serves the model)

```bash
python -m venv .venv
.venv/Scripts/pip install -r api/requirements.txt   # Windows
cd api
uvicorn main:app --port 8000
```

### 2. Dashboard (Next.js)

```bash
cd dashboard
npm install
npm run dev        # http://localhost:3000
```

The dashboard proxies `/api/*` → `http://localhost:8000` (configurable via
`NEXT_PUBLIC_API_URL`). Open **http://localhost:3000**, enter a car (or hit
**🎲 Load a real car**) and get a live price with a confidence range.

### 3. Retrain (optional)

```bash
.venv/Scripts/python xgb_price.py       # regenerates everything in outputs/
```

---

## 🧠 How it works (the honest version)

- **Features:** `make, model, trim, body, transmission, state, color, interior`
  (categorical, native XGBoost handling) + `car_age, odometer, condition, sale_month, year`.
- **Dropped on purpose:** `vin` (unique ID → memorisation), `seller` (noise), and
  `mmr` (a price predicting a price — leakage) from the deployed model.
- **`car_age = sale_year − year`** beats raw `year`: it directly encodes depreciation.
- **Not production-ready as-is.** Before trusting it live you'd add: scheduled
  retraining (concept drift), prediction intervals, and guardrails for unseen
  makes/models. See the notebook's Phase 5 conclusion for the full critique.

*Built as a freelance-style, end-to-end data-science deliverable: model → notebook → API → dashboard.*
