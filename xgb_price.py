"""
XGBoost Car Price Predictor
===========================
Goal: predict `sellingprice` from car attributes.
Two models are trained on the SAME test split so we can compare:
  A) attributes-only  (deployable, honest)
  B) attributes + mmr (upper bound; mostly echoes MMR)
Saves a JSON metrics/artifact bundle + a comparison chart for the notebook & dashboard.
"""
import os, json, re, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

t0 = time.time()
DATA = r"E:\codeputs\ongoing\ai\learning-ai\car_prices.csv\car_prices.csv"
OUT  = r"E:\codeputs\ongoing\ai\learning-ai\outputs"
os.makedirs(OUT, exist_ok=True)
SEED = 42

print("Loading data...")
df = pd.read_csv(DATA, on_bad_lines="skip", low_memory=False)
print(f"  raw rows: {len(df):,}")

# ---- Clean ---------------------------------------------------------------
df.columns = [c.strip().lower() for c in df.columns]
for c in ["year", "condition", "odometer", "mmr", "sellingprice"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# drop rows with no target / nonsense prices
df = df[df["sellingprice"].notna()]
df = df[(df["sellingprice"] >= 500) & (df["sellingprice"] <= 200000)]
df = df[(df["odometer"].notna()) & (df["odometer"] >= 0)]
df = df[df["year"].notna()]

# saledate -> sale_year / sale_month
def parse_year(s):
    m = re.search(r"\b(20\d{2})\b", str(s));  return int(m.group(1)) if m else np.nan
def parse_month(s):
    months = dict(Jan=1,Feb=2,Mar=3,Apr=4,May=5,Jun=6,Jul=7,Aug=8,Sep=9,Oct=10,Nov=11,Dec=12)
    m = re.search(r"\b([A-Z][a-z]{2})\b", str(s));  return months.get(m.group(1), np.nan) if m else np.nan

df["sale_year"]  = df["saledate"].map(parse_year)
df["sale_month"] = df["saledate"].map(parse_month)
df["car_age"]    = df["sale_year"] - df["year"]
df.loc[(df["car_age"] < 0) | (df["car_age"] > 40), "car_age"] = np.nan

print(f"  clean rows: {len(df):,}")

CATS = ["make", "model", "trim", "body", "transmission", "state", "color", "interior"]
NUMS = ["car_age", "odometer", "condition", "sale_month", "year"]

for c in CATS:
    df[c] = df[c].astype("category")

def build_X(use_mmr):
    cols = CATS + NUMS + (["mmr"] if use_mmr else [])
    return df[cols].copy()

y = df["sellingprice"].values

def train(use_mmr, tag):
    X = build_X(use_mmr)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        tree_method="hist", enable_categorical=True,
        n_jobs=-1, random_state=SEED,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    pred = model.predict(Xte)
    mae  = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2   = r2_score(yte, pred)
    print(f"  [{tag}] MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R2={r2:.4f}")
    return model, dict(mae=mae, rmse=rmse, r2=r2), (Xte, yte, pred)

# ---- Baseline (mean) -----------------------------------------------------
ytr_full, yte_full = train_test_split(y, test_size=0.2, random_state=SEED)
base_pred = np.full_like(yte_full, ytr_full.mean(), dtype=float)
base = dict(mae=mean_absolute_error(yte_full, base_pred),
            rmse=mean_squared_error(yte_full, base_pred) ** 0.5, r2=0.0)
print(f"  [baseline-mean] MAE=${base['mae']:,.0f}  RMSE=${base['rmse']:,.0f}")

print("\nTraining XGBoost (attributes only)...")
m_no, met_no, (Xte, yte, pred_no) = train(False, "no-mmr")
print("Training XGBoost (attributes + mmr)...")
m_yes, met_yes, _ = train(True, "with-mmr")

# ---- Feature importance (attributes-only model) --------------------------
imp = m_no.get_booster().get_score(importance_type="gain")
imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:12]

# ---- Save model + metadata for the API/dashboard -------------------------
m_no.save_model(os.path.join(OUT, "xgb_price_no_mmr.json"))
m_yes.save_model(os.path.join(OUT, "xgb_price_with_mmr.json"))

# category vocab so the API can rebuild the same dtype
vocab = {c: [str(v) for v in df[c].cat.categories.tolist()] for c in CATS}
meta = dict(
    cats=CATS, nums=NUMS,
    vocab_top={c: sorted(df[c].value_counts().head(40).index.astype(str).tolist()) for c in CATS},
    metrics=dict(baseline=base, no_mmr=met_no, with_mmr=met_yes),
    importance=imp,
    n_rows=int(len(df)),
    year_range=[int(df["year"].min()), int(df["year"].max())],
)
with open(os.path.join(OUT, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
with open(os.path.join(OUT, "cat_vocab.json"), "w") as f:
    json.dump(vocab, f)

# ---- Recent sales snapshot for the dashboard "latest prices" -------------
recent = (df.dropna(subset=["sale_year", "sale_month"])
            .sort_values(["sale_year", "sale_month"], ascending=False)
            .head(400)[["year","make","model","trim","body","odometer",
                        "condition","color","sellingprice","mmr","state"]])
recent.to_json(os.path.join(OUT, "recent_sales.json"), orient="records")

# ---- Charts --------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
models = ["Baseline", "XGB (no MMR)", "XGB (+MMR)"]
maes   = [base["mae"], met_no["mae"], met_yes["mae"]]
r2s    = [base["r2"], met_no["r2"], met_yes["r2"]]
ax[0].bar(models, maes, color=["#94a3b8","#2563eb","#16a34a"])
ax[0].set_title("Mean Absolute Error (lower = better)"); ax[0].set_ylabel("$ error")
for i,v in enumerate(maes): ax[0].text(i, v, f"${v:,.0f}", ha="center", va="bottom")
ax[1].bar(models, r2s, color=["#94a3b8","#2563eb","#16a34a"])
ax[1].set_title("R2 score (higher = better)"); ax[1].set_ylim(0,1)
for i,v in enumerate(r2s): ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "xgb_comparison.png"), dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
labels = [k for k,_ in imp][::-1]; vals = [v for _,v in imp][::-1]
ax[0].barh(labels, vals, color="#2563eb"); ax[0].set_title("XGBoost feature importance (gain)")
idx = np.random.RandomState(0).choice(len(yte), 4000, replace=False)
ax[1].scatter(yte[idx], pred_no[idx], s=4, alpha=0.25, color="#2563eb")
lims=[0, min(120000, yte.max())]; ax[1].plot(lims, lims, "r--")
ax[1].set_xlabel("Actual $"); ax[1].set_ylabel("Predicted $")
ax[1].set_title(f"Predicted vs Actual (no-MMR)  R2={met_no['r2']:.3f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "xgb_diagnostics.png"), dpi=110); plt.close()

print(f"\nDone in {time.time()-t0:.0f}s. Artifacts written to outputs/.")
print(json.dumps(meta["metrics"], indent=2))
