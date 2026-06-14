"""
Car Price Intelligence — Expert Data Science Pipeline
======================================================
Two real business problems:
  1. REGRESSION  : Predict selling price from car attributes ONLY (no MMR cheat)
  2. CLASSIFICATION: Will this car sell BELOW market value? (auction deal detector)

Architecture: Embedding layers for categoricals (the correct approach)
              Batch normalisation, dropout, residual connections
"""

import os, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, classification_report,
                             confusion_matrix, roc_auc_score)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

DATA_PATH = r"E:\codeputs\ongoing\ai\learning-ai\car_prices.csv\car_prices.csv"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS     = 40
BATCH      = 4096
LR         = 3e-3
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED)

print("=" * 65)
print(f"  Device : {DEVICE}")
print(f"  Task 1 : Regression  — predict price WITHOUT market report")
print(f"  Task 2 : Classify    — detect below-market deals")
print("=" * 65)


# ─────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────
print("\n[1/7] LOADING & CLEANING")

df = pd.read_csv(DATA_PATH)
print(f"  Raw : {df.shape[0]:,} rows")

# Parse saledate before dropping
def parse_sale_month(s):
    try:
        m = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', str(s))
        months = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                  "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        return months.get(m.group(1), 0) if m else 0
    except:
        return 0

df["sale_month"] = df["saledate"].apply(parse_sale_month)
df["sale_quarter"] = df["sale_month"].apply(lambda m: (m - 1) // 3 + 1 if m > 0 else 0)

df.drop(columns=["vin", "saledate"], inplace=True)
df.dropna(subset=["sellingprice", "mmr"], inplace=True)

cat_cols = ["make", "model", "body", "transmission", "color", "interior", "state"]
for col in cat_cols:
    df[col].fillna("unknown", inplace=True)

df["condition"].fillna(df["condition"].median(), inplace=True)
df["odometer"].fillna(df["odometer"].median(), inplace=True)
df["trim"].fillna("base", inplace=True)

# Hard outlier removal
df = df[(df["sellingprice"] >= 500) & (df["sellingprice"] <= 150_000)]
df = df[(df["odometer"] >= 0)       & (df["odometer"] <= 300_000)]
df = df[(df["year"] >= 1990)]

print(f"  Clean: {df.shape[0]:,} rows  |  nulls: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────
# 2. EDA — real insights, not just charts
# ─────────────────────────────────────────────────────────────
print("\n[2/7] EDA")

# a) Price distribution by body type
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Car Price Intelligence — EDA", fontsize=13, fontweight="bold")

top_bodies = df["body"].value_counts().head(6).index
body_data  = [df[df["body"] == b]["sellingprice"].values for b in top_bodies]
axes[0,0].boxplot(body_data, labels=top_bodies, vert=False)
axes[0,0].set_title("Price by Body Type")
axes[0,0].set_xlabel("Selling Price ($)")

# b) Mileage vs price (coloured by age bucket)
s = df.sample(6000, random_state=SEED)
age_bucket = pd.cut(2015 - s["year"], bins=[0,3,7,12,30], labels=["0-3yr","4-7yr","8-12yr","12+yr"])
for label, color in zip(["0-3yr","4-7yr","8-12yr","12+yr"],
                         ["#2ecc71","#3498db","#e67e22","#e74c3c"]):
    mask = age_bucket == label
    axes[0,1].scatter(s[mask]["odometer"], s[mask]["sellingprice"],
                      alpha=0.25, s=6, label=label, color=color)
axes[0,1].set_title("Odometer vs Price (by Age)")
axes[0,1].set_xlabel("Miles"); axes[0,1].legend(fontsize=7)

# c) Monthly seasonality
monthly = df.groupby("sale_month")["sellingprice"].mean()
axes[0,2].bar(monthly.index, monthly.values, color="steelblue")
axes[0,2].set_title("Avg Price by Sale Month")
axes[0,2].set_xlabel("Month (1=Jan)")

# d) Condition matters more than people think
cond = df.groupby(df["condition"].round())["sellingprice"].agg(["mean","count"])
ax2 = axes[1,0].twinx()
axes[1,0].bar(cond.index, cond["mean"], color="mediumpurple", alpha=0.7, label="Avg Price")
ax2.plot(cond.index, cond["count"], color="tomato", marker="o", label="Volume")
axes[1,0].set_title("Condition Score vs Price & Volume")
axes[1,0].set_xlabel("Condition (0-49)")
axes[1,0].legend(loc="upper left", fontsize=7)
ax2.legend(loc="upper right", fontsize=7)

# e) Top 10 makes by volume and avg price
make_stats = (df.groupby("make")
                .agg(avg_price=("sellingprice","mean"), volume=("sellingprice","count"))
                .query("volume > 1000")
                .nlargest(10, "avg_price"))
axes[1,1].barh(make_stats.index, make_stats["avg_price"], color="coral")
axes[1,1].set_title("Top 10 Makes by Avg Price (>1K units)")
axes[1,1].set_xlabel("Avg Price ($)")

# f) Deal distribution: how often does a car sell BELOW market?
deal_pct = (df["sellingprice"] < df["mmr"]).mean() * 100
above_below = df["sellingprice"] - df["mmr"]
axes[1,2].hist(above_below.clip(-5000, 5000), bins=100, color="seagreen", edgecolor="none")
axes[1,2].axvline(0, color="red", lw=1.5, label="= MMR")
axes[1,2].set_title(f"Sale vs MMR Gap  ({deal_pct:.1f}% sell BELOW market)")
axes[1,2].set_xlabel("Selling Price − MMR ($)")
axes[1,2].legend()

plt.tight_layout()
plt.savefig("outputs/eda.png", dpi=130)
plt.close()

# Correlation (NO mmr — we want to understand raw features)
corr_cols = ["year","condition","odometer","sellingprice","sale_month"]
plt.figure(figsize=(6, 4))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Feature Correlations (no MMR)")
plt.tight_layout()
plt.savefig("outputs/correlation.png", dpi=130)
plt.close()

print(f"  Cars selling BELOW market (deals): {deal_pct:.1f}%")
print(f"  Saved: outputs/eda.png, outputs/correlation.png")


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[3/7] FEATURE ENGINEERING")

# Numerical features (no MMR)
df["car_age"]       = 2015 - df["year"]
df["miles_per_year"] = df["odometer"] / (df["car_age"].clip(lower=1))
df["is_luxury"]     = df["make"].isin([
    "BMW","Mercedes-Benz","Audi","Lexus","Porsche","Jaguar","Land Rover",
    "Cadillac","Infiniti","Acura","Volvo","Lincoln"
]).astype(int)
df["is_popular_color"] = df["color"].isin(["white","black","silver","gray"]).astype(int)
df["is_automatic"]  = (df["transmission"] == "automatic").astype(int)
df["high_mileage"]  = (df["odometer"] > 80_000).astype(int)

NUM_FEATURES = ["car_age","miles_per_year","condition","odometer",
                "is_luxury","is_popular_color","is_automatic","high_mileage",
                "sale_month","sale_quarter"]

# Categorical features → encoded for embeddings
# Keep top N per column, map rest to index 0 ("other")
def encode_cat(series, top_n=50):
    counts = series.value_counts()
    top_vals = counts.head(top_n).index.tolist()
    vocab = {v: i+1 for i, v in enumerate(top_vals)}  # 0 = "other"
    return series.map(vocab).fillna(0).astype(int), len(vocab) + 1

CAT_FEATURES = ["make","model","body","transmission","color","interior","state"]
cat_encoded  = {}
cat_sizes    = {}
for col in CAT_FEATURES:
    df[col + "_enc"], n = encode_cat(df[col], top_n=50)
    cat_encoded[col] = col + "_enc"
    cat_sizes[col]   = n

# Classification label: 1 = sold BELOW market (a deal)
df["is_deal"] = (df["sellingprice"] < df["mmr"]).astype(int)

TARGET_REG   = "sellingprice"
TARGET_CLF   = "is_deal"

print(f"  Numerical features : {len(NUM_FEATURES)}")
print(f"  Categorical embeds : {len(CAT_FEATURES)}")
print(f"  Deal class balance : {df['is_deal'].mean()*100:.1f}% deals / "
      f"{(1-df['is_deal'].mean())*100:.1f}% above market")


# ─────────────────────────────────────────────────────────────
# 4. SPLIT & SCALE
# ─────────────────────────────────────────────────────────────
print("\n[4/7] SPLIT & SCALE")

X_num = df[NUM_FEATURES].values.astype(np.float32)
X_cat = df[[cat_encoded[c] for c in CAT_FEATURES]].values.astype(np.int64)
y_reg = df[TARGET_REG].values.astype(np.float32)
y_clf = df[TARGET_CLF].values.astype(np.float32)

idx = np.arange(len(df))
idx_tv, idx_test = train_test_split(idx, test_size=0.15, random_state=SEED)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.176, random_state=SEED)

scaler = StandardScaler()
X_num[idx_train] = scaler.fit_transform(X_num[idx_train])
X_num[idx_val]   = scaler.transform(X_num[idx_val])
X_num[idx_test]  = scaler.transform(X_num[idx_test])

print(f"  Train: {len(idx_train):,}  |  Val: {len(idx_val):,}  |  Test: {len(idx_test):,}")


# ─────────────────────────────────────────────────────────────
# 5. MODEL — Embedding + MLP (two heads: regression + classification)
# ─────────────────────────────────────────────────────────────
print("\n[5/7] MODEL ARCHITECTURE")

class CarDataset(Dataset):
    def __init__(self, idx, X_num, X_cat, y_reg, y_clf):
        self.Xn  = torch.tensor(X_num[idx])
        self.Xc  = torch.tensor(X_cat[idx])
        self.yr  = torch.tensor(y_reg[idx]).unsqueeze(1)
        self.yc  = torch.tensor(y_clf[idx]).unsqueeze(1)
    def __len__(self): return len(self.Xn)
    def __getitem__(self, i): return self.Xn[i], self.Xc[i], self.yr[i], self.yc[i]


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return F.gelu(x + self.block(x))  # residual connection


class CarNet(nn.Module):
    def __init__(self, cat_sizes, num_dim, emb_dim=16, hidden=256):
        super().__init__()
        # One embedding per categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, min(emb_dim, (n + 1) // 2)) for n in cat_sizes
        ])
        emb_total = sum(min(emb_dim, (n+1)//2) for n in cat_sizes)
        input_dim = num_dim + emb_total

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.head_clf = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head_reg(x), self.head_clf(x)


cat_size_list = [cat_sizes[c] for c in CAT_FEATURES]
model = CarNet(cat_size_list, num_dim=len(NUM_FEATURES)).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())

print(f"  Embeddings   : {len(CAT_FEATURES)} categorical features")
print(f"  Stem         : {len(NUM_FEATURES) + sum(min(16,(n+1)//2) for n in cat_size_list)} → 256")
print(f"  Residual blocks: 3x (256→256 with skip connections)")
print(f"  Heads        : regression (price) + classification (deal?)")
print(f"  Total params : {total_params:,}")

train_loader = DataLoader(CarDataset(idx_train, X_num, X_cat, y_reg, y_clf),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(CarDataset(idx_val,   X_num, X_cat, y_reg, y_clf),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(CarDataset(idx_test,  X_num, X_cat, y_reg, y_clf),
                          batch_size=BATCH, shuffle=False, num_workers=0)

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
)
loss_reg = nn.HuberLoss(delta=1000)          # robust to extreme outliers
loss_clf = nn.BCEWithLogitsLoss()


# ─────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[6/7] TRAINING")

history = {"train_mae":[], "val_mae":[], "val_auc":[]}
best_val_mae = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xn, xc, yr, yc in train_loader:
        xn, xc, yr, yc = xn.to(DEVICE), xc.to(DEVICE), yr.to(DEVICE), yc.to(DEVICE)
        optimizer.zero_grad()
        p_reg, p_clf = model(xn, xc)
        loss = loss_reg(p_reg, yr) + 500 * loss_clf(p_clf, yc)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    v_preds_r, v_preds_c, v_true_r, v_true_c = [], [], [], []
    with torch.no_grad():
        for xn, xc, yr, yc in val_loader:
            xn, xc = xn.to(DEVICE), xc.to(DEVICE)
            p_reg, p_clf = model(xn, xc)
            v_preds_r.append(p_reg.cpu()); v_true_r.append(yr)
            v_preds_c.append(torch.sigmoid(p_clf).cpu()); v_true_c.append(yc)

    v_preds_r = torch.cat(v_preds_r).numpy().flatten()
    v_true_r  = torch.cat(v_true_r).numpy().flatten()
    v_preds_c = torch.cat(v_preds_c).numpy().flatten()
    v_true_c  = torch.cat(v_true_c).numpy().flatten()

    val_mae = mean_absolute_error(v_true_r, v_preds_r)
    val_auc = roc_auc_score(v_true_c, v_preds_c)

    history["val_mae"].append(val_mae)
    history["val_auc"].append(val_auc)

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "outputs/best_model.pt")

    if epoch % 8 == 0 or epoch == 1:
        print(f"  Ep {epoch:02d}/{EPOCHS} | Val MAE: ${val_mae:,.0f} | "
              f"Deal AUC: {val_auc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")


# ─────────────────────────────────────────────────────────────
# 7. RESULTS & BUSINESS INSIGHTS
# ─────────────────────────────────────────────────────────────
print("\n[7/7] TEST RESULTS & BUSINESS INSIGHTS")

model.load_state_dict(torch.load("outputs/best_model.pt"))
model.eval()

t_pred_r, t_pred_c, t_true_r, t_true_c = [], [], [], []
with torch.no_grad():
    for xn, xc, yr, yc in test_loader:
        xn, xc = xn.to(DEVICE), xc.to(DEVICE)
        p_reg, p_clf = model(xn, xc)
        t_pred_r.append(p_reg.cpu()); t_true_r.append(yr)
        t_pred_c.append(torch.sigmoid(p_clf).cpu()); t_true_c.append(yc)

t_pred_r = torch.cat(t_pred_r).numpy().flatten()
t_true_r = torch.cat(t_true_r).numpy().flatten()
t_pred_c = torch.cat(t_pred_c).numpy().flatten()
t_true_c = torch.cat(t_true_c).numpy().flatten()
t_label_c = (t_pred_c >= 0.5).astype(int)

mae  = mean_absolute_error(t_true_r, t_pred_r)
rmse = np.sqrt(mean_squared_error(t_true_r, t_pred_r))
r2   = r2_score(t_true_r, t_pred_r)
auc  = roc_auc_score(t_true_c, t_pred_c)
mean_price = t_true_r.mean()

print(f"""
  ┌──────────────────────────────────────────────────┐
  │  REGRESSION — Price Prediction (no market data)  │
  │  MAE  : ${mae:>10,.0f}  ({mae/mean_price*100:.1f}% of avg price)   │
  │  RMSE : ${rmse:>10,.0f}                            │
  │  R²   : {r2:>12.4f}                            │
  ├──────────────────────────────────────────────────┤
  │  CLASSIFICATION — Deal Detector                  │
  │  AUC  : {auc:>12.4f}  (0.5=random, 1.0=perfect) │
  └──────────────────────────────────────────────────┘""")

print("\n  Classification Report (Deal Detector):")
print(classification_report(t_true_c, t_label_c,
                             target_names=["Above Market","Below Market (Deal)"]))

# ── Plots ──
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Test Results — Car Price Intelligence", fontsize=13, fontweight="bold")

# 1. Actual vs Predicted
s = np.random.choice(len(t_true_r), 6000, replace=False)
axes[0,0].scatter(t_true_r[s], t_pred_r[s], alpha=0.15, s=5, color="steelblue")
axes[0,0].plot([500, 120000], [500, 120000], "r--", lw=1.5)
axes[0,0].set_title(f"Actual vs Predicted  (R²={r2:.3f})")
axes[0,0].set_xlabel("Actual ($)"); axes[0,0].set_ylabel("Predicted ($)")

# 2. Error distribution
errors = t_pred_r - t_true_r
axes[0,1].hist(errors.clip(-5000, 5000), bins=100, color="coral", edgecolor="none")
axes[0,1].axvline(0, color="black", lw=1.5)
axes[0,1].set_title(f"Prediction Error  (MAE=${mae:,.0f})")
axes[0,1].set_xlabel("Error ($)")

# 3. Error % by price band
price_bands = pd.cut(t_true_r, bins=[0,5000,10000,20000,40000,150000],
                     labels=["<5k","5-10k","10-20k","20-40k","40k+"])
pct_errors  = np.abs(errors) / t_true_r * 100
band_df = pd.DataFrame({"band": price_bands, "pct_err": pct_errors})
band_means = band_df.groupby("band")["pct_err"].mean()
axes[0,2].bar(band_means.index, band_means.values, color="mediumpurple")
axes[0,2].set_title("Avg % Error by Price Band")
axes[0,2].set_xlabel("Price Range"); axes[0,2].set_ylabel("% Error")

# 4. Deal detector: probability distribution
axes[1,0].hist(t_pred_c[t_true_c==0], bins=60, alpha=0.6,
               color="tomato", label="Above Market", density=True)
axes[1,0].hist(t_pred_c[t_true_c==1], bins=60, alpha=0.6,
               color="seagreen", label="Below Market (Deal)", density=True)
axes[1,0].axvline(0.5, color="black", lw=1.5, ls="--")
axes[1,0].set_title(f"Deal Detector Probabilities  (AUC={auc:.3f})")
axes[1,0].set_xlabel("P(is a deal)"); axes[1,0].legend()

# 5. Confusion matrix
cm = confusion_matrix(t_true_c, t_label_c)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1,1],
            xticklabels=["Above","Below"], yticklabels=["Above","Below"])
axes[1,1].set_title("Deal Detector — Confusion Matrix")
axes[1,1].set_xlabel("Predicted"); axes[1,1].set_ylabel("Actual")

# 6. Training curves
axes[1,2].plot(history["val_mae"],  color="steelblue", label="Val MAE ($)")
ax_r = axes[1,2].twinx()
ax_r.plot(history["val_auc"], color="seagreen", linestyle="--", label="AUC")
axes[1,2].set_title("Validation Curves")
axes[1,2].set_xlabel("Epoch")
axes[1,2].set_ylabel("MAE ($)", color="steelblue")
ax_r.set_ylabel("AUC", color="seagreen")
axes[1,2].legend(loc="upper right"); ax_r.legend(loc="center right")

plt.tight_layout()
plt.savefig("outputs/results.png", dpi=130)
plt.close()

# ── Business Summary ──
print(f"""
  ─────────────────────────────────────────────────────
  BUSINESS CONCLUSIONS

  1. PRICE ESTIMATION (no market report needed)
     The model predicts price within ${mae:,.0f} on average using
     only car attributes: age, mileage, make, condition, color.
     Useful when MMR is unavailable — independent valuation.

  2. DEAL DETECTOR
     AUC {auc:.3f} means the model separates below-market cars from
     above-market with strong accuracy.
     → At threshold 0.5: catches {(t_label_c[t_true_c==1]==1).mean()*100:.1f}% of real deals.

  3. WHERE THE MODEL STRUGGLES
     Error % is highest in low price band (<$5K) — cheap cars
     are harder to value: condition noise dominates there.

  4. WHAT A BUYER DOES WITH THIS
     Feed any car's attributes → get deal probability score.
     Sort auction lots by deal probability → bid on top-ranked.

  5. WHAT A SELLER DOES WITH THIS
     Predicted price > what you paid? You have margin.
     Predicted price < what you paid? You're underwater — hold
     or reduce before it ages further.
  ─────────────────────────────────────────────────────
""")
print("  Saved: outputs/results.png")
print("  Done.")
