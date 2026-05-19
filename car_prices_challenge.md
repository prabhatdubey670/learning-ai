# Car Prices Dataset — End-to-End ML Challenge

**Dataset:** `car_prices.csv` — 558,837 used car auction records  
**Target Variable:** `sellingprice`  
**Date Created:** 2026-05-19  

## Columns
`year`, `make`, `model`, `trim`, `body`, `transmission`, `vin`, `state`, `condition`, `odometer`, `color`, `interior`, `seller`, `mmr`, `sellingprice`, `saledate`

`mmr` = Manheim Market Report price (wholesale benchmark)

---

## Phase 1 — Understand Before You Touch Code
*Estimated time: 2-3 hours of thinking. No keyboard.*

Answer these in writing before opening the notebook:

1. `sellingprice` is your target. What does `mmr` represent? Why is it dangerous to use `mmr` as a feature if your goal is to predict `sellingprice`? What specific problem does that create?

2. `condition` is a column. Before looking at its values — what are 3 hypotheses about how it's measured? What values would you *expect* it to have?

3. `vin` is a unique identifier per car. Should you use it as a feature? Why or why not? What rule does this establish for ALL unique identifiers?

4. `saledate` is a timestamp. Name 2 ways the date could affect selling price that have nothing to do with the car itself.

5. The dataset has 558K rows. Why is that NOT a reason to skip exploratory analysis and jump straight to modeling?

**Write answers to all 5 before Phase 2. No exceptions.**

---

## Phase 2 — Exploratory Data Analysis (EDA)
*Deliverable: A notebook with observations, not just plots.*

### 2A — Data Quality Audit
- How many nulls per column? Which columns have >5% missing?
- What are the unique values in `condition`? Was your Phase 1 hypothesis right?
- What is the min/max/mean of `sellingprice`? Are there rows where `sellingprice` is 0 or negative? What do you do with them?
- Are there duplicate VINs? What does a duplicate VIN mean in this context?

### 2B — Distribution Analysis
- Plot the distribution of `sellingprice`. Is it normal? Skewed? What does that tell you about your modeling approach?
- Plot `sellingprice` vs `odometer`. Describe the relationship in one sentence.
- Plot `sellingprice` vs `year`. What trend do you observe?
- Which `make` appears most? Which has the highest median selling price?

### 2C — The Dangerous Question
- Plot `sellingprice` vs `mmr`. Compute the correlation. What do you observe?
- Go back and revisit your Phase 1 answer about `mmr`. Update it if needed.

**After each plot, write one sentence: "This tells me ___."**  
Plots without interpretation are useless.

---

## Phase 3 — Feature Engineering
*The most important phase. This is where ML actually happens — not in the model.*

### 3A — Handle Missing Data
For each column with nulls, decide: drop row / fill with mode / fill with median / create "unknown" category.  
Write your reasoning for each decision in a comment.

### 3B — Create New Features
- `car_age` = current year - `year`. Why is this better than using `year` raw?
- From `saledate`: extract `sale_month` and `sale_year`. Why might month matter?
- `price_vs_mmr` ratio: `sellingprice / mmr`. Don't use as a feature (why not?) — use it to spot bad rows.

### 3C — Encode Categorical Variables
- `make`, `model`, `body`, `transmission`, `color`, `interior`, `state` are categorical. How do you convert them to numbers?
- Two approaches: One-Hot Encoding vs Label Encoding. When do you use which?
- `model` has hundreds of unique values. What problem does one-hot encoding create there?

### 3D — Feature Selection
For every feature you keep, write: "I'm using this because ___."  
For every feature you drop, write: "I'm dropping this because ___."

---

## Phase 4 — Modeling
*Do this in order. Don't jump to neural networks.*

### 4A — Baseline First (mandatory)
- Predict `sellingprice` = mean of all training `sellingprice`
- Compute MAE and RMSE
- Any model that doesn't beat this is useless

### 4B — Model 1: Linear Regression
- Train on 80%, test on 20%
- Compute MAE, RMSE, R² on test set
- Did it beat the baseline? By how much?
- Find 5 predictions with the largest error. What do they have in common?

### 4C — Model 2: Random Forest
- Train with default hyperparameters first
- Compare MAE/RMSE to Linear Regression
- Extract feature importances. What are the top 5? Does this surprise you?

### 4D — Model 3: XGBoost (stretch)
- Compare all 3 models on the same test set
- Make a table:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | | | |
| Linear Regression | | | |
| Random Forest | | | |
| XGBoost | | | |

**Checkpoint:** After 4C, answer — "Why might Random Forest outperform Linear Regression on this dataset?" Answer in 2-3 sentences using what you know about the data distribution.

---

## Phase 5 — Conclusions and Critique
*Where most beginners fail — they stop at "model trained, accuracy good."*

Answer these before calling the project done:

1. What is the MAE of your best model? In plain English: "My model's predictions are off by $___ on average." Is that acceptable for an auction house making buy/sell decisions?

2. Your model was trained on auction data. Would it work for predicting prices on Craigslist listings? Why or why not? What concept does this test?

3. Name 2 ways your model could be completely wrong in production even if test MAE was low.

4. What is the ONE most important feature for predicting selling price? Did that match your intuition from Phase 1?

5. If a client asked: "Is this model production-ready?" — what would you say, and what 3 things would you need to do before it was?

---

## Deliverables Checklist

| # | What | Done when |
|---|---|---|
| 1 | Phase 1 written answers | Before touching keyboard |
| 2 | EDA notebook with interpreted plots | Every plot has a "this tells me" sentence |
| 3 | Feature engineering notebook section | Every drop/keep decision has a written reason |
| 4 | Model comparison table | 3 models, same test set, same metrics |
| 5 | Phase 5 written answers | Honest critique, not marketing |

---

## The Trap to Avoid

You will feel the urge to copy-paste ML code and run it. That gives you output with no understanding.

The challenge is not "can you get a model to run."  
The challenge is: **can you explain every decision you made, every number in your output, and every way the model could fail.**

If you can't explain a line of code, delete it.

---

## Progress Log
*Update this as you complete phases.*

- [ ] Phase 1 — Written answers complete
- [ ] Phase 2 — EDA complete
- [ ] Phase 3 — Feature engineering complete
- [ ] Phase 4 — Models trained and compared
- [ ] Phase 5 — Critique complete
