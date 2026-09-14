# PriceLens AI — AI-Powered Dynamic Pricing & Revenue Optimization

A Streamlit decision-support product that turns historical food-delivery transaction
data into center/meal-level price recommendations, built on top of an existing,
already-validated demand-modeling pipeline.

---

## Problem Statement

Businesses frequently change prices or offer discounts, but they don't know whether
the change generated additional demand or simply reduced revenue on sales that would
have happened anyway. Decision-makers need a repeatable way to answer:

> **"What price should I test next?"**

## Solution

PriceLens AI analyzes historical transaction behavior, predicts demand under
different price scenarios with a validated demand model, and searches through
safe price scenarios (within configurable business guardrails) to recommend the
price that maximizes **modeled expected revenue**. It never changes prices
automatically — every recommendation is decision support for a human-approved,
controlled test.

---

## Architecture

```
Historical Transactions
        |
  Data Cleaning              (src/data_processing.py)
        |
  Feature Engineering        (src/feature_engineering.py)
        |
  Demand Modeling            (src/demand_model.py)
        |
  Price Elasticity Analysis  (src/elasticity.py)
        |
  Price Scenario Simulation  (src/optimizer.py)
        |
  Optimization
        |
  Guardrail Filtering
        |
  AI Price Recommendation    (src/pricing_engine.py -> outputs/recommendations/)
        |
  PriceLens AI (Streamlit)   (app.py, src/dashboard_utils.py, src/visualization.py)
```

The Streamlit app is a presentation and simulation layer on top of the existing
pipeline. It does not replace any modeling logic — it loads the pipeline's own
outputs and, for the live Price Simulator, retrains the same model in-memory
(cached) so a manager can move a price slider and see an instant, consistent
prediction.

## Dataset

`data/processed/transactions_clean.csv` — weekly center/meal transaction records
(price, discount, promotion flags, orders, category, cuisine) across 77 centers
and 51 meals, weeks 1-145. Produced from raw order data by
`src/data_processing.py`.

## Machine Learning Methodology

- **Feature engineering** (`src/feature_engineering.py`): leakage-safe features —
  lagged and rolling historical demand (`lag_1_orders`, `lag_4_orders`,
  `rolling_4_orders`), seasonality (`week_sin`/`week_cos`), discount percentage,
  price-vs-base ratio, and a combined promotion flag. All historical-demand
  features are shifted so no feature for week *t* ever contains information from
  week *t*.
- **Model** (`src/demand_model.py`): `regularized_log_demand` — a dependency-free,
  closed-form ridge regression on `log1p(num_orders)` using a small nonlinear
  design matrix (log price, discount, promotion interactions, seasonality,
  center/meal identifiers, log-lag demand terms).
- **Model validation**: strict **time-based holdout**, not a random split — the
  model trains through week 133 and is evaluated on the following 12 weeks, to
  mimic how it would actually be used (predicting demand it hasn't seen yet).

  | Metric | Value |
  |---|---|
  | MAE | 80.75 |
  | RMSE | 165.48 |
  | R2 | 0.714 |
  | Train through week | 133 |
  | Test horizon | 12 weeks |

  **R2 is a regression fit statistic, not classification accuracy.** R2 = 0.714
  means the model explains ~71.4% of the variance in demand on the holdout — it
  is never described as "71.4% accuracy" anywhere in this app.

- **Price elasticity** (`src/elasticity.py`): a descriptive, regularized,
  per-meal regression of log demand on log price (controlling for discount and
  promotion), reported as a historical association — explicitly not a causal
  estimate.

## Optimization Methodology

For each center-meal combination, `src/optimizer.py` evaluates a grid of
candidate prices around the current price. For each candidate, the demand model
predicts demand; expected revenue is computed as `price x predicted demand`. The
system selects the candidate with the highest modeled expected revenue among
those that satisfy the configured guardrails. This is **revenue optimization**,
not profit optimization — there is no verified unit-cost column in the dataset,
so profit is never claimed.

## Guardrails

Configurable in the app sidebar:

- **Maximum price change** — default +/-20%
- **Maximum modeled demand loss** — default 15%

Every recommendation is labeled **PASS**, **CAUTION** (near a limit), or
**BLOCKED** (exceeds a limit). The system never auto-publishes a price.

## Dashboard Features (PriceLens AI Streamlit app)

1. **Overview** — executive KPIs and five portfolio-level charts.
2. **AI Recommendations** — filterable table of all recommendations with a
   detailed "why this price?" recommendation card per row.
3. **Price Simulator** — live, slider-driven what-if scenario using the same
   trained model and optimizer, with current-vs-proposed comparison and
   price-response charts.
4. **Demand & Revenue Analytics** — historical trends and relationships from
   actual transaction data.
5. **Price Elasticity** — beginner-friendly elasticity bands and visuals from
   `meal_elasticity.csv`.
6. **Model Performance** — MAE/RMSE/R2 with plain-language explanations, plus
   actual-vs-predicted and residual charts recomputed live from the holdout.
7. **Business Insights** — auto-generated Insight/Impact/Action cards from the
   live recommendation data.
8. **Methodology / About** — full pipeline explanation, data limitations,
   production roadmap, and CSV downloads.

## An issue found and resolved while building this app

`model/model.pkl` and `model/model_features.pkl` are an earlier exploratory
**XGBoost** model with one-hot encoded center/meal features — a different
schema from the `regularized_log_demand` pipeline that actually produced
`outputs/recommendations/*.csv`. Using the pickle for live predictions would
have silently produced numbers inconsistent with the rest of the app (a second,
disagreeing pricing algorithm). Instead, the app retrains the documented
`regularized_log_demand` model at startup (cached) — verified to reproduce the
exact MAE = 80.75, RMSE = 165.48, R2 = 0.714 shown in `model_metrics.csv` — and
uses it consistently for both the static tables and the live simulator. The
legacy pickle files are kept in the repo for provenance and explained on the
Methodology page.

## Limitations

- The dataset is **observational**: historical price/demand associations do not
  establish that a price change *caused* a demand change.
- Elasticity estimates are descriptive, not causal.
- No verified unit-cost data exists, so only revenue (not profit) is optimized.
- Recommendations assume the historical demand pattern continues to hold.

## Future Improvements / Production Roadmap

1. A/B testing before full rollout
2. Inventory constraints
3. Capacity constraints
4. Competitor pricing signals
5. Cost/margin integration for true profit optimization
6. Real-time data ingestion
7. Human approval workflow
8. Monitoring and model-drift detection

---

## Local Setup

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
streamlit run app.py
```

The app reads directly from `data/processed/transactions_clean.csv` and
`outputs/recommendations/*.csv`. If you change the underlying data or modeling
code, regenerate those outputs first:

```bash
python run_pipeline.py
```

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **New app**.
4. Select this repository and branch.
5. Set the main file to `app.py`.
6. Click **Deploy**.

No API keys, no paid services, no Azure/OpenAI dependency — the app runs
entirely on the bundled CSV/model artifacts using Streamlit, Pandas, NumPy, and
Plotly. All file paths are relative (via `pathlib.Path`), so it works
identically on Streamlit Cloud and locally.

