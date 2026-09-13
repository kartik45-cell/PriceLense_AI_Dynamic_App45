# Pulse browser dashboard

The dashboard is a browser-based model studio for the generated pricing outputs. It includes a model explainer, live guardrail controls, demand-response charts, recommendation filtering, and CSV export.

## Run locally

Run the pipeline first from the project root, then serve the **project root** so the dashboard can read the generated CSV files:

```powershell
cd dynamic-pricing-analytics
& "C:\Users\karti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" run_pipeline.py
& "C:\Users\karti\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m http.server 8000
```

Open `http://localhost:8000/dashboard/`.

The browser controls update a local scenario preview. They do not retrain the Python model; use `run_pipeline.py` to regenerate the source recommendations after changing the underlying data or model code.

## Beginner workflow

1. Open **Start here** in the left menu.
2. Choose **Recommended** for your first test. It uses a 20% maximum price change and a 15% maximum modeled demand loss.
3. Read **What could happen?** before running anything.
4. Select **Preview this choice**. Human approval stays on, so this never publishes a price automatically.
5. Use **Careful** for smaller experiments or **Growth** when you are ready to test a larger change.

## Power BI dashboard plan

Load these files from `../outputs/recommendations/`:

- `pricing_recommendations.csv`
- `meal_elasticity.csv`
- `model_metrics.csv`

Use `meal_id` to relate recommendations to elasticity. Add meal/category information from `../data/raw/meal_info.csv`.

## One-page layout

Top cards: model R², average predicted revenue, number of recommendations, and average price change.

Left: slicers for center, meal, category and elasticity band. Center: clustered bars comparing current and recommended price by meal. Right: scatter plot of elasticity versus price change; size by expected revenue. Bottom: recommendation table with current price, recommended price, predicted demand, expected revenue, and demand-loss guardrail.

## Demo interaction

Select one high-revenue meal, show its current and recommended price, then explain: “This price wins within a ±20% guardrail and keeps modeled demand loss below 15%.”
