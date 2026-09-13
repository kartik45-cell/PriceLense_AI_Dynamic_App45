"""End-to-end recommendation generation."""
from pathlib import Path
import pandas as pd
from .data_processing import load_raw_data, clean_transactions, save_processed
from .feature_engineering import make_model_frame
from .demand_model import train_demand_model
from .elasticity import estimate_elasticity
from .optimizer import optimize_price


def run(data_dir: str | Path, output_dir: str | Path, objective: str = "revenue") -> dict:
    raw, meals = load_raw_data(data_dir)
    clean = clean_transactions(raw, meals)
    save_processed(clean, data_dir)
    frame = make_model_frame(clean)
    model = train_demand_model(frame)
    latest = frame.sort_values("week").groupby(["center_id", "meal_id"], as_index=False).tail(1)
    recommendations = []
    for _, row in latest.iterrows():
        sim = optimize_price(model, row, objective=objective)
        best = sim[sim.is_recommended].iloc[0]
        recommendations.append({"center_id": row.center_id, "meal_id": row.meal_id, "current_price": row.checkout_price,
            "recommended_price": best.checkout_price, "price_change_pct": best.checkout_price / row.checkout_price - 1,
            "expected_demand": best.predicted_demand, "expected_revenue": best.expected_revenue,
            "demand_loss_vs_current": best.demand_loss_vs_current})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recommendations).sort_values("expected_revenue", ascending=False).to_csv(output_dir / "pricing_recommendations.csv", index=False)
    estimate_elasticity(clean).to_csv(output_dir / "meal_elasticity.csv", index=False)
    pd.DataFrame([model.metrics]).to_csv(output_dir / "model_metrics.csv", index=False)
    return model.metrics
