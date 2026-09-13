"""Constrained price simulations for revenue or margin decisions."""
import numpy as np
import pandas as pd


def optimize_price(model, current_row: pd.Series, objective: str = "revenue", cost_pct: float | None = None,
                   max_change: float = 0.20, grid_size: int = 21, max_demand_loss: float = 0.15) -> pd.DataFrame:
    current_price = float(current_row.checkout_price)
    prices = np.linspace(current_price * (1 - max_change), current_price * (1 + max_change), grid_size).round(2)
    scenario = pd.DataFrame([current_row] * len(prices)).reset_index(drop=True)
    scenario["checkout_price"] = prices
    scenario["discount_pct"] = ((scenario.base_price - prices) / scenario.base_price).clip(lower=0)
    scenario["price_vs_base"] = prices / scenario.base_price
    scenario["predicted_demand"] = model.predict(scenario)
    baseline = float(scenario.loc[np.abs(prices - current_price).argmin(), "predicted_demand"])
    scenario["expected_revenue"] = prices * scenario.predicted_demand
    if cost_pct is not None:
        scenario["expected_profit"] = (prices * (1 - cost_pct)) * scenario.predicted_demand
        score = scenario.expected_profit if objective == "profit" else scenario.expected_revenue
    else:
        score = scenario.expected_revenue
    scenario["demand_loss_vs_current"] = 1 - scenario.predicted_demand / max(baseline, 1e-9)
    safe = scenario.demand_loss_vs_current <= max_demand_loss
    scenario["is_recommended"] = False
    scenario.loc[score[safe].idxmax() if safe.any() else score.idxmax(), "is_recommended"] = True
    return scenario
