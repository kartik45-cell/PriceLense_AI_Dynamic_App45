"""Descriptive, regularised price elasticity estimates by product."""
import numpy as np
import pandas as pd


def estimate_elasticity(df: pd.DataFrame, min_observations: int = 12) -> pd.DataFrame:
    rows = []
    for meal_id, part in df.groupby("meal_id", observed=True):
        part = part[(part.checkout_price > 0) & (part.num_orders >= 0)]
        if len(part) < min_observations or part.checkout_price.nunique() < 4:
            continue
        X = np.column_stack([
            np.log(part.checkout_price), part.discount_pct,
            part.emailer_for_promotion, part.homepage_featured,
        ])
        y = np.log1p(part.num_orders)
        # Small ridge penalty stabilizes correlated price and discount signals.
        X = np.column_stack([np.ones(len(X)), X])
        scale = X[:, 1:].std(axis=0)
        scale[scale == 0] = 1
        z = X.copy()
        z[:, 1:] = (X[:, 1:] - X[:, 1:].mean(axis=0)) / scale
        beta = np.linalg.solve(z.T @ z + np.diag([0, 1.0, 1.0, 1.0, 1.0]), z.T @ y)
        # Convert the standardised log-price coefficient back to natural units.
        e = float(beta[1] / scale[0])
        rows.append({"meal_id": meal_id, "observations": len(part), "price_points": part.checkout_price.nunique(),
                     "elasticity": e, "elasticity_band": "elastic" if e < -1 else "inelastic" if e < 0 else "positive / investigate"})
    return pd.DataFrame(rows).sort_values("elasticity") if rows else pd.DataFrame()
