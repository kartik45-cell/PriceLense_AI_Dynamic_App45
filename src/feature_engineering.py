"""Leakage-safe features available when a new weekly price is chosen."""
import numpy as np
import pandas as pd


def make_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["center_id", "meal_id"]
    x = df.sort_values(keys + ["week"]).copy()
    x["week_sin"] = np.sin(2 * np.pi * x["week"] / 52)
    x["week_cos"] = np.cos(2 * np.pi * x["week"] / 52)
    grouped = x.groupby(keys, observed=True)["num_orders"]
    x["lag_1_orders"] = grouped.shift(1)
    x["lag_4_orders"] = grouped.shift(4)
    x["rolling_4_orders"] = grouped.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    x["price_vs_base"] = x["checkout_price"] / x["base_price"]
    x["promotion_any"] = ((x["emailer_for_promotion"] == 1) | (x["homepage_featured"] == 1)).astype(int)
    return x.dropna(subset=["lag_1_orders", "lag_4_orders", "rolling_4_orders"]).reset_index(drop=True)


FEATURES = [
    "checkout_price", "base_price", "discount_pct", "price_vs_base",
    "emailer_for_promotion", "homepage_featured", "promotion_any", "week_sin", "week_cos",
    "center_id", "meal_id", "lag_1_orders", "lag_4_orders", "rolling_4_orders",
]
