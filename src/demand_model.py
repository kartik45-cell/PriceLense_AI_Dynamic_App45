"""Time-aware demand model training and evaluation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .feature_engineering import FEATURES


@dataclass
class DemandModel:
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    log_demand_bounds: tuple[float, float]
    features: list[str]
    metrics: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = _design_matrix(frame[self.features])
        Z = (X - self.means) / self.scales
        Z[:, 0] = 1.0
        log_prediction = np.clip(Z @ self.coefficients, *self.log_demand_bounds)
        return np.maximum(0, np.expm1(log_prediction))


def _design_matrix(features: pd.DataFrame) -> np.ndarray:
    """A dependency-free nonlinear ridge design matrix."""
    # Logs make the target relationship more stable; historical-demand terms are
    # deliberately shifted in feature_engineering so they contain no current-week target.
    price = np.log(features["checkout_price"].to_numpy(dtype=float))
    discount = features["discount_pct"].to_numpy(dtype=float)
    return np.column_stack([
        np.ones(len(features)), price, discount,
        features[["emailer_for_promotion", "homepage_featured", "promotion_any", "week_sin", "week_cos", "center_id", "meal_id"]].to_numpy(dtype=float),
        np.log1p(features[["lag_1_orders", "lag_4_orders", "rolling_4_orders"]].to_numpy(dtype=float)),
        price * features["promotion_any"].to_numpy(dtype=float),
    ])


def train_demand_model(frame: pd.DataFrame, test_weeks: int = 12, random_state: int = 42) -> DemandModel:
    cutoff = frame.week.max() - test_weeks
    train, test = frame[frame.week <= cutoff], frame[frame.week > cutoff]
    if train.empty or test.empty:
        raise ValueError("Not enough time periods for a time-based train/test split.")
    X = _design_matrix(train[FEATURES])
    means, scales = X.mean(axis=0), X.std(axis=0)
    scales[scales == 0] = 1
    Z = (X - means) / scales
    Z[:, 0] = 1.0
    penalty = np.eye(Z.shape[1]) * 8.0
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(Z.T @ Z + penalty, Z.T @ np.log1p(train.num_orders.to_numpy()))
    bounds = tuple(np.quantile(np.log1p(train.num_orders.to_numpy()), [0.01, 0.995]))
    candidate = DemandModel(coefficients, means, scales, bounds, FEATURES, {})
    pred = candidate.predict(test)
    err = test.num_orders.to_numpy() - pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denominator = float(np.sum((test.num_orders - test.num_orders.mean()) ** 2))
    r2 = 1 - float(np.sum(err ** 2)) / denominator if denominator else 0.0
    metrics = {"model": "regularized_log_demand", "MAE": round(mae, 2),
               "RMSE": round(rmse, 2), "R2": round(r2, 3),
               "train_through_week": int(cutoff), "test_weeks": int(test_weeks)}
    candidate.metrics = metrics
    return candidate
