"""Shared data-loading, caching, and business-logic helpers for the
PriceLens AI Streamlit application.

Design principle: this module never invents numbers. Every figure shown in
the app is either read directly from the existing pipeline outputs
(``outputs/recommendations/*.csv``) or derived from them / from the trained
``regularized_log_demand`` model using plain, auditable arithmetic that is
documented inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .demand_model import DemandModel, train_demand_model
from .elasticity import estimate_elasticity  # noqa: F401  (re-exported for reuse)
from .feature_engineering import make_model_frame
from .optimizer import optimize_price

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "recommendations"
PROCESSED_PATH = DATA_DIR / "processed" / "transactions_clean.csv"
REC_PATH = OUTPUT_DIR / "pricing_recommendations.csv"
ELASTICITY_PATH = OUTPUT_DIR / "meal_elasticity.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"

DEFAULT_MAX_PRICE_CHANGE = 0.20
DEFAULT_MAX_DEMAND_LOSS = 0.15


# --------------------------------------------------------------------------- #
# File presence / validation
# --------------------------------------------------------------------------- #
@dataclass
class DataStatus:
    ok: bool
    missing: list[str]
    warnings: list[str]


def check_required_files() -> DataStatus:
    """Confirm the files the whole app depends on are present before we touch them."""
    required = {
        "Processed transactions": PROCESSED_PATH,
        "Pricing recommendations": REC_PATH,
        "Meal elasticity": ELASTICITY_PATH,
        "Model metrics": METRICS_PATH,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return DataStatus(ok=not missing, missing=missing, warnings=[])


# --------------------------------------------------------------------------- #
# Cached data loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH)
    expected_numeric = ["checkout_price", "base_price", "num_orders", "revenue", "discount_pct"]
    missing = [c for c in expected_numeric if c not in df.columns]
    if missing:
        raise ValueError(f"Processed transactions file is missing columns: {missing}")
    return df


@st.cache_data(show_spinner=False)
def load_recommendations() -> pd.DataFrame:
    df = pd.read_csv(REC_PATH)
    required = {
        "center_id", "meal_id", "current_price", "recommended_price",
        "price_change_pct", "expected_demand", "expected_revenue", "demand_loss_vs_current",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Recommendations file is missing columns: {sorted(missing)}")

    # Derive the modeled *current-price* revenue from the demand-loss field that the
    # optimizer already computed (demand_loss_vs_current = 1 - predicted/baseline),
    # so "current" and "expected" revenue sit on the same modeled basis. This is pure
    # algebra on existing columns -- no new data is fabricated.
    safe_loss = df["demand_loss_vs_current"].clip(upper=0.999)
    baseline_demand = df["expected_demand"] / (1 - safe_loss)
    df["baseline_current_demand"] = baseline_demand
    df["baseline_current_revenue"] = df["current_price"] * baseline_demand

    txn = load_transactions()
    meal_lookup = txn[["meal_id", "category", "cuisine"]].drop_duplicates("meal_id")
    df = df.merge(meal_lookup, on="meal_id", how="left")
    return df


@st.cache_data(show_spinner=False)
def load_elasticity() -> pd.DataFrame:
    df = pd.read_csv(ELASTICITY_PATH)
    required = {"meal_id", "observations", "price_points", "elasticity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Elasticity file is missing columns: {sorted(missing)}")
    df["elasticity_band_5"] = df["elasticity"].apply(classify_elasticity_band)
    txn = load_transactions()
    meal_lookup = txn[["meal_id", "category", "cuisine"]].drop_duplicates("meal_id")
    df = df.merge(meal_lookup, on="meal_id", how="left")
    return df


@st.cache_data(show_spinner=False)
def load_model_metrics() -> dict:
    df = pd.read_csv(METRICS_PATH)
    if df.empty:
        raise ValueError("model_metrics.csv is empty.")
    return df.iloc[0].to_dict()


def classify_elasticity_band(e: float) -> str:
    """Five-band, beginner-friendly elasticity classification.

    Bands are defined on the magnitude of the (expected-negative) price
    elasticity of demand. Positive values are flagged separately since a
    positive price-demand association is a data anomaly worth investigating,
    not a normal elastic/inelastic response.
    """
    if pd.isna(e):
        return "Not available"
    if e > 0:
        return "Positive / Investigate"
    a = abs(e)
    if a >= 2.0:
        return "Highly Elastic"
    if a >= 1.1:
        return "Elastic"
    if a >= 0.9:
        return "Approximately Unit Elastic"
    if a >= 0.3:
        return "Inelastic"
    return "Highly Inelastic"


# --------------------------------------------------------------------------- #
# Model: retrain the SAME dependency-free regularized_log_demand model that
# produced outputs/recommendations/*.csv, so the live simulator and the
# static recommendation tables always agree. This trains deterministically
# (closed-form ridge solve) in a couple of seconds and is cached for the
# life of the app session.
#
# NOTE ON model/model.pkl: that artifact is a separate exploratory XGBoost
# model (one-hot encoded centers/meals) from an earlier notebook iteration.
# Its feature schema does not match the pipeline that generated the shipped
# recommendation/metrics CSVs (regularized_log_demand), and using it here
# would silently produce numbers inconsistent with the rest of the app. To
# avoid running two disagreeing pricing algorithms side by side, the live
# app retrains and reuses only the documented regularized_log_demand model.
# The legacy file is kept in the repo for provenance and is described on the
# Methodology page.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_trained_model() -> tuple[DemandModel, pd.DataFrame]:
    txn = load_transactions()
    frame = make_model_frame(txn)
    model = train_demand_model(frame)
    return model, frame


@st.cache_data(show_spinner=False)
def get_latest_rows() -> pd.DataFrame:
    """One row per (center, meal): the most recent observed week, feature-ready."""
    _, frame = get_trained_model()
    return frame.sort_values("week").groupby(["center_id", "meal_id"], as_index=False).tail(1)


def simulate_price_grid(model: DemandModel, row: pd.Series, prices: np.ndarray) -> pd.DataFrame:
    """Predict demand/revenue for an arbitrary set of candidate prices around one
    center-meal's latest known record, using exactly the same scenario
    construction as ``src.optimizer.optimize_price`` (same feature transforms,
    same trained model). This lets the simulator move a continuous slider
    without retraining or introducing a second pricing formula.
    """
    scenario = pd.DataFrame([row] * len(prices)).reset_index(drop=True)
    scenario["checkout_price"] = prices
    scenario["discount_pct"] = ((scenario.base_price - prices) / scenario.base_price).clip(lower=0)
    scenario["price_vs_base"] = prices / scenario.base_price
    scenario["predicted_demand"] = model.predict(scenario)
    scenario["expected_revenue"] = prices * scenario["predicted_demand"]
    return scenario


def guardrail_status(price_change_pct: float, demand_loss_pct: float,
                      max_price_change: float, max_demand_loss: float,
                      caution_ratio: float = 0.85) -> tuple[str, str]:
    """Return (status, css_class) among PASS / CAUTION / BLOCKED.

    BLOCKED  -- exceeds either configured guardrail.
    CAUTION  -- within limits but past `caution_ratio` of either budget.
    PASS     -- comfortably within both guardrails.
    """
    change_ratio = abs(price_change_pct) / max_price_change if max_price_change else 0
    loss_ratio = demand_loss_pct / max_demand_loss if max_demand_loss else 0

    if change_ratio > 1 or loss_ratio > 1:
        return "BLOCKED", "status-blocked"
    if change_ratio >= caution_ratio or loss_ratio >= caution_ratio:
        return "CAUTION", "status-caution"
    return "PASS", "status-pass"


def apply_guardrails(df: pd.DataFrame, max_price_change: float, max_demand_loss: float) -> pd.DataFrame:
    df = df.copy()
    statuses = df.apply(
        lambda r: guardrail_status(r["price_change_pct"], r["demand_loss_vs_current"],
                                    max_price_change, max_demand_loss)[0],
        axis=1,
    )
    df["guardrail_status"] = statuses
    return df


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_currency(value: float, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not available"
    return f"\u20b9{value:,.{decimals}f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not available"
    return f"{value * 100:,.{decimals}f}%"


def fmt_number(value: float, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not available"
    return f"{value:,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Business insights (auto-generated from the loaded data only)
# --------------------------------------------------------------------------- #
def generate_business_insights(rec: pd.DataFrame, elas: pd.DataFrame, top_n: int = 10) -> list[dict]:
    insights = []

    top_rev = rec.sort_values("expected_revenue", ascending=False).iloc[0]
    insights.append({
        "title": "Highest revenue opportunity",
        "insight": f"Center {int(top_rev.center_id)}, Meal {int(top_rev.meal_id)} shows the highest modeled "
                    f"expected revenue at {fmt_currency(top_rev.recommended_price)} "
                    f"(current {fmt_currency(top_rev.current_price)}).",
        "impact": f"Modeled expected revenue of {fmt_currency(top_rev.expected_revenue)} at the recommended price.",
        "action": "Prioritize this center-meal pair for a controlled price test.",
    })

    if not elas.empty:
        elastic = elas[elas.elasticity < 0].sort_values("elasticity").iloc[0]
        insights.append({
            "title": "Most price-sensitive meal",
            "insight": f"Meal {int(elastic.meal_id)} has the strongest historical negative association between "
                        f"price and demand (elasticity {elastic.elasticity:.2f}, {elastic.elasticity_band_5}).",
            "impact": "Price increases on this meal historically coincide with larger demand pull-back.",
            "action": "Favor smaller, closely monitored price steps for this meal.",
        })

    by_center = rec.groupby("center_id")["expected_revenue"].sum().sort_values(ascending=False)
    if not by_center.empty:
        top_center = by_center.index[0]
        insights.append({
            "title": "Center with strongest revenue opportunity",
            "insight": f"Center {int(top_center)} has the highest total modeled expected revenue across all "
                        f"its recommended prices ({fmt_currency(by_center.iloc[0])}).",
            "impact": "Concentrating pilot testing here may surface the largest measurable business impact.",
            "action": "Run the first controlled A/B price test at this center.",
        })

    top_increase = rec.sort_values("price_change_pct", ascending=False).iloc[0]
    insights.append({
        "title": "Largest recommended price increase",
        "insight": f"Center {int(top_increase.center_id)}, Meal {int(top_increase.meal_id)} has the largest "
                    f"recommended increase at {fmt_pct(top_increase.price_change_pct)}.",
        "impact": f"Modeled demand loss for this scenario is {fmt_pct(top_increase.demand_loss_vs_current)}.",
        "action": "Validate with a small-scale test before wider rollout.",
    })

    near_guardrail = rec.assign(gap=(0.15 - rec.demand_loss_vs_current).abs()).sort_values("gap").iloc[0]
    insights.append({
        "title": "Recommendation closest to the demand guardrail",
        "insight": f"Center {int(near_guardrail.center_id)}, Meal {int(near_guardrail.meal_id)} sits closest to "
                    f"the modeled demand-loss guardrail ({fmt_pct(near_guardrail.demand_loss_vs_current)} loss).",
        "impact": "Little headroom remains before this scenario would be blocked under current guardrails.",
        "action": "Monitor closely if tested; consider a smaller price step.",
    })

    top10 = rec.sort_values("expected_revenue", ascending=False).head(top_n)
    insights.append({
        "title": f"Top {top_n} pricing opportunities",
        "insight": "See the table below for the ten center-meal recommendations with the highest modeled "
                    "expected revenue.",
        "impact": "These represent the largest modeled upside currently identified.",
        "action": "Sequence controlled tests starting from the top of this list.",
        "table": top10,
    })

    return insights
