"""Reusable, professionally themed Plotly chart builders for PriceLens AI."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# Palette: sophisticated navy / teal enterprise theme.
NAVY = "#0B1F3A"
NAVY_LIGHT = "#132A4C"
ACCENT = "#2DD4BF"          # teal accent for "recommended / positive"
ACCENT_2 = "#F5A623"        # amber for "current / caution"
DANGER = "#EF4444"
MUTED_TEXT = "#8CA0B8"
GRID = "rgba(140, 160, 184, 0.15)"
FONT = "Inter, -apple-system, Segoe UI, sans-serif"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color="#E2E8F0", size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=NAVY_LIGHT, font_size=12, font_family=FONT),
)


def _style(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    fig.update_layout(**BASE_LAYOUT, title=dict(text=title, x=0.01, xanchor="left",
                                                 font=dict(size=16, color="#F1F5F9")))
    fig.update_xaxes(title=xaxis_title, gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(title=yaxis_title, gridcolor=GRID, zeroline=False, showline=False)
    return fig


def current_vs_recommended_price(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    agg = (df.groupby("meal_id")[["current_price", "recommended_price"]]
             .mean().sort_values("recommended_price", ascending=False).head(top_n).reset_index())
    agg = agg.sort_values("recommended_price")
    fig = go.Figure()
    fig.add_bar(y=agg.meal_id.astype(str), x=agg.current_price, name="Current Price",
                orientation="h", marker_color=ACCENT_2,
                hovertemplate="Meal %{y}<br>Current: \u20b9%{x:,.0f}<extra></extra>")
    fig.add_bar(y=agg.meal_id.astype(str), x=agg.recommended_price, name="Recommended Price",
                orientation="h", marker_color=ACCENT,
                hovertemplate="Meal %{y}<br>Recommended: \u20b9%{x:,.0f}<extra></extra>")
    fig.update_layout(barmode="group")
    return _style(fig, "Current vs. Recommended Price (Top Meals by Recommended Price)",
                  "Price (\u20b9)", "Meal ID")


def expected_revenue_by_meal(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    agg = (df.groupby("meal_id")["expected_revenue"].sum()
             .sort_values(ascending=False).head(top_n).reset_index())
    agg = agg.sort_values("expected_revenue")
    fig = go.Figure(go.Bar(
        y=agg.meal_id.astype(str), x=agg.expected_revenue, orientation="h",
        marker_color=ACCENT,
        hovertemplate="Meal %{y}<br>Expected Revenue: \u20b9%{x:,.0f}<extra></extra>",
    ))
    return _style(fig, f"Expected Revenue by Meal (Top {top_n})", "Expected Revenue (\u20b9)", "Meal ID")


def price_change_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df.price_change_pct * 100, nbinsx=40, marker_color=ACCENT,
        hovertemplate="Price change: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    return _style(fig, "Distribution of Recommended Price Changes", "Price Change (%)", "Number of Recommendations")


def demand_vs_revenue_scatter(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df.expected_demand, y=df.expected_revenue, mode="markers",
        marker=dict(color=df.price_change_pct * 100, colorscale="Tealrose",
                    colorbar=dict(title="Price<br>Change %"), size=7, opacity=0.75,
                    line=dict(width=0)),
        text=[f"Center {c}, Meal {m}" for c, m in zip(df.center_id, df.meal_id)],
        hovertemplate="%{text}<br>Demand: %{x:,.0f}<br>Revenue: \u20b9%{y:,.0f}<extra></extra>",
    ))
    return _style(fig, "Expected Demand vs. Expected Revenue", "Expected Demand (units)", "Expected Revenue (\u20b9)")


def revenue_by_center(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    agg = (df.groupby("center_id")["expected_revenue"].sum()
             .sort_values(ascending=False).head(top_n).reset_index())
    agg = agg.sort_values("expected_revenue")
    fig = go.Figure(go.Bar(
        y=agg.center_id.astype(str), x=agg.expected_revenue, orientation="h",
        marker_color=ACCENT_2,
        hovertemplate="Center %{y}<br>Expected Revenue: \u20b9%{x:,.0f}<extra></extra>",
    ))
    return _style(fig, f"Revenue Opportunity by Center (Top {top_n})", "Expected Revenue (\u20b9)", "Center ID")


def trend_line(df: pd.DataFrame, x: str, y: str, title: str, y_title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines", line=dict(color=ACCENT, width=2.5),
                                fill="tozeroy", fillcolor="rgba(45, 212, 191, 0.08)"))
    return _style(fig, title, "Week", y_title)


def relationship_scatter(df: pd.DataFrame, x: str, y: str, title: str, x_title: str, y_title: str,
                          sample_n: int = 4000) -> go.Figure:
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    fig = go.Figure(go.Scatter(
        x=df[x], y=df[y], mode="markers",
        marker=dict(color=ACCENT, size=5, opacity=0.35, line=dict(width=0)),
    ))
    return _style(fig, title, x_title, y_title)


def category_bar(series: pd.Series, title: str, x_title: str, top_n: int = 15) -> go.Figure:
    agg = series.sort_values(ascending=False).head(top_n).reset_index()
    agg.columns = ["label", "value"]
    agg = agg.sort_values("value")
    fig = go.Figure(go.Bar(
        y=agg.label.astype(str), x=agg.value, orientation="h", marker_color=ACCENT,
        hovertemplate="%{y}<br>%{x:,.0f}<extra></extra>",
    ))
    return _style(fig, title, x_title, "")


def elasticity_distribution(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=df.elasticity, nbinsx=25, marker_color=ACCENT,
        hovertemplate="Elasticity: %{x:.2f}<br>Meals: %{y}<extra></extra>",
    ))
    fig.add_vline(x=-1, line_dash="dash", line_color=MUTED_TEXT,
                  annotation_text="Unit elastic (-1)", annotation_font_color=MUTED_TEXT)
    return _style(fig, "Distribution of Meal-Level Price Elasticity", "Elasticity", "Number of Meals")


def elasticity_by_meal(df: pd.DataFrame) -> go.Figure:
    agg = df.sort_values("elasticity")
    colors = [DANGER if v > 0 else ACCENT for v in agg.elasticity]
    fig = go.Figure(go.Bar(
        y=agg.meal_id.astype(str), x=agg.elasticity, orientation="h", marker_color=colors,
        hovertemplate="Meal %{y}<br>Elasticity: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(height=max(400, 18 * len(agg)))
    return _style(fig, "Price Elasticity by Meal", "Elasticity", "Meal ID")


def elasticity_vs_price_change(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df.elasticity, y=df.avg_price_change_pct * 100, mode="markers",
        marker=dict(color=ACCENT, size=10, opacity=0.8),
        text=df.meal_id.astype(str),
        hovertemplate="Meal %{text}<br>Elasticity: %{x:.2f}<br>Avg Price Change: %{y:.1f}%<extra></extra>",
    ))
    return _style(fig, "Elasticity vs. Recommended Price Change", "Elasticity", "Avg. Recommended Price Change (%)")


def elasticity_vs_revenue(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df.elasticity, y=df.total_expected_revenue, mode="markers",
        marker=dict(color=ACCENT_2, size=10, opacity=0.8),
        text=df.meal_id.astype(str),
        hovertemplate="Meal %{text}<br>Elasticity: %{x:.2f}<br>Expected Revenue: \u20b9%{y:,.0f}<extra></extra>",
    ))
    return _style(fig, "Elasticity vs. Expected Revenue", "Elasticity", "Total Expected Revenue (\u20b9)")


def price_response_chart(scenario: pd.DataFrame, current_price: float, proposed_price: float,
                          optimal_price: float | None, y_col: str, y_title: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scenario.checkout_price, y=scenario[y_col], mode="lines",
                              line=dict(color=ACCENT, width=2.5), name="Modeled response"))
    fig.add_vline(x=current_price, line_dash="dot", line_color=MUTED_TEXT,
                  annotation_text="Current", annotation_font_color=MUTED_TEXT)
    fig.add_vline(x=proposed_price, line_dash="dash", line_color=ACCENT_2,
                  annotation_text="Proposed", annotation_font_color=ACCENT_2)
    if optimal_price is not None:
        fig.add_vline(x=optimal_price, line_dash="dashdot", line_color="#FFFFFF",
                      annotation_text="AI Recommended", annotation_font_color="#FFFFFF")
    return _style(fig, title, "Price (\u20b9)", y_title)


def actual_vs_predicted(actual, predicted) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                              marker=dict(color=ACCENT, size=5, opacity=0.35), name="Holdout weeks"))
    lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                              line=dict(color=MUTED_TEXT, dash="dash"), name="Perfect prediction"))
    return _style(fig, "Actual vs. Predicted Demand (Time-Based Holdout)", "Actual Orders", "Predicted Orders")


def residual_histogram(residuals) -> go.Figure:
    fig = go.Figure(go.Histogram(x=residuals, nbinsx=50, marker_color=ACCENT_2))
    fig.add_vline(x=0, line_dash="dash", line_color=MUTED_TEXT)
    return _style(fig, "Residual Distribution (Actual \u2212 Predicted)", "Residual (Orders)", "Frequency")
