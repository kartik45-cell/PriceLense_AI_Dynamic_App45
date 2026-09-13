"""PriceLens AI - AI-Powered Dynamic Pricing & Revenue Optimization.

Streamlit entry point. Reads the existing dynamic-pricing pipeline outputs
(model, recommendations, elasticity, metrics) and presents them as a
polished, judge-ready decision-support product. See src/dashboard_utils.py
for the data/model layer and src/visualization.py for chart builders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import dashboard_utils as du
from src import visualization as viz
from src.optimizer import optimize_price

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PriceLens AI",
    page_icon="\U0001F4B0",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def load_css() -> None:
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    tone_class = f" kpi-{tone}" if tone else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card{tone_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """


def kpi_row(cards: list[str], cols: int = 4) -> None:
    columns = st.columns(cols)
    for i, card in enumerate(cards):
        with columns[i % cols]:
            st.markdown(card, unsafe_allow_html=True)


def status_badge(status: str) -> str:
    cls = {"PASS": "badge-pass", "CAUTION": "badge-caution", "BLOCKED": "badge-blocked"}.get(status, "badge-pass")
    return f'<span class="status-badge {cls}">{status}</span>'


def section_title(title: str, subtitle: str = "") -> None:
    sub_html = f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="section-title">{title}</div>{sub_html}', unsafe_allow_html=True)


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="disclaimer-box">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Startup: validate required files before doing anything else
# --------------------------------------------------------------------------- #
def guard_startup() -> bool:
    status = du.check_required_files()
    if not status.ok:
        st.error(
            "**Required project files not found.**\n\n"
            "The following files are missing:\n\n"
            + "\n".join(f"- {m}" for m in status.missing)
            + "\n\nPlease run `run_pipeline.py` first to generate the pipeline outputs, "
              "then reload this application."
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
PAGES = [
    ("Overview", "\U0001F4CA"),
    ("AI Recommendations", "\U0001F3AF"),
    ("Price Simulator", "\U0001F39B\uFE0F"),
    ("Demand & Revenue Analytics", "\U0001F4C8"),
    ("Price Elasticity", "\U0001F4C9"),
    ("Model Performance", "\U0001F9EA"),
    ("Business Insights", "\U0001F4A1"),
    ("Methodology / About", "\U0001F4D8"),
]


def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-title">PriceLens AI</span>'
            '<div class="brand-sub">AI Dynamic Pricing</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Navigation")
        labels = [f"{icon}  {name}" for name, icon in PAGES]
        choice = st.radio("Navigation", labels, label_visibility="collapsed")
        page = choice.split("  ", 1)[1]

        st.markdown("---")
        st.markdown("#### Business Guardrails")
        max_price_change = st.slider("Max price change", 0.05, 0.50,
                                      du.DEFAULT_MAX_PRICE_CHANGE, 0.01, format="%.0f%%",
                                      key="max_price_change")
        max_demand_loss = st.slider("Max modeled demand loss", 0.05, 0.50,
                                     du.DEFAULT_MAX_DEMAND_LOSS, 0.01, format="%.0f%%",
                                     key="max_demand_loss")

        st.markdown("---")
        st.markdown("#### System Status")
        status = du.check_required_files()
        checks = [
            ("Model trained", True),
            ("Recommendations loaded", "Pricing recommendations" not in status.missing),
            ("Data loaded", "Processed transactions" not in status.missing),
            ("Guardrails active", True),
        ]
        for label, ok in checks:
            icon = "\u2705" if ok else "\u274C"
            st.markdown(f'<div class="status-line">{icon} {label}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Decision-support tool. Recommendations require human approval before deployment.")
    return page


# --------------------------------------------------------------------------- #
# PAGE 1 - EXECUTIVE OVERVIEW
# --------------------------------------------------------------------------- #
def page_overview(rec: pd.DataFrame, metrics: dict, max_price_change: float, max_demand_loss: float) -> None:
    st.markdown('<div class="hero-title">AI Dynamic Pricing Command Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Turn historical demand signals into actionable pricing decisions.</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(
                '<div class="story-box">'
                '<b>The problem:</b> businesses frequently change prices or offer discounts without knowing '
                'whether the change generated additional demand or simply reduced revenue on sales that would '
                'have happened anyway.<br><br>'
                '<b>What this system does:</b> it analyzes historical transaction behavior, predicts demand '
                'under different price scenarios, and recommends a price that maximizes modeled expected '
                'revenue while respecting business guardrails.<br><br>'
                '<b>The question it answers:</b> "What price should I test next?"'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="story-box">'
                f'<b>Model:</b> {metrics["model"]}<br>'
                f'<b>Validation:</b> Time-based holdout (train through week {int(metrics["train_through_week"])}, '
                f'{int(metrics["test_weeks"])}-week test)<br>'
                f'<b>Holdout R\u00b2:</b> {float(metrics["R2"]):.3f}<br>'
                f'<b>Recommendations generated:</b> {len(rec):,}<br><br>'
                f'<span class="muted">Historical data is observational \u2014 associations, not causal proof. '
                f'Recommendations are decision support, not automatic price changes.</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    current_revenue = rec["baseline_current_revenue"].sum()
    expected_revenue = rec["expected_revenue"].sum()
    uplift = expected_revenue - current_revenue
    uplift_pct = uplift / current_revenue if current_revenue else np.nan

    section_title("Key Performance Indicators")
    kpi_row([
        kpi_card("Current Revenue", du.fmt_currency(current_revenue),
                 "Modeled at current price"),
        kpi_card("Expected Revenue", du.fmt_currency(expected_revenue),
                 "Modeled at recommended price", tone="accent"),
        kpi_card("Revenue Uplift", f"{du.fmt_currency(uplift)} ({du.fmt_pct(uplift_pct)})",
                 "Same modeled basis \u2014 apples to apples", tone="accent"),
        kpi_card("Avg. Recommended Price Change", du.fmt_pct(rec["price_change_pct"].mean()),
                 "Across all recommendations"),
    ])
    kpi_row([
        kpi_card("Avg. Expected Demand", du.fmt_number(rec["expected_demand"].mean()),
                 "Units per center-meal"),
        kpi_card("Number of Recommendations", f'{len(rec):,}', "Center-meal combinations covered"),
        kpi_card("Model R\u00b2", f'{float(metrics["R2"]):.3f}',
                 "Share of holdout variance explained"),
        kpi_card("Demand Guardrail", du.fmt_pct(max_demand_loss),
                 "Max modeled demand loss allowed"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Portfolio Visuals", "Every chart below is generated directly from the pipeline outputs.")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(viz.current_vs_recommended_price(rec), width="stretch")
    with r1c2:
        st.plotly_chart(viz.expected_revenue_by_meal(rec), width="stretch")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(viz.price_change_distribution(rec), width="stretch")
    with r2c2:
        st.plotly_chart(viz.demand_vs_revenue_scatter(rec), width="stretch")

    st.plotly_chart(viz.revenue_by_center(rec), width="stretch")


# --------------------------------------------------------------------------- #
# PAGE 2 - AI PRICE RECOMMENDATIONS
# --------------------------------------------------------------------------- #
def page_recommendations(rec: pd.DataFrame, max_price_change: float, max_demand_loss: float) -> None:
    section_title("AI Price Recommendations", "Filter, review, and drill into individual pricing decisions.")

    rec = du.apply_guardrails(rec, max_price_change, max_demand_loss)

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            centers = st.multiselect("Center", sorted(rec.center_id.unique()))
        with f2:
            meals = st.multiselect("Meal ID", sorted(rec.meal_id.unique()))
        with f3:
            rec_type = st.multiselect("Recommendation Status", ["PASS", "CAUTION", "BLOCKED"])
        with f4:
            price_range = st.slider("Current price range", float(rec.current_price.min()),
                                     float(rec.current_price.max()),
                                     (float(rec.current_price.min()), float(rec.current_price.max())))
        f5, f6, f7 = st.columns(3)
        with f5:
            change_range = st.slider("Price change range", float(rec.price_change_pct.min() * 100),
                                      float(rec.price_change_pct.max() * 100),
                                      (float(rec.price_change_pct.min() * 100), float(rec.price_change_pct.max() * 100)),
                                      format="%.0f%%")
        with f6:
            revenue_range = st.slider("Expected revenue range", float(rec.expected_revenue.min()),
                                       float(rec.expected_revenue.max()),
                                       (float(rec.expected_revenue.min()), float(rec.expected_revenue.max())))
        with f7:
            loss_range = st.slider("Demand loss range", float(rec.demand_loss_vs_current.min() * 100),
                                    float(rec.demand_loss_vs_current.max() * 100),
                                    (float(rec.demand_loss_vs_current.min() * 100),
                                     float(rec.demand_loss_vs_current.max() * 100)), format="%.0f%%")

    filtered = rec.copy()
    if centers:
        filtered = filtered[filtered.center_id.isin(centers)]
    if meals:
        filtered = filtered[filtered.meal_id.isin(meals)]
    if rec_type:
        filtered = filtered[filtered.guardrail_status.isin(rec_type)]
    filtered = filtered[filtered.current_price.between(*price_range)]
    filtered = filtered[(filtered.price_change_pct * 100).between(*change_range)]
    filtered = filtered[filtered.expected_revenue.between(*revenue_range)]
    filtered = filtered[(filtered.demand_loss_vs_current * 100).between(*loss_range)]

    st.caption(f"Showing {len(filtered):,} of {len(rec):,} recommendations")

    if filtered.empty:
        st.warning("No recommendations match the current filters. Try widening the ranges above.")
        return

    display = filtered.sort_values("expected_revenue", ascending=False).copy()
    display_view = pd.DataFrame({
        "Center": display.center_id,
        "Meal ID": display.meal_id,
        "Current Price": display.current_price.map(lambda v: du.fmt_currency(v, 2)),
        "Recommended Price": display.recommended_price.map(lambda v: du.fmt_currency(v, 2)),
        "Price Change %": display.price_change_pct.map(du.fmt_pct),
        "Expected Demand": display.expected_demand.map(lambda v: du.fmt_number(v, 0)),
        "Expected Revenue": display.expected_revenue.map(lambda v: du.fmt_currency(v, 0)),
        "Demand Loss %": display.demand_loss_vs_current.map(du.fmt_pct),
        "Status": display.guardrail_status,
    })

    event = st.dataframe(
        display_view, width="stretch", hide_index=True, height=420,
        on_select="rerun", selection_mode="single-row", key="rec_table",
    )

    selected_idx = None
    if event and event.selection and event.selection.rows:
        selected_idx = display.index[event.selection.rows[0]]

    if selected_idx is None:
        st.info("Select a row above to see the full AI recommendation card and explanation.")
        return

    row = display.loc[selected_idx]
    render_recommendation_card(row, max_price_change, max_demand_loss)


def render_recommendation_card(row: pd.Series, max_price_change: float, max_demand_loss: float) -> None:
    status, _ = du.guardrail_status(row.price_change_pct, row.demand_loss_vs_current,
                                     max_price_change, max_demand_loss)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="rec-card-header">AI RECOMMENDATION \u2014 Center {int(row.center_id)}, '
        f'Meal {int(row.meal_id)} {status_badge(status)}</div>',
        unsafe_allow_html=True,
    )
    kpi_row([
        kpi_card("Current Price", du.fmt_currency(row.current_price, 2)),
        kpi_card("Recommended Price", du.fmt_currency(row.recommended_price, 2), tone="accent"),
        kpi_card("Expected Demand", f"{du.fmt_number(row.expected_demand)} units"),
        kpi_card("Expected Revenue", du.fmt_currency(row.expected_revenue), tone="accent"),
    ])
    kpi_row([
        kpi_card("Price Change", du.fmt_pct(row.price_change_pct)),
        kpi_card("Demand Impact", f"-{du.fmt_pct(row.demand_loss_vs_current)}"),
        kpi_card("Guardrail", status, tone="accent" if status == "PASS" else "danger"),
        kpi_card("Category", str(row.get("category", "N/A"))),
    ], cols=4)

    st.markdown('<div class="why-box"><b>Why this price?</b><br>'
                'This scenario provides the highest modeled expected revenue among the tested prices while '
                'remaining within the configured demand-loss and price-change guardrails. This is a modeled '
                'expected outcome, not a guaranteed result \u2014 validate with a controlled test before full '
                'rollout.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# PAGE 3 - PRICE SIMULATOR
# --------------------------------------------------------------------------- #
def page_simulator(max_price_change: float, max_demand_loss: float) -> None:
    section_title("Price Simulator", "Move the proposed price and see the modeled outcome update instantly.")

    model, _ = du.get_trained_model()
    latest = du.get_latest_rows()

    c1, c2 = st.columns(2)
    with c1:
        center = st.selectbox("Center", sorted(latest.center_id.unique()))
    with c2:
        meals_for_center = sorted(latest[latest.center_id == center].meal_id.unique())
        meal = st.selectbox("Meal", meals_for_center)

    row = latest[(latest.center_id == center) & (latest.meal_id == meal)].iloc[0]
    current_price = float(row.checkout_price)

    proposed_price = st.slider(
        "Proposed price (\u20b9)",
        float(current_price * 0.5), float(current_price * 1.5), float(current_price), 0.5,
    )

    grid = np.linspace(current_price * (1 - max_price_change), current_price * (1 + max_price_change), 41)
    grid = np.unique(np.append(grid, [current_price, proposed_price]))
    scenario = du.simulate_price_grid(model, row, grid)

    current_pred = scenario.loc[(scenario.checkout_price - current_price).abs().idxmin()]
    proposed_pred = scenario.loc[(scenario.checkout_price - proposed_price).abs().idxmin()]

    optimizer_result = optimize_price(model, row, max_change=max_price_change, max_demand_loss=max_demand_loss)
    best = optimizer_result[optimizer_result.is_recommended].iloc[0]

    price_change_pct = proposed_price / current_price - 1
    demand_change_pct = proposed_pred.predicted_demand / current_pred.predicted_demand - 1
    demand_loss_pct = -demand_change_pct if demand_change_pct < 0 else 0.0
    revenue_change_pct = proposed_pred.expected_revenue / current_pred.expected_revenue - 1
    status, _ = du.guardrail_status(price_change_pct, demand_loss_pct, max_price_change, max_demand_loss)

    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown('<div class="scenario-label scenario-current">CURRENT SCENARIO</div>', unsafe_allow_html=True)
        kpi_row([
            kpi_card("Price", du.fmt_currency(current_price, 2)),
            kpi_card("Predicted Demand", f"{du.fmt_number(current_pred.predicted_demand)} units"),
        ], cols=2)
        kpi_row([kpi_card("Expected Revenue", du.fmt_currency(current_pred.expected_revenue))], cols=1)
    with sc2:
        st.markdown('<div class="scenario-label scenario-proposed">PROPOSED SCENARIO</div>', unsafe_allow_html=True)
        kpi_row([
            kpi_card("Price", du.fmt_currency(proposed_price, 2), du.fmt_pct(price_change_pct), tone="accent"),
            kpi_card("Predicted Demand", f"{du.fmt_number(proposed_pred.predicted_demand)} units",
                     du.fmt_pct(demand_change_pct)),
        ], cols=2)
        kpi_row([kpi_card("Expected Revenue", du.fmt_currency(proposed_pred.expected_revenue),
                           du.fmt_pct(revenue_change_pct), tone="accent")], cols=1)

    st.markdown(
        f'<div class="guardrail-strip">Guardrail check: {status_badge(status)} &nbsp; '
        f'Price change {du.fmt_pct(price_change_pct)} (limit {du.fmt_pct(max_price_change)}) &nbsp;|&nbsp; '
        f'Demand loss {du.fmt_pct(demand_loss_pct)} (limit {du.fmt_pct(max_demand_loss)})</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="why-box"><b>AI recommended price for this item:</b> '
        f'{du.fmt_currency(best.checkout_price, 2)} \u2014 this scenario produced the highest modeled expected '
        f'revenue among the tested prices while remaining within the \u00b1{du.fmt_pct(max_price_change)} '
        f'price-change and {du.fmt_pct(max_demand_loss)} modeled demand-loss guardrails. This is a modeled '
        f'expected outcome, not a guaranteed one.</div>',
        unsafe_allow_html=True,
    )

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(viz.price_response_chart(scenario, current_price, proposed_price, best.checkout_price,
                                                  "predicted_demand", "Predicted Demand (units)",
                                                  "Price vs. Predicted Demand"), width="stretch")
    with r2:
        st.plotly_chart(viz.price_response_chart(scenario, current_price, proposed_price, best.checkout_price,
                                                  "expected_revenue", "Expected Revenue (\u20b9)",
                                                  "Price vs. Expected Revenue"), width="stretch")

    disclaimer(
        "The optimizer evaluates multiple price scenarios around the current price. For each scenario, the "
        "demand model predicts demand. Expected revenue is calculated as price \u00d7 predicted demand. The "
        "system selects the scenario with the highest modeled expected revenue subject to business guardrails."
    )


# --------------------------------------------------------------------------- #
# PAGE 4 - DEMAND & REVENUE ANALYTICS
# --------------------------------------------------------------------------- #
def page_analytics(txn: pd.DataFrame) -> None:
    section_title("Demand & Revenue Analysis", "Historical trends and relationships from actual transaction data.")

    with st.expander("Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            centers = st.multiselect("Center", sorted(txn.center_id.unique()), key="an_center")
        with f2:
            meals = st.multiselect("Meal ID", sorted(txn.meal_id.unique()), key="an_meal")
        with f3:
            categories = st.multiselect("Category", sorted(txn.category.dropna().unique()), key="an_cat")

    df = txn.copy()
    if centers:
        df = df[df.center_id.isin(centers)]
    if meals:
        df = df[df.meal_id.isin(meals)]
    if categories:
        df = df[df.category.isin(categories)]

    if df.empty:
        st.warning("No transactions match the current filters.")
        return

    weekly = df.groupby("week", as_index=False).agg(
        num_orders=("num_orders", "sum"), revenue=("revenue", "sum"), checkout_price=("checkout_price", "mean"),
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        st.plotly_chart(viz.trend_line(weekly, "week", "num_orders", "Demand Trend Over Time", "Total Orders"),
                         width="stretch")
    with t2:
        st.plotly_chart(viz.trend_line(weekly, "week", "revenue", "Revenue Trend Over Time", "Total Revenue (\u20b9)"),
                         width="stretch")
    with t3:
        st.plotly_chart(viz.trend_line(weekly, "week", "checkout_price", "Price Trend Over Time",
                                        "Avg. Checkout Price (\u20b9)"), width="stretch")

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(viz.relationship_scatter(df, "checkout_price", "num_orders",
                                                  "Price vs. Demand Relationship", "Checkout Price (\u20b9)",
                                                  "Orders"), width="stretch")
    with r2:
        st.plotly_chart(viz.relationship_scatter(df, "checkout_price", "revenue",
                                                  "Price vs. Revenue Relationship", "Checkout Price (\u20b9)",
                                                  "Revenue (\u20b9)"), width="stretch")

    r3, r4 = st.columns(2)
    with r3:
        st.plotly_chart(viz.category_bar(df.groupby("center_id").revenue.sum(), "Center Performance (Revenue)",
                                          "Total Revenue (\u20b9)"), width="stretch")
    with r4:
        st.plotly_chart(viz.category_bar(df.groupby("meal_id").revenue.sum(), "Meal Performance (Revenue)",
                                          "Total Revenue (\u20b9)"), width="stretch")


# --------------------------------------------------------------------------- #
# PAGE 5 - PRICE ELASTICITY
# --------------------------------------------------------------------------- #
def page_elasticity(elas: pd.DataFrame, rec: pd.DataFrame) -> None:
    section_title("Price Elasticity Intelligence")
    st.markdown(
        '<div class="story-box">Price elasticity measures how strongly demand responds to price changes. '
        'If elasticity is strongly negative, customers are more sensitive to price changes. If elasticity is '
        'closer to zero, demand historically changes less with price.</div>',
        unsafe_allow_html=True,
    )

    band_counts = elas.elasticity_band_5.value_counts()
    kpi_row([kpi_card(band, str(count)) for band, count in band_counts.items()], cols=min(5, len(band_counts)))

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(viz.elasticity_distribution(elas), width="stretch")
    with r2:
        st.plotly_chart(viz.elasticity_by_meal(elas), width="stretch")

    meal_change = rec.groupby("meal_id").agg(avg_price_change_pct=("price_change_pct", "mean"),
                                              total_expected_revenue=("expected_revenue", "sum")).reset_index()
    merged = elas.merge(meal_change, on="meal_id", how="inner")

    r3, r4 = st.columns(2)
    with r3:
        st.plotly_chart(viz.elasticity_vs_price_change(merged), width="stretch")
    with r4:
        st.plotly_chart(viz.elasticity_vs_revenue(merged), width="stretch")

    disclaimer("These are historical descriptive associations, not causal guarantees.")

    with st.expander("Full elasticity table"):
        table = elas[["meal_id", "category", "cuisine", "observations", "price_points",
                      "elasticity", "elasticity_band_5"]].sort_values("elasticity")
        st.dataframe(table, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# PAGE 6 - MODEL PERFORMANCE
# --------------------------------------------------------------------------- #
def page_model_performance(metrics: dict) -> None:
    section_title("Model Performance", "How the demand model was validated.")

    kpi_row([
        kpi_card("Model", str(metrics["model"])),
        kpi_card("MAE", f'{float(metrics["MAE"]):.2f}', "Mean Absolute Error (orders)"),
        kpi_card("RMSE", f'{float(metrics["RMSE"]):.2f}', "Root Mean Squared Error (orders)"),
        kpi_card("R\u00b2", f'{float(metrics["R2"]):.3f}', "Share of variance explained"),
    ])
    kpi_row([
        kpi_card("Validation Method", "Time-based holdout"),
        kpi_card("Training Through Week", str(int(metrics["train_through_week"]))),
        kpi_card("Test Horizon", f'{int(metrics["test_weeks"])} weeks'),
        kpi_card("R\u00b2 vs. Accuracy", "Not classification", "See explanation below"),
    ])

    st.markdown(
        f'<div class="story-box"><b>What does R\u00b2 = {float(metrics["R2"]):.3f} mean?</b><br>'
        f'This indicates that the model explains approximately {float(metrics["R2"]) * 100:.1f}% of the '
        f'variation in the modeled target on the time-based holdout. This is <b>not</b> the same as '
        f'"{float(metrics["R2"]) * 100:.1f}% accuracy" \u2014 R\u00b2 is a regression fit statistic, not a '
        f'classification accuracy rate.<br><br>'
        f'<b>Why time-based holdout, not random split?</b> A random split can leak future information into '
        f'training. Training only through week {int(metrics["train_through_week"])} and testing on the next '
        f'{int(metrics["test_weeks"])} weeks mimics how the model would actually be used in production: '
        f'predicting demand it has not seen yet.</div>',
        unsafe_allow_html=True,
    )

    model, frame = du.get_trained_model()
    cutoff = frame.week.max() - int(metrics["test_weeks"])
    test = frame[frame.week > cutoff]
    predicted = model.predict(test)
    actual = test.num_orders.to_numpy()
    residuals = actual - predicted

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(viz.actual_vs_predicted(actual, predicted), width="stretch")
    with r2:
        st.plotly_chart(viz.residual_histogram(residuals), width="stretch")


# --------------------------------------------------------------------------- #
# PAGE 7 - BUSINESS INSIGHTS
# --------------------------------------------------------------------------- #
def page_business_insights(rec: pd.DataFrame, elas: pd.DataFrame) -> None:
    section_title("Business Insights", "Automatically generated from the current recommendation data.")

    insights = du.generate_business_insights(rec, elas)
    for ins in insights:
        with st.container():
            st.markdown(
                f'<div class="insight-card"><div class="insight-title">{ins["title"]}</div>'
                f'<div class="insight-row"><b>Insight:</b> {ins["insight"]}</div>'
                f'<div class="insight-row"><b>Impact:</b> {ins["impact"]}</div>'
                f'<div class="insight-row"><b>Action:</b> {ins["action"]}</div></div>',
                unsafe_allow_html=True,
            )
            if "table" in ins:
                view = pd.DataFrame({
                    "Center": ins["table"].center_id, "Meal ID": ins["table"].meal_id,
                    "Current Price": ins["table"].current_price.map(lambda v: du.fmt_currency(v, 2)),
                    "Recommended Price": ins["table"].recommended_price.map(lambda v: du.fmt_currency(v, 2)),
                    "Price Change %": ins["table"].price_change_pct.map(du.fmt_pct),
                    "Expected Revenue": ins["table"].expected_revenue.map(lambda v: du.fmt_currency(v, 0)),
                })
                st.dataframe(view, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# PAGE 8 - METHODOLOGY / ABOUT
# --------------------------------------------------------------------------- #
def page_methodology(metrics: dict) -> None:
    section_title("Methodology", "How PriceLens AI turns historical transactions into a price recommendation.")

    stages = ["Historical Transactions", "Data Cleaning", "Feature Engineering", "Demand Modeling",
              "Price Elasticity Analysis", "Price Scenario Simulation", "Optimization",
              "Guardrail Filtering", "AI Price Recommendation"]
    flow_html = '<div class="pipeline-flow">'
    for i, stage in enumerate(stages):
        flow_html += f'<div class="pipeline-node">{stage}</div>'
        if i < len(stages) - 1:
            flow_html += '<div class="pipeline-arrow">\u2193</div>'
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="story-box">'
        '<b>1. Data cleaning</b> removes duplicates, invalid prices, and invalid order counts.<br>'
        '<b>2. Feature engineering</b> builds leakage-safe features: lagged and rolling historical demand, '
        'seasonality (week sine/cosine), discount percentage, and promotion flags \u2014 all shifted so no '
        'feature ever contains information from the week being predicted.<br>'
        f'<b>3. Demand modeling</b> fits a regularized log-demand model (<code>{metrics["model"]}</code>), '
        f'validated on a strict time-based holdout (train through week {int(metrics["train_through_week"])}, '
        f'test on the next {int(metrics["test_weeks"])} weeks) to avoid look-ahead bias.<br>'
        '<b>4. Price elasticity analysis</b> estimates a descriptive, per-meal price-demand association from '
        'historical data.<br>'
        '<b>5\u20137. Scenario simulation, optimization, and guardrail filtering</b>: for each center-meal '
        'combination, the system evaluates multiple candidate prices around the current price, predicts demand '
        'for each, computes expected revenue as price \u00d7 predicted demand, and selects the highest-revenue '
        'scenario that stays within the configured maximum price-change and demand-loss guardrails.<br>'
        '<b>8. AI price recommendation</b> is surfaced to the manager as decision support \u2014 never applied '
        'automatically.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="disclaimer-box"><b>Data limitation:</b> the dataset is observational. Historical '
        'associations do not establish that price changes caused changes in demand.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("About model/model.pkl (legacy exploratory artifact)"):
        st.markdown(
            "This repository also contains `model/model.pkl`, an earlier exploratory XGBoost model with "
            "one-hot encoded center/meal features. Its feature schema does not match the "
            f"`{metrics['model']}` pipeline that produced the recommendation and metrics files shipped in "
            "`outputs/recommendations/`. To avoid running two disagreeing pricing algorithms side by side, "
            "this application retrains and uses only the documented, reproducible "
            f"`{metrics['model']}` model \u2014 verified to reproduce the exact same MAE/RMSE/R\u00b2 shown on "
            "the Model Performance page. The legacy file is kept for provenance only."
        )

    section_title("Production Roadmap")
    roadmap = ["A/B testing", "Inventory constraints", "Capacity constraints", "Competitor pricing",
               "Cost/margin integration", "Real-time data", "Human approval workflow", "Monitoring and drift detection"]
    cols = st.columns(4)
    for i, item in enumerate(roadmap):
        with cols[i % 4]:
            st.markdown(f'<div class="roadmap-chip">{item}</div>', unsafe_allow_html=True)

    section_title("Downloads")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Download pricing recommendations", du.REC_PATH.read_bytes(),
                            "pricing_recommendations.csv", "text/csv")
    with d2:
        st.download_button("Download elasticity data", du.ELASTICITY_PATH.read_bytes(),
                            "meal_elasticity.csv", "text/csv")
    with d3:
        st.download_button("Download model metrics", du.METRICS_PATH.read_bytes(),
                            "model_metrics.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    load_css()
    if not guard_startup():
        st.stop()

    page = sidebar_nav()
    max_price_change = st.session_state.get("max_price_change", du.DEFAULT_MAX_PRICE_CHANGE)
    max_demand_loss = st.session_state.get("max_demand_loss", du.DEFAULT_MAX_DEMAND_LOSS)

    try:
        txn = du.load_transactions()
        rec = du.load_recommendations()
        elas = du.load_elasticity()
        metrics = du.load_model_metrics()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load project data: {exc}")
        st.stop()
        return

    if page == "Overview":
        page_overview(rec, metrics, max_price_change, max_demand_loss)
    elif page == "AI Recommendations":
        page_recommendations(rec, max_price_change, max_demand_loss)
    elif page == "Price Simulator":
        page_simulator(max_price_change, max_demand_loss)
    elif page == "Demand & Revenue Analytics":
        page_analytics(txn)
    elif page == "Price Elasticity":
        page_elasticity(elas, rec)
    elif page == "Model Performance":
        page_model_performance(metrics)
    elif page == "Business Insights":
        page_business_insights(rec, elas)
    elif page == "Methodology / About":
        page_methodology(metrics)


if __name__ == "__main__":
    main()
