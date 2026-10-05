"""Daily risk dashboard.

    streamlit run app/dashboard.py

Every number comes from quant_risk.report; this file only lays it out.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_risk.paper import VAR_DIR, Journal
from quant_risk.report import build_report, strategy_comparison, var_backtest_report
from quant_risk.strategies import STRATEGIES

st.set_page_config(page_title="Quant Risk Dashboard", page_icon="📉", layout="wide")

ZONE_COLOURS = {"green": "#2e7d32", "yellow": "#f9a825", "red": "#c62828"}


def usd(value: float) -> str:
    return "n/a" if pd.isna(value) else f"${value:,.0f}"


def pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.1%}"


@st.cache_data(show_spinner="Building the risk report...")
def cached_report(strategy: str, capital: float):
    return build_report(strategy, capital)


@st.cache_data(show_spinner="Backtesting the VaR models...")
def cached_var_backtest(strategy: str, capital: float):
    return var_backtest_report(strategy, capital)


@st.cache_data(show_spinner="Running every strategy...")
def cached_comparison(capital: float):
    return strategy_comparison(capital)


with st.sidebar:
    st.title("Quant risk")
    strategy = st.selectbox("Strategy", list(STRATEGIES), index=list(STRATEGIES).index("time_series_momentum"))
    capital = st.number_input("Capital (USD)", value=10_000_000, step=1_000_000, min_value=1_000_000)
    st.caption("Limits: config/limits.toml · Scenarios: config/scenarios.toml")

report = cached_report(strategy, float(capital))
head = report.headline

st.header(f"Daily risk report · {report.as_of:%d %b %Y}")
st.caption(f"{strategy.replace('_', ' ')} · capital {usd(capital)} · 1-day horizon, USD")

cols = st.columns(5)
cols[0].metric("Gross exposure", f"{head['gross exposure'] / capital:.2f}x", usd(head["gross exposure"]), delta_color="off")
cols[1].metric("VaR 99%", usd(head["VaR 99% (1d)"]), pct(head["VaR 99% (1d)"] / capital) + " of capital", delta_color="off")
cols[2].metric("ES 97.5%", usd(head["ES 97.5% (1d)"]), pct(head["ES 97.5% (1d)"] / capital) + " of capital", delta_color="off")
cols[3].metric("Worst stress", usd(head["worst stress P&L"]), head["worst stress"], delta_color="off")
cols[4].metric("Highest limit use", pct(head["highest limit use %"]), head["highest limit use"], delta_color="off")

tab_book, tab_var, tab_backtest, tab_stress, tab_perf, tab_paper = st.tabs(
    ["Book & limits", "VaR & ES", "VaR backtest", "Stress & climate", "Performance", "Paper trading"]
)

with tab_book:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Limit utilisation")
        limits = report.limits.reset_index()
        limits["colour"] = pd.cut(limits["utilisation"], [-1, 0.75, 0.9, 1e9], labels=["ok", "watch", "breach"])
        fig = px.bar(limits, x="utilisation", y="limit", orientation="h", color="colour",
                     color_discrete_map={"ok": "#2e7d32", "watch": "#f9a825", "breach": "#c62828"},
                     range_x=[0, max(1.1, limits["utilisation"].max() * 1.1)])
        fig.add_vline(x=1.0, line_dash="dash", line_color="#c62828")
        fig.update_layout(height=320, showlegend=False, xaxis_tickformat=".0%", yaxis_title=None, margin={"t": 10})
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Exposure by sector")
        exposure = report.exposure_by_sector.reset_index()
        fig = go.Figure([
            go.Bar(name="long", y=exposure["sector"], x=exposure["long"], orientation="h", marker_color="#1565c0"),
            go.Bar(name="short", y=exposure["sector"], x=exposure["short"], orientation="h", marker_color="#ef6c00"),
        ])
        fig.update_layout(barmode="relative", height=320, margin={"t": 10}, xaxis_title="USD notional")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Positions")
    st.dataframe(
        report.positions.style.format({"price": "{:,.4g}", "notional": "${:,.0f}", "% of capital": "{:.1%}",
                                       "contracts": "{:,.0f}"}),
        width="stretch",
    )

with tab_var:
    st.subheader("VaR and Expected Shortfall by method")
    st.dataframe(report.var.style.format("${:,.0f}"), width="stretch")
    st.caption(
        "Historical: last 500 days replayed on today's book. Filtered historical: the same days rescaled to "
        "today's volatility. Parametric: normal with an EWMA (lambda 0.94) covariance. Monte Carlo: same "
        "covariance with Student-t (5 dof) tails."
    )
    st.subheader("What drives VaR (Euler contributions, 99% parametric)")
    contrib = report.contributions.reset_index().rename(columns={"index": "symbol"})
    contrib = contrib[contrib["contracts"] != 0]
    fig = px.bar(contrib, x="component VaR", y="symbol", color="sector", orientation="h",
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(height=max(300, 22 * len(contrib)), margin={"t": 10}, xaxis_title="USD",
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    st.caption("A negative bar is a hedge: that position lowers the portfolio's VaR.")

with tab_backtest:
    summary, series = cached_var_backtest(strategy, float(capital))
    st.subheader("Did losses exceed 99% VaR as often as promised?")
    shown = summary[["exceptions", "expected", "exception rate", "Kupiec p-value", "independence p-value",
                     "last 250d exceptions", "Basel zone", "verdict"]]
    st.dataframe(
        shown.style.format({"expected": "{:.0f}", "exception rate": "{:.2%}", "Kupiec p-value": "{:.3f}",
                            "independence p-value": "{:.3f}"}),
        width="stretch",
    )
    st.caption(f"{int(summary['days'].iloc[0]):,} trading days since 2004. Kupiec tests the exception count, "
               "Christoffersen tests whether exceptions cluster, Basel zone counts the last 250 days.")
    method = st.radio("Model", list(summary.index), horizontal=True, index=1)
    data = series[["P&L", f"{method} VaR", f"{method} zone"]].dropna(subset=[f"{method} VaR"])
    breaches = data[data["P&L"] < -data[f"{method} VaR"]]
    fig = go.Figure([
        go.Scatter(x=data.index, y=data["P&L"], mode="lines", name="daily P&L", line={"width": 0.6, "color": "#90a4ae"}),
        go.Scatter(x=data.index, y=-data[f"{method} VaR"], mode="lines", name="-VaR 99%", line={"color": "#1565c0"}),
        go.Scatter(x=breaches.index, y=breaches["P&L"], mode="markers", name="exception",
                   marker={"color": "#c62828", "size": 5}),
    ])
    fig.update_layout(height=380, margin={"t": 10}, yaxis_title="USD")
    st.plotly_chart(fig, width="stretch")
    zones = data[f"{method} zone"].dropna()
    fig = px.scatter(x=zones.index, y=[1] * len(zones), color=zones, color_discrete_map=ZONE_COLOURS)
    fig.update_traces(marker={"symbol": "square", "size": 6})
    fig.update_layout(height=120, margin={"t": 0, "b": 0}, yaxis_visible=False, legend_title="Basel zone",
                      xaxis_title=None)
    st.plotly_chart(fig, width="stretch")

with tab_stress:
    stress = report.stress.reset_index()
    st.subheader("Scenario P&L on today's book")
    fig = px.bar(stress.sort_values("P&L"), x="P&L", y="scenario", color="category", orientation="h",
                 hover_data=["biggest losses"])
    fig.update_layout(height=520, margin={"t": 10}, xaxis_title="USD")
    st.plotly_chart(fig, width="stretch")
    category = st.radio("Show", ["climate", "historical", "hypothetical"], horizontal=True)
    for _, row in stress[stress["category"] == category].iterrows():
        with st.expander(f"{row['scenario']}: {usd(row['P&L'])} ({pct(row['% of capital'])})"):
            st.write(row["description"])
            st.caption(f"Biggest losses: {row['biggest losses']}")
    left, right = st.columns(2)
    with left:
        st.subheader("Reverse stress: worst historical windows")
        st.dataframe(report.reverse_stress.style.format({"worst P&L": "${:,.0f}"}), width="stretch")
    with right:
        st.subheader("Sector shock grid (USD)")
        grid = report.sector_grid
        fig = px.imshow(grid, text_auto=",.0f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                        aspect="auto")
        fig.update_layout(height=320, margin={"t": 10}, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

with tab_perf:
    perf = report.performance
    cols = st.columns(5)
    cols[0].metric("Annual return", pct(perf["annual_return"]))
    cols[1].metric("Volatility", pct(perf["annual_volatility"]))
    cols[2].metric("Sharpe", f"{perf['sharpe']:.2f}")
    cols[3].metric("Max drawdown", pct(perf["max_drawdown"]))
    cols[4].metric("Skew", f"{perf['skew']:.2f}")
    fig = px.line(report.equity, labels={"value": "equity (USD)", "date": ""})
    fig.update_layout(height=300, showlegend=False, margin={"t": 10})
    st.plotly_chart(fig, width="stretch")
    fig = px.area(report.drawdown, labels={"value": "drawdown", "date": "", "index": ""},
                  color_discrete_sequence=["#c62828"])
    fig.update_layout(height=200, showlegend=False, margin={"t": 10}, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    if st.toggle("Compare every strategy"):
        st.dataframe(
            cached_comparison(float(capital)).style.format("{:.2f}").format(
                "{:.1%}", subset=["annual_return", "annual_volatility", "max_drawdown", "worst_day", "best_day"]
            ),
            width="stretch",
        )

with tab_paper:
    path = VAR_DIR / "journal.sqlite"
    if not path.exists():
        st.info("No paper-trading journal yet. Run: python scripts/paper_cycle.py --replay 2025-03-03 2025-05-30 --reset")
    else:
        journal = Journal(path)
        cycles = journal.read("cycles")
        journal.close()
        if cycles.empty:
            st.info("The journal is empty.")
        else:
            for name, group in cycles.groupby("strategy"):
                st.subheader(name.replace("_", " "))
                group = group.set_index(pd.to_datetime(group["date"]).dt.date)
                fig = px.line(group, y="equity", labels={"date": "", "equity": "equity (USD)"})
                halted = group[group["halted"] == 1]
                fig.add_scatter(x=halted.index, y=halted["equity"], mode="markers", name="halted",
                                marker={"color": "#c62828", "size": 9})
                fig.update_layout(height=280, margin={"t": 10})
                st.plotly_chart(fig, width="stretch")
                table = group[["status", "daily_pnl", "drawdown", "var_99", "orders", "note"]].iloc[::-1]
                table.columns = ["status", "daily P&L", "drawdown", "VaR 99%", "orders", "risk engine notes"]
                st.dataframe(table.style.format({"daily P&L": "{:+,.0f}", "drawdown": "{:.1%}", "VaR 99%": "${:,.0f}"}),
                             width="stretch")
