"""Streamlit dashboard for Liquidity project.

Loads processed data from data/08_reporting and model inputs to render charts.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORT_DIR = DATA_DIR / "08_reporting"
INPUT_DIR = DATA_DIR / "05_model_input"
MODEL_OUT_DIR = DATA_DIR / "07_model_output"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if exists, else empty DataFrame.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame or empty if missing.
    """

    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def section(title: str, description: str):
    """Create a bordered section with a title and description."""
    st.markdown(
        f"""
        <div class="section">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the dashboard."""

    st.set_page_config(page_title="Liquidity Forecasting", layout="wide")

    st.markdown(
        """
        <style>
        .section { border: 1px solid var(--text-color); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
        .css-18e3th9 { padding-top: 2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Liquidity Forecasting Dashboard")
    st.caption("Aggregations and forecasts built with Kedro pipelines")

    # Sidebar filters
    st.sidebar.header("Filters")
    country_daily = load_csv(INPUT_DIR / "country_daily.csv")
    forecast = load_csv(MODEL_OUT_DIR / "country_forecast.csv")
    latest_summary = load_csv(REPORT_DIR / "latest_summary.csv")
    country_totals = load_csv(REPORT_DIR / "country_totals.csv")

    # Ensure types
    for df in (country_daily, forecast):
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    all_countries = sorted(set(country_daily.get("country", pd.Series(dtype=str)).dropna().unique()))
    countries = st.sidebar.multiselect("Country", options=all_countries, default=all_countries)

    if not country_daily.empty and "date" in country_daily.columns:
        min_d = country_daily["date"].min()
        max_d = country_daily["date"].max()
    else:
        min_d = dt.date(2024, 1, 1)
        max_d = dt.date(2024, 1, 31)

    date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, max_d

    # Filter data
    if not country_daily.empty:
        mask = (
            country_daily["country"].isin(countries)
            & (country_daily["date"] >= start_date)
            & (country_daily["date"] <= end_date)
        )
        cd = country_daily.loc[mask].copy()
    else:
        cd = country_daily.copy()

    if not forecast.empty:
        fmask = forecast["country"].isin(countries)
        fc = forecast.loc[fmask].copy()
    else:
        fc = forecast.copy()

    # Section 1: Actual deposits
    section("Actual Deposits", "Daily total deposits by country over time.")
    if not cd.empty:
        chart = (
            alt.Chart(cd)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("total_deposit:Q", title="Total Deposit"),
                color=alt.Color("country:N", title="Country"),
                tooltip=["country", "date", "total_deposit"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No actuals available.")

    # Section 2: Forecast
    section("Forecast", "Country-level deposit forecast based on average interest rates.")
    if not fc.empty:
        fchart = (
            alt.Chart(fc)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("forecast_deposit:Q", title="Forecasted Deposit"),
                color=alt.Color("country:N", title="Country"),
                tooltip=["country", "date", "forecast_deposit"],
            )
            .properties(height=300)
        )
        st.altair_chart(fchart, use_container_width=True)
    else:
        st.info("No forecast available.")

    # Section 3: Summaries
    section("Summary Tables", "Latest snapshots and historical totals by country.")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Latest Snapshot")
        st.dataframe(latest_summary)
    with cols[1]:
        st.subheader("Historical Totals")
        st.dataframe(country_totals)


if __name__ == "__main__":
    main()

