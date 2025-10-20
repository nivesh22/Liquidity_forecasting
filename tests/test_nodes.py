"""Unit tests for forecasting pipeline nodes with full coverage.

These tests validate data handling, joins, aggregations, features,
OLS helper, forecasting, and reporting tables.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from liquidity.pipelines.forecasting.nodes import (
    _ols_slope_intercept,
    aggregate_country_daily,
    compute_country_features,
    forecast_deposits,
    link_balances_with_clients,
    make_reporting_tables,
    preprocess_clients,
    preprocess_eod_balances,
)


def test_preprocess_eod_balances_happy_path() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "client_id": ["C1", "C2"],
            "balance": [100.0, 200.0],
            "currency": ["USD", "USD"],
            "country": ["USA", "USA"],
        }
    )
    out = preprocess_eod_balances(raw)
    assert set(out.columns) == {"date", "client_id", "balance", "currency", "country"}
    assert out["balance"].min() >= 0
    assert out["date"].dtype == object  # datetime.date becomes object in pandas


def test_preprocess_eod_balances_missing_columns() -> None:
    raw = pd.DataFrame({"client_id": ["C1"], "balance": [10.0]})
    try:
        preprocess_eod_balances(raw)
    except ValueError as exc:
        assert "date" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing date column")


def test_preprocess_clients_join_and_defaults() -> None:
    clients = pd.DataFrame(
        {
            "client_id": ["C1"],
            "remuneration_condition_id": ["R1"],
            "account_type_id": ["A1"],
        }
    )
    remuneration = pd.DataFrame(
        {
            "remuneration_condition_id": ["R1"],
            "rate_type": ["fixed"],
            "base_rate": [0.01],
            "spread": [0.005],
            "effective_date": ["2024-01-01"],
        }
    )
    account_types = pd.DataFrame(
        {"account_type_id": ["A1"], "type_name": ["Checking"], "currency": ["USD"], "country": ["USA"]}
    )
    out = preprocess_clients(clients, remuneration, account_types)
    assert "acct_country" in out.columns and out.loc[0, "acct_country"] == "USA"
    assert "rem_base_rate" in out.columns and out.loc[0, "rem_base_rate"] == 0.01


def test_link_balances_with_clients_prefers_client_metadata() -> None:
    eod = pd.DataFrame(
        {
            "date": [date(2024, 1, 1)],
            "client_id": ["C1"],
            "balance": [100.0],
            "currency": ["USD"],
            "country": ["USA"],
        }
    )
    clients_master = pd.DataFrame({"client_id": ["C1"], "acct_country": ["UK"], "acct_currency": ["GBP"]})
    out = link_balances_with_clients(eod, clients_master)
    assert out.loc[0, "country"] == "UK"
    assert out.loc[0, "currency"] == "GBP"


def test_aggregate_country_daily_sums() -> None:
    df = pd.DataFrame(
        {
            "date": [date(2024, 1, 1), date(2024, 1, 1)],
            "country": ["USA", "USA"],
            "balance": [100.0, 50.0],
        }
    )
    out = aggregate_country_daily(df)
    assert out.shape == (1, 3)
    assert out.loc[0, "total_deposit"] == 150.0


def test_compute_country_features_rolling_and_join() -> None:
    country_daily = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "country": ["USA", "USA"],
            "total_deposit": [100.0, 300.0],
        }
    )
    rates = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "country": ["USA", "USA"], "avg_interest_rate": [0.02, 0.03]})
    out = compute_country_features(country_daily, rates)
    assert "roll_mean_7" in out.columns and out.loc[out.index[-1], "roll_mean_7"] == 200.0
    assert "avg_interest_rate" in out.columns


def test_ols_slope_intercept_degenerate_and_normal() -> None:
    # Degenerate (constant x)
    a, b = _ols_slope_intercept(np.array([1, 1, 1]), np.array([1.0, 2.0, 3.0]))
    assert b == 0.0
    # Normal case
    a, b = _ols_slope_intercept(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
    assert round(b, 6) == 2.0 and round(a, 6) == 1.0


def test_forecast_deposits_creates_future_rows() -> None:
    features = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "country": ["USA", "USA"],
            "total_deposit": [100.0, 200.0],
            "avg_interest_rate": [0.02, 0.03],
        }
    )
    out = forecast_deposits(features, horizon_days=3, default_avg_rate=0.02, country_avg_rate_overrides=None)
    assert out.shape[0] == 3
    assert set(out["country"]) == {"USA"}
    assert out["forecast_deposit"].min() >= 0.0


def test_forecast_deposits_invalid_horizon() -> None:
    features = pd.DataFrame({"date": [], "country": [], "total_deposit": [], "avg_interest_rate": []})
    try:
        forecast_deposits(features, horizon_days=0, default_avg_rate=0.02, country_avg_rate_overrides=None)
    except ValueError as exc:
        assert "horizon_days" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive horizon")


def test_make_reporting_tables_outputs() -> None:
    hist = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "country": ["USA", "USA"],
            "total_deposit": [100.0, 120.0],
        }
    )
    fc = pd.DataFrame({"date": ["2024-01-03"], "country": ["USA"], "forecast_deposit": [130.0]})
    latest, totals = make_reporting_tables(hist, fc)
    assert "latest_deposit" in latest.columns and latest.shape[0] == 1
    assert "total_historical_deposit" in totals.columns and totals.loc[0, "total_historical_deposit"] == 220.0
