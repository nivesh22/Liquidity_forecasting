"""Nodes for processing deposit data and forecasting by country.

All functions are pure, documented, and designed for Kedro pipelines.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def preprocess_eod_balances(eod: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize end-of-day balance data.

    Args:
        eod: Raw EOD balances with columns: ``[date, client_id, balance, currency, country]``.

    Returns:
        DataFrame with parsed dates, non-negative balances, and standardized columns.
    """

    df = eod.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("'date' column required in eod balances")
    if "client_id" not in df.columns:
        raise ValueError("'client_id' column required in eod balances")
    if "balance" not in df.columns:
        raise ValueError("'balance' column required in eod balances")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce").fillna(0.0)
    df = df[df["balance"] >= 0.0]
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown").astype(str)
    else:
        df["country"] = "Unknown"
    if "currency" in df.columns:
        df["currency"] = df["currency"].fillna("UNK").astype(str)
    else:
        df["currency"] = "UNK"
    return df[["date", "client_id", "balance", "currency", "country"]]


def preprocess_clients(
    clients: pd.DataFrame, remuneration: pd.DataFrame, account_types: pd.DataFrame
) -> pd.DataFrame:
    """Create a master client table by joining clients, remuneration, and account types.

    Args:
        clients: Client-level attributes, contains ``client_id``, ``remuneration_condition_id``,
            and ``account_type_id`` at minimum.
        remuneration: Remuneration conditions with columns such as
            ``[remuneration_condition_id, rate_type, base_rate, spread, effective_date]``.
        account_types: Account type metadata with columns such as
            ``[account_type_id, type_name, currency, country]``.

    Returns:
        A master client DataFrame with enriched remuneration and account type fields.
    """

    c = clients.copy()
    r = remuneration.copy()
    a = account_types.copy()

    for df in (c, r, a):
        df.columns = [x.strip().lower() for x in df.columns]

    if "client_id" not in c.columns or "account_type_id" not in c.columns:
        raise ValueError("clients must have 'client_id' and 'account_type_id'")

    # Ensure proper types
    if "effective_date" in r.columns:
        r["effective_date"] = pd.to_datetime(r["effective_date"], errors="coerce").dt.date

    master = c.merge(
        r.add_prefix("rem_"),
        left_on="remuneration_condition_id",
        right_on="rem_remuneration_condition_id",
        how="left",
    ).merge(a.add_prefix("acct_"), left_on="account_type_id", right_on="acct_account_type_id", how="left")

    # Standardize
    master["acct_country"] = master.get("acct_country", "Unknown").fillna("Unknown").astype(str)
    master["acct_currency"] = master.get("acct_currency", "UNK").fillna("UNK").astype(str)
    master["rem_base_rate"] = pd.to_numeric(master.get("rem_base_rate", 0.0), errors="coerce").fillna(0.0)
    master["rem_spread"] = pd.to_numeric(master.get("rem_spread", 0.0), errors="coerce").fillna(0.0)

    return master


def link_balances_with_clients(eod: pd.DataFrame, clients_master: pd.DataFrame) -> pd.DataFrame:
    """Join EOD balances with client master to attach country/currency and remuneration info.

    Args:
        eod: Preprocessed EOD balances.
        clients_master: Output of ``preprocess_clients``.

    Returns:
        Joined DataFrame suitable for aggregation.
    """

    left = eod.copy()
    right = clients_master.copy()

    df = left.merge(right, on="client_id", how="left")
    # Fallback country if missing in client master
    df["country"] = df["acct_country"].fillna(df["country"]).fillna("Unknown")
    df["currency"] = df["acct_currency"].fillna(df["currency"]).fillna("UNK")
    return df


def aggregate_country_daily(linked: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily deposits by country.

    Args:
        linked: Output of ``link_balances_with_clients``.

    Returns:
        DataFrame with columns ``[date, country, total_deposit]``.
    """

    df = linked.copy()
    g = (
        df.groupby(["date", "country"], as_index=False)["balance"].sum().rename(columns={"balance": "total_deposit"})
    )
    g["total_deposit"] = g["total_deposit"].astype(float)
    return g


def compute_country_features(
    country_daily: pd.DataFrame, interest_rates: pd.DataFrame
) -> pd.DataFrame:
    """Attach interest rates and rolling features for country-level deposits.

    Args:
        country_daily: Output of ``aggregate_country_daily``.
        interest_rates: Daily interest rates with ``[date, country, avg_interest_rate]``.

    Returns:
        Country-level features with rolling means.
    """

    cd = country_daily.copy()
    ir = interest_rates.copy()
    cd["date"] = pd.to_datetime(cd["date"]).dt.date
    ir["date"] = pd.to_datetime(ir["date"]).dt.date

    feat = cd.merge(ir, on=["date", "country"], how="left")
    feat["avg_interest_rate"] = pd.to_numeric(feat["avg_interest_rate"], errors="coerce").fillna(0.0)

    # Rolling features per country (7-day and 30-day mean)
    feat = feat.sort_values(["country", "date"])  # stable rolling
    feat["total_deposit"] = feat["total_deposit"].astype(float)
    feat["roll_mean_7"] = (
        feat.groupby("country")["total_deposit"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    feat["roll_mean_30"] = (
        feat.groupby("country")["total_deposit"].transform(lambda s: s.rolling(30, min_periods=1).mean())
    )
    return feat


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute simple OLS slope and intercept for y ~ a + b x.

    Args:
        x: Feature vector (1D).
        y: Response vector (1D).

    Returns:
        Tuple of (intercept, slope).
    """

    if x.size == 0 or y.size == 0:
        return 0.0, 0.0
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0.0:
        return y_mean, 0.0
    slope = float(((x - x_mean) * (y - y_mean)).sum()) / denom
    intercept = y_mean - slope * x_mean
    return float(intercept), float(slope)


def forecast_deposits(
    features: pd.DataFrame,
    horizon_days: int,
    default_avg_rate: float,
    country_avg_rate_overrides: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Forecast future deposits by country using a simple elasticity model.

    The model estimates log(1 + deposit) = a + b * interest_rate per country.
    Forecasts use provided average interest rate (per-country override if given) over the horizon.

    Args:
        features: Country features with columns ``[date, country, total_deposit, avg_interest_rate]``.
        horizon_days: Number of days to forecast ahead.
        default_avg_rate: Fallback average interest rate if no per-country override is found.
        country_avg_rate_overrides: Optional mapping of country -> average interest rate.

    Returns:
        DataFrame with future dates and forecasted deposits: ``[date, country, forecast_deposit]``.
    """

    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    df = features.copy()
    df = df.dropna(subset=["country", "total_deposit"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["country", "date"])  # stable ordering

    forecasts: list[dict[str, object]] = []
    overrides = country_avg_rate_overrides or {}

    for country, g in df.groupby("country"):
        x = pd.to_numeric(g["avg_interest_rate"], errors="coerce").fillna(0.0).to_numpy()
        y = np.log1p(pd.to_numeric(g["total_deposit"], errors="coerce").fillna(0.0).to_numpy())

        intercept, slope = _ols_slope_intercept(x, y)
        rate_future = float(overrides.get(str(country), default_avg_rate))

        # Predict constant log-deposit over forecast horizon using provided rate
        yhat = intercept + slope * rate_future
        deposit_pred = float(np.expm1(yhat))
        last_date: date = g["date"].max()

        for i in range(1, horizon_days + 1):
            d = last_date + timedelta(days=i)
            forecasts.append({
                "date": d,
                "country": country,
                "forecast_deposit": max(0.0, deposit_pred),
            })

    out = pd.DataFrame(forecasts)
    if not out.empty:
        out = out.sort_values(["country", "date"]).reset_index(drop=True)
    return out


def make_reporting_tables(
    country_daily: pd.DataFrame, forecast: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate reporting tables for the dashboard.

    Args:
        country_daily: Aggregated deposits by date and country.
        forecast: Forecasted deposits by date and country.

    Returns:
        Tuple of (latest_summary, country_totals):
            - latest_summary: latest actual deposit by country with simple day-over-day change.
            - country_totals: total historical deposits by country.
    """

    hist = country_daily.copy()
    hist["date"] = pd.to_datetime(hist["date"]).dt.date
    latest_date = hist["date"].max() if not hist.empty else None

    if latest_date is None:
        latest_summary = pd.DataFrame(columns=["country", "latest_deposit", "dod_change"])
    else:
        latest = hist[hist["date"] == latest_date][["country", "total_deposit"]].rename(
            columns={"total_deposit": "latest_deposit"}
        )
        prev = hist[hist["date"] == (latest_date - timedelta(days=1))][
            ["country", "total_deposit"]
        ].rename(columns={"total_deposit": "prev_deposit"})
        latest_summary = latest.merge(prev, on="country", how="left")
        latest_summary["prev_deposit"] = latest_summary["prev_deposit"].fillna(0.0)
        latest_summary["dod_change"] = (
            latest_summary["latest_deposit"] - latest_summary["prev_deposit"]
        )

    country_totals = (
        hist.groupby("country", as_index=False)["total_deposit"].sum().rename(
            columns={"total_deposit": "total_historical_deposit"}
        )
    )
    return latest_summary, country_totals
