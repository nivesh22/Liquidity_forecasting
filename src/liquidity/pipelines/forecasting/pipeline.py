"""Kedro pipeline definition for forecasting workflow."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    aggregate_country_daily,
    compute_country_features,
    forecast_deposits,
    link_balances_with_clients,
    make_reporting_tables,
    preprocess_clients,
    preprocess_eod_balances,
)


def create_pipeline() -> Pipeline:
    """Create the forecasting pipeline.

    Returns:
        A ``Pipeline`` instance chaining data cleaning, aggregation, feature engineering,
        forecasting, and reporting tables.
    """

    return Pipeline(
        [
            node(
                func=preprocess_eod_balances,
                inputs="raw_eod_balances",
                outputs="processed_eod_balances",
                name="preprocess_eod",
            ),
            node(
                func=preprocess_clients,
                inputs=["raw_clients", "raw_remuneration", "raw_account_types"],
                outputs="clients_master",
                name="preprocess_clients",
            ),
            node(
                func=link_balances_with_clients,
                inputs=["processed_eod_balances", "clients_master"],
                outputs="linked_balances",
                name="link_balances",
            ),
            node(
                func=aggregate_country_daily,
                inputs="linked_balances",
                outputs="country_daily",
                name="aggregate_country_daily",
            ),
            node(
                func=compute_country_features,
                inputs=["country_daily", "raw_interest_rates"],
                outputs="country_features",
                name="compute_features",
            ),
            node(
                func=forecast_deposits,
                inputs=dict(
                    features="country_features",
                    horizon_days="params:forecast_horizon_days",
                    default_avg_rate="params:default_avg_interest_rate",
                    country_avg_rate_overrides="params:country_avg_rate_overrides",
                ),
                outputs="country_forecast",
                name="forecast_deposits",
            ),
            node(
                func=make_reporting_tables,
                inputs=["country_daily", "country_forecast"],
                outputs=["latest_summary", "country_totals"],
                name="reporting_tables",
            ),
        ]
    )

