"""Pipeline registry for the Liquidity project.

This exposes a default Kedro pipeline composed of the forecasting pipeline.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from .pipelines.forecasting import pipeline as forecasting


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline name to a ``Pipeline`` instance. The ``__default__``
        key provides the pipeline that runs when using ``kedro run``.
    """

    forecast = forecasting.create_pipeline()
    return {
        "forecasting": forecast,
        "__default__": forecast,
    }

