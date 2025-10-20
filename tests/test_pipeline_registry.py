from __future__ import annotations

from kedro.pipeline import Pipeline

from liquidity.pipeline_registry import register_pipelines


def test_register_pipelines_has_default() -> None:
    registry = register_pipelines()
    assert "__default__" in registry
    assert isinstance(registry["__default__"], Pipeline)
    assert isinstance(registry.get("forecasting"), Pipeline)

