"""Project settings for the Liquidity Kedro project.

This configures the Kedro configuration loader and sets a default
configuration environment so running ``kedro run`` uses the
``conf/liquidity`` environment unless explicitly overridden.
"""

from __future__ import annotations

from kedro.config import OmegaConfigLoader

# Use OmegaConf-based loader (default in Kedro 1.x)
CONFIG_LOADER_CLASS = OmegaConfigLoader

# Base and default run environment. ``default_run_env`` controls which
# conf/<env> folder is merged in when ``kedro run`` is executed without
# ``-e/--env``. This pairs with the project's ``.env`` which sets
# ``KEDRO_ENV=liquidity`` for clarity when using other tooling.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "liquidity",
}

# Explicit empty hooks tuple (can register project hooks here later)
HOOKS: tuple = ()

