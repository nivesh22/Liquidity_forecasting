Local Configuration
-------------------

This directory is reserved for developer- or machine-specific configuration that overrides `conf/base`.

Common files you may place here (all optional):
- `parameters.yml` – override any parameters.
- `credentials.yml` – local secrets (never commit real secrets).
- `catalog.yml` – override any dataset entries or filepaths.

Kedro loads both `conf/base` and `conf/local` by default; leaving this folder present but empty is valid.
