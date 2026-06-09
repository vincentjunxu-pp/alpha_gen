# Data Layout

- `panels/`: parquet datasets consumed by the frameworks.
- `metadata/`: field rules and dataset configuration. See `metadata/README.md`.
- `reference/`: reviewed column lists and source-screen extracts used by builders.

Generated experiment results do not belong in `data/`; write them under
`artifacts/results/`. Reports and their figures belong under `docs/reports/`.
