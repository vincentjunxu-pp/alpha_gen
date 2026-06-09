# alpha_gen

This repository is organized around stable import packages, reusable scripts, and generated artifacts.

## Layout

- `core/`: shared factor calculation, preprocessing, GA, metrics, and Torch backend.
- `behavior_gen/`: structured behavior-finance factor generation framework.
- `behavior_tree_gp/`: typed tree GP scout framework for discovering candidate behavior modes.
- `free_gp_cuda/`: CUDA free-form GP framework.
- `examples/`: runnable experiment entrypoints.
- `tests/`: smoke and regression checks.
- `data/panels/`: parquet panel datasets.
- `data/metadata/`: metadata grouped into production, fixtures, configs, generated files, and archives.
- `data/reference/`: reviewed column lists and screenshot-derived reference files.
- `scripts/data_builders/`: scripts that create panels or metadata.
- `scripts/analysis/`: plotting and result-table analysis scripts.
- `scripts/metadata/`: metadata maintenance scripts.
- `notebooks/`: exploratory notebooks.
- `docs/reports/`: generated reports and report figures.
- `docs/papers/`: reference papers.
- `artifacts/results/`: generated CSV, JSON, and image outputs from runs.

Core package directories are intentionally kept at the repository root so existing imports such as `alpha_gen.core`, `alpha_gen.behavior_gen`, `alpha_gen.behavior_tree_gp`, and `alpha_gen.free_gp_cuda` remain stable.

See `data/metadata/README.md` before adding or replacing metadata files. Runtime code
must use `data/metadata/production/`; tests and mock examples must use
`data/metadata/fixtures/`.

Large datasets, generated results, archived metadata, reports, papers, and
notebooks are local workspace assets and are excluded from source-only Git
uploads.
