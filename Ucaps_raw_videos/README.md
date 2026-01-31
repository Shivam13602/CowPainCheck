# UCAPS training/evaluation code (versioned)

This directory is organized by **model version**, so each version has its own training script(s), evaluation scripts, and a results summary.

## Versions

- `v2.0/`: early baseline training code (legacy)
- `v2.3/`: legacy baseline (multi-task + evaluation utilities)
- `v2.4/`: **regression-only** (primary regression baseline for the paper)
- `v2.5/`: **classification-only** (dual classification heads)
- `analysis/`: shared statistical analysis and weight assignment notes

## What results should be pushed to GitHub (recommended)

For each version, commit:

- **Code used to train** (the version’s training script)
- **Code used to evaluate on the held-out test set** (e.g., `evaluate_test_set_*.py`)
- **A stable results summary** (markdown) with:
  - test-set ensemble metrics
  - per-moment breakdown (especially M2/M3/M4)
  - any key failure modes
- **Lightweight figures** used in the paper (PNG/SVG/PDF)

Do **not** commit:

- raw videos / extracted frames
- model checkpoints (`.pt`, `.pth`, etc.)
- large intermediate caches

