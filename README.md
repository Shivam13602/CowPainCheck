# CowPainCheck

Code + documentation for **CowPainCheck** (temporal facial models for cattle pain assessment), organized by **model version** and accompanied by **results summaries**.

## Repository layout

- `Ucaps_raw_videos/`
  - `v2.0/`: original training code (legacy)
  - `v2.3/`: v2.3 pipeline + evaluation (legacy baseline)
  - `v2.4/`: **regression-only** (2D CNN + LSTM + attention) + test evaluation + summary
  - `v2.5/`: **classification-only** (dual-head) + test evaluation + summary + figures
  - `analysis/`: shared statistical analysis used to justify weights/targets
- `paper/`: manuscript artifacts (figures + notes)
- `presentation/`: talk/deck assets
- `docs/`: slides and misc docs (no raw data)

## What is intentionally NOT included

The training/evaluation scripts assume access to UCAPS frame sequences and checkpoint folders, but **raw videos/frames and model checkpoints are not included** in this GitHub repo.

## Quick links (v2.4 / v2.5)

- **v2.4 (Regression-only)**
  - Code: `Ucaps_raw_videos/v2.4/v2.4_training_regression_only.py`
  - Test eval: `Ucaps_raw_videos/v2.4/evaluate_test_set_v2.4.py`
  - Results summary: `Ucaps_raw_videos/v2.4/V2.4_TRAINING_SUMMARY.md`

- **v2.5 (Classification-only)**
  - Code: `Ucaps_raw_videos/v2.5/v2.5_training_classification_only.py`
  - Test eval: `Ucaps_raw_videos/v2.5/evaluate_test_set_v2.5.py`
  - Results summary: `Ucaps_raw_videos/v2.5/V2.5_TRAINING_SUMMARY.md`
  - Figures: `Ucaps_raw_videos/v2.5/visualizations/`

