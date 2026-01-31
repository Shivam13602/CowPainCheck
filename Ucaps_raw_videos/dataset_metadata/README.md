# Dataset metadata (v2)

This folder contains **lightweight metadata files** required by multiple training/evaluation scripts:

- `v2/train_val_test_splits_v2.json`: cow-exclusive CV splits + held-out test animals
- `v2/sequence_label_mapping_v2.json`: per-sequence label mapping used by dataset loaders
- `v2/sequence_label_mapping_v2.csv`: CSV export of the same mapping (for inspection)
- `v2/averaged_labels_v2.csv`, `v2/animal_variance_analysis_v2.csv`: analysis exports

Raw video/frames are intentionally **not** included in this repo.

