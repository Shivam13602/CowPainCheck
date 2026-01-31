# Paper draft (running document)

This folder contains the running draft assets for the IEEE-style journal paper based on the work completed in this repository (dataset + temporal models + results).

## What’s in here

- `02_related_work.md`: Draft of **Section II (Related Work)** with **peer‑reviewed journal** references published **after 2020**.

## How we will write the paper (high-level workflow)

- Start from the **main contribution**: **v2.5 dual classification** (binary pain vs no-pain + 3-class pain intensity moment).
- Include **v2.4 regression** as a **secondary/ablation** experiment to show the same efficient backbone can also predict continuous scores.
- Use **v2.3** only as a **prior baseline** (3D CNN + BiLSTM + attention / hybrid objective) to motivate the design pivot to 2D CNN + LSTM.

## Figures (you will add from Colab)

We will reference figures by filename once you paste them here. Suggested filenames:

- `fig_pipeline.png` (data → face crop/sequence → model → outputs)
- `fig_arch_v23_vs_v25.png` (3D CNN+BiLSTM vs 2D CNN+LSTM)
- `fig_task1_confusion.png`, `fig_task2_confusion.png`
- `fig_val_per_fold.png`, `fig_test_ensemble.png`

## Notes / constraints for this draft

- We **do not claim edge hardware deployment** or on-device latency/energy experiments yet (we can add later).
- We **do** report that **all folds were evaluated** and results are from the completed training/evaluation summaries.

---

## III. Dataset

### A. Dataset provenance and study context

Our bovine facial pain dataset was provided by **Dr. Stelio P. L. Luna** and originates from the study and scoring protocol reported in Tomacheuski *et al.* (*Animals*, 2023), which evaluates the **UNESP‑Botucatu Cattle Pain Scale** and the **Cow Pain Scale** for postoperative pain assessment in bulls \[[D1](#references)\]. We use the shared dataset under the same core definitions of **animals**, **postoperative time moments**, and **pain scale variables**.

### B. Subjects, moments, and labels

From the curated label table used in our modeling pipeline (`raw.data.animals.csv`), the dataset contains:

- **Animals**: 20 unique animals
- **Breeds**: *Bos indicus* (Nelore) and *Bos taurus* (Angus)
- **Moments**: M0, M1, M2, M3, M4 (5 temporal stages)
- **Label rows**: 381 averaged records (animal × moment, averaged across evaluators)
- **Pain scales**: NRS (0–10) and VAS (0–100)
- **Facial features (UNESP‑Botucatu; 0–2 each)**: 7 facial action units, plus **Total Facial Scale** (0–14)

These values and definitions are summarized in our internal analysis document (`Ucaps_raw_videos/dataanlasis.md`).

### C. Video representation and sequence construction (for deep learning)

The raw inputs are facial video clips. For modeling, videos are converted into **fixed-length frame sequences** (uniform sampling) and resized for efficient training on available GPUs (T4/L4). Labels are attached at the **sequence level** according to the animal’s moment and associated scale values.

For our primary (v2.5) classification experiments, we use two targets derived from moments:

- **Task 1 (binary pain)**: No‑pain = {M0, M1}; Pain = {M2, M3, M4}
- **Task 2 (3‑class intensity moment)**:
  - Class 0 (No pain) = {M0, M1}
  - Class 1 (Acute pain) = {M2}
  - Class 2 (Residual pain) = {M3, M4}

### D. Data splits and evaluation principle

All available cross‑validation folds were evaluated, and final reporting uses the completed fold-wise validation analysis and held‑out test evaluation described in `Ucaps_raw_videos/V2.5_TRAINING_SUMMARY.md`.

---

## References

- **[D1]** R. M. Tomacheuski *et al*., “Reliability and validity of UNESP-Botucatu Cattle Pain Scale and Cow Pain Scale in *Bos taurus* and *Bos indicus* bulls to assess postoperative pain of surgical orchiectomy,” *Animals*, vol. 13, no. 3, Art. no. 364, 2023, doi: `10.3390/ani13030364`. Available: `https://pmc.ncbi.nlm.nih.gov/articles/PMC9913732/`


