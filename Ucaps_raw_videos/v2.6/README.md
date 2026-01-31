# v2.6 — Classification (improved v2.5)

This version focuses on improving **generalization and minority-class performance** of the temporal pain classifier under label imbalance and ambiguous recovery stages.

## What changed vs v2.5 (high level)

- **Pretrained backbone** (ImageNet) instead of training the 2D CNN from scratch.
- **Stronger robustness augmentations** for farm-like variability.
- **Improved class-imbalance handling** using peer-reviewed loss ideas:
  - *Focal Loss* (Lin et al., ICCV 2017) for focusing on hard examples.
  - *Class-Balanced reweighting* (Cui et al., CVPR 2019) based on effective number of samples.
  - Optional *label smoothing* (Szegedy et al., CVPR 2016) to reduce overconfidence.
- **Task 2 redefined as 4-class** (instead of 3-class “Residual Pain”):
  - Class 0: No Pain (M0/M1)
  - Class 1: Acute Pain (M2)
  - Class 2: Declining Pain (M3)
  - Class 3: Recovery (M4)

This addresses the v2.5 failure mode where **M4 collapsed** inside the merged residual class.

## Key files

- Training: `v2.6_training_classification.py`
- Test evaluation: `evaluate_test_set_v2.6.py`

## Outputs (saved during training/eval)

Training and evaluation scripts save lightweight artifacts (CSV/PNG) into:

- `facial_pain_project_v2/checkpoints_v2.6/`
- `facial_pain_project_v2/results_v2.6/`

## References

- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017. (`https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html`)
- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019. (`https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html`)
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision* (label smoothing), CVPR 2016. (`https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf`)
- Hendrycks et al., *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty*, ICLR 2020. (`https://openreview.net/forum?id=S1gmrxHFvB`)
- Zhou et al., *Domain Generalization with MixStyle*, ICLR 2021. (`https://arxiv.org/abs/2107.02053`)

