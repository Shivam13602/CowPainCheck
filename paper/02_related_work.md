**Title (draft):** CowPainCheck: An IoT-Oriented Edge-to-Cloud Pipeline for Bovine Pain Monitoring From Facial Video Using Efficient CNN–LSTM Sequence Learning

**Authors:** (to be added)  
**Affiliations:** (to be added)  

## Abstract

Reliable on-farm pain monitoring in cattle remains constrained by intermittent expert observation, subjective scoring, and limited scalability in large herds. This paper presents CowPainCheck, an end-to-end, IoT-oriented system that integrates high-accuracy facial detection, standardized face-crop sequence construction, and compact temporal deep learning models for postoperative pain assessment in cattle. The pipeline follows an edge-first design in which face localization and temporal inference are performed near the camera node, while only compact metadata are transmitted upstream, consistent with cloud–fog–edge architectures for smart agriculture. Pain inference is formulated as a dual-head temporal learning problem, comprising binary pain versus no-pain classification and a three-class intensity–moment categorization aligned with the UNESP‑Botucatu Cattle Pain Scale (no pain, acute pain, residual pain). A complementary regression model predicts facial action units and Total Facial Scale values to support continuous pain assessment. Experiments conducted using animal-wise cross-validation with an independent held-out test set (Animals 14 and 17; 35 sequences) demonstrate strong performance for binary pain detection (accuracy 0.9143, F1-score 0.9333, recall 1.0000) and more challenging performance for intensity–moment classification (accuracy 0.6000) and regression (MAE 1.6527, \(R^2 = 0.3125\), \(r = 0.5696\)). Deployment-oriented considerations, including model size, inference latency, metadata-first communication, and security and privacy requirements, are discussed to position CowPainCheck within practical edge-enabled IoT architectures for precision livestock pain monitoring.

## Index Terms

Internet of Things (IoT); precision livestock farming; animal welfare; bovine pain assessment; facial expression analysis; video classification; temporal deep learning; You Only Look Once (YOLO); long short-term memory (LSTM); attention; edge computing; fog computing; security and privacy.

## I. Introduction

Pain management is a central concern in cattle welfare and clinical practice, yet routine on-farm monitoring is constrained by time, expertise availability, and the inherent subjectivity of manual scoring. Recent systematic syntheses emphasize that common pain assessment tools (e.g., numerical rating scales and visual analogue scales) can be informative but remain labor-intensive and prone to inter-observer variability \[1\]. In bovine postoperative contexts, validated composite scoring instruments—such as the UNESP‑Botucatu Cattle Pain Scale and the Cow Pain Scale—provide structured facial and behavioral assessment protocols that support more reliable evaluation \[12\]. These factors motivate automated tools that can provide objective, repeatable, and scalable pain assessment without requiring continuous expert observation.

Computer vision offers a natural path toward automation by leveraging facial cues and temporal dynamics in video. Recent surveys of animal pain and emotion recognition highlight both the promise of visual monitoring and the practical difficulties of on-farm deployment, including domain shift, occlusion, viewpoint variation, and the need for temporal modeling to capture brief affective events \[3\]–\[5\]. In parallel, Precision Livestock Farming literature frames monitoring as an end-to-end sensing and analytics pipeline, emphasizing constraints such as robustness and scalability \[8\]–\[10\]. Within this context, facial-video-based temporal learning provides an opportunity to translate validated pain-scale frameworks into automated decision support using convolutional neural networks (CNNs) and recurrent temporal models.

This paper develops and evaluates an end-to-end pipeline for bovine pain inference from facial video, framed for IoT-enabled welfare monitoring. The approach combines (i) face localization using trained single-stage detectors, (ii) standardized sequence construction from detected face crops, and (iii) compact temporal deep learning models for pain inference. The primary output is dual-head classification to provide clinically interpretable decisions (pain presence and intensity-moment grouping), while a regression model predicting validated facial action units is included as supporting evidence that the same backbone can model continuous targets. Evaluation is animal-wise to emphasize generalization to unseen individuals and includes an independent held-out test set.

The main contributions are threefold. First, an IoT-oriented, end-to-end camera-node pipeline is presented for pain monitoring, including an explicit system model (edge–fog–cloud roles, data flow, and security considerations). Second, an efficient temporal sequence learner is developed to produce clinically interpretable classification outputs from face-crop video sequences, with a supporting regression experiment tied to validated pain-scale components. Third, the pipeline is evaluated under animal-wise splitting with held-out animals to prioritize generalization in realistic monitoring settings.

The remainder of the paper is organized as follows. Section II reviews related work with emphasis on IoT architectures and edge intelligence for livestock monitoring. Section III presents the IoT system model and framework. Section IV describes the dataset and preprocessing. Section V presents the end-to-end methodology and model designs. Section VI describes the evaluation protocol. Section VII reports results. Sections VIII–X discuss findings, limitations, and future directions.

## II. Related Work

Automated pain assessment in livestock has increasingly shifted from subjective, intermittent scoring toward **continuous and objective monitoring**, motivated by welfare, productivity, and the practical limits of expert observation in large herds. Prior work spans **IoT architectures for monitoring**, **edge intelligence for smart agriculture**, and **computer vision methods for animal pain/emotion recognition**.

### A. IoT architectures for livestock monitoring and welfare sensing

Precision Livestock Farming (PLF) literature frames animal welfare monitoring as an end-to-end pipeline spanning sensing, connectivity, and analytics. A Journal of Cleaner Production review surveys wearable IoT technologies and practical constraints for scalable PLF, including robustness and sustainability considerations \[8\]. A Frontiers systematic review on dairy-cattle welfare sensing emphasizes the role of validated sensor technologies for continuous assessment and highlights the engineering trade-offs involved in real deployments \[15\]. A recent Animals review further summarizes IoT sensor applications in dairy cattle farming, underscoring the importance of integrating heterogeneous sensors into farm management workflows \[16\]. These studies collectively motivate architecture-first approaches that treat perception models as one component in a larger monitoring system.

### B. Edge intelligence and cloud–fog–edge perspectives

Smart-agriculture IoT commonly adopts cloud–fog–edge partitioning to balance latency, bandwidth, and reliability. A Sensors survey reviews the role of cloud–fog–edge combinations in smart agriculture systems \[10\]. Complementary work in Science of The Total Environment illustrates how cloud/edge architectures support real-world constraints in livestock supply-chain contexts (connectivity, device capability, workflow) \[9\]. In welfare monitoring, sensor-based affective computing perspectives further emphasize the need for reliable, scalable data acquisition and processing pipelines \[17\].

### C. IoT communication protocols and security considerations

For operational IoT deployments, the choice of protocols and security mechanisms can be as important as model accuracy. Recent peer-reviewed work summarizes IoT communication protocol properties and common security threats, motivating encrypted transport (e.g., Transport Layer Security (TLS)) and access control for camera and metadata streams \[18\]. These requirements are particularly relevant for on-farm video systems, where incidental capture of workers and sensitive operational context may occur.

### D. Computer vision for animal pain and emotion recognition

Computer vision for animal affect recognition has progressed from coarse detection and tracking toward **behavioral and affective state inference**. A comprehensive survey in *International Journal of Computer Vision* reviews the state of the art across species, emphasizing that robust pain/emotion recognition typically requires modeling of subtle facial/body cues, domain shift, and temporal dynamics \[3\]. For large domestic animals, a Frontiers systematic review examines whether facial-expression-based pain scoring is reliable and accurate, and discusses sources of disagreement and bias that automated systems must handle (viewpoint, occlusion, individual morphology) \[4\]. A more recent narrative review summarizes algorithmic pipelines and evaluation practices for animal pain recognition technologies \[5\]. Beyond surveys, journal studies demonstrate automated/assisted approaches for dairy cattle pain using facial-expression protocols \[6\] and multimodal sensing combined with visual observation for sickness/pain detection \[7\]. Collectively, these works support two recurring findings that motivate sequence-based learning: (i) **temporal cues matter**, because pain-related facial events can be brief; and (ii) **farm data are noisy**, requiring generalization across animals and recording conditions.

### E. Veterinary pain assessment in cattle and the role of facial cues

Veterinary practice commonly relies on clinical scoring systems (e.g., numerical rating scales and visual analogue scales), but these remain labor-intensive and can exhibit variability across observers and contexts. Recent cattle-focused synthesis studies summarize existing scoring approaches and their measurement properties, underscoring both the need for standardization and the opportunity for objective tools that can generalize across environments and animals \[1\]. In the specific context of bovine postoperative pain, recent work reports reliability and validity evidence for cattle pain scales in both *Bos taurus* and *Bos indicus* \[12\]. Parallel work on grimace scales across non-human mammals reviews reliability and validity considerations, highlighting why facial-action-based measures are attractive targets for automation \[2\].

### F. Positioning of this study

In contrast to many prior works that focus on a single output, the proposed pipeline emphasizes **temporal sequence modeling** and produces **clinically interpretable classification outputs** (binary pain vs. no-pain; and a 3-class intensity-moment grouping). The implementation choices (efficient CNN backbones with recurrent temporal modeling) align with the broader trend toward practical, deployable monitoring pipelines discussed in the PLF and edge–cloud literature \[8\]–\[10\], while the evaluation protocol prioritizes generalization across folds and animals as recommended by recent pain-recognition surveys \[3\]–\[5\].

## III. IoT System Model and Framework

This section summarizes the system-level framing required for IoT-enabled welfare monitoring. The proposed system is designed around camera nodes that acquire facial video, an edge compute module that performs face localization and sequence inference, an optional fog layer for aggregation across multiple cameras/animals, and a cloud layer for long-term storage and analytics. This partitioning follows common cloud–fog–edge design principles for smart agriculture \[10\].

### A. System architecture and data flow

In the intended deployment, each camera node streams video to a nearby edge compute unit (or performs on-device processing if hardware permits). The edge module performs face localization, crop/sequence generation, and temporal inference, and then transmits compact metadata (pain scores, confidence, animal ID, timestamp, and quality flags) to upstream services. Raw video is retained locally for short periods and is uploaded only on-demand (e.g., low confidence, anomalous events, veterinary audit), reducing network overhead in bandwidth-limited rural settings \[9\], \[10\].

The end-to-end IoT system model and data flow are illustrated in **Fig. 2**, including the roles of camera nodes, edge inference, fog aggregation, and cloud analytics.

**Fig. 2.** System model of CowPainCheck within an IoT edge–fog–cloud architecture. Camera nodes acquire facial video streams; an edge module performs face localization, crop/sequence construction, and temporal inference; a fog layer aggregates multi-animal metadata for local decision support and alerting; and a cloud layer provides long-term storage, analytics, and integration with herd-management services. The architecture emphasizes metadata-first transmission with event-triggered video upload.

### B. Communication and interoperability considerations

Metadata-first communication enables interoperability with farm management systems and reduces network load compared to continuous raw-video streaming. Protocol selection and message delivery guarantees should consider reliability and fault tolerance, and encrypted transport is recommended for both metadata and event-triggered video \[18\]. While protocol benchmarking is outside the scope of the current experiments, the system model is compatible with lightweight publish/subscribe workflows commonly used in IoT deployments.

### C. Security and privacy considerations

Video-based livestock monitoring systems introduce privacy and security risks, including unauthorized access to camera streams and incidental capture of humans in barn environments. A secure design should include encrypted communication channels, authentication/authorization, and access-controlled storage, consistent with the threat landscape described in recent protocol/security surveys \[18\]. These measures are architectural requirements even when model training and evaluation are performed offline.

## IV. Dataset and Preprocessing

### A. Dataset origin and pain scale definitions

The dataset used in this work originates from the study protocol and annotations described in Tomacheuski *et al.* \[12\], which evaluates the **UNESP‑Botucatu Cattle Pain Scale** and the **Cow Pain Scale** for postoperative pain assessment in bulls. The dataset includes repeated measurements across standardized postoperative time points (“moments”) and contains both continuous pain scales (e.g., NRS and VAS) and facial-action-based scores.

### B. Subjects, video assets, and moment coverage

The video corpus comprises **20 animals** and **138 facial videos** (≈16.39 GB; ≈1.11 hours total). All videos are high definition (1920×1080) with a consistent frame rate of 24 fps. After matching videos to available pain-score records, **19 animals** contribute **130 video–score pairs** with usable labels for modeling and evaluation. Two animals (14 and 17) are reserved as an independent held-out test set and are not used for model fitting or model selection; the remaining labeled animals are used within animal-wise cross-validation.

Videos are organized into five postoperative **moments** (M0–M4). Overall moment coverage is complete for M0, M1, and M4 (20/20 animals) and slightly reduced for M2 and M3 due to missing recordings. This missingness is carried through downstream analyses and is handled via animal-wise splitting to avoid leakage across train/validation/test partitions.

The canonical postoperative pain progression used throughout the manuscript (baseline → acute pain → recovery) is depicted in **Fig. 1** to motivate the intensity-moment grouping used in classification and the moment-weighted objectives used in regression.

**Fig. 1.** Moment-wise pain trajectory of UNESP‑Botucatu facial action units (0–2 ordinal intensities) across postoperative moments (M0 baseline; M1 ~30 min; M2 ~2–4 h; M3 ~6–8 h; M4 ~24 h). Curves show mean facial-action-unit intensity aggregated over evaluator-level annotations; shaded bands indicate the approximate 95% confidence interval of the mean. The Numerical Rating Scale (NRS; 1–10) is linearly rescaled to a 0–2 range for visual comparability (diamond markers). The highlighted region and dashed vertical line indicate the acute pain window centered on M2.

### C. Label tables and evaluator structure

Two complementary label representations are used. First, evaluator-level annotations are available per animal and moment, including NRS, VAS, behavioral components, and UNESP‑Botucatu facial scores. Second, for modeling, evaluator-level records are aggregated (averaged) at the animal–moment level to mitigate inter-rater variability; the resulting table contains 381 animal–moment records and serves as the primary target source for moment-wise analysis and weighting.

### D. Preprocessing for face detection annotation (frame extraction)

To train and validate face-detection components used in the downstream pipeline, representative frames were extracted from the raw videos for manual annotation using a deterministic-yet-diverse sampling strategy. In total, 3,500 frames were targeted, with 15–35 frames sampled per video depending on duration. Frames were distributed approximately uniformly across each video, with frame indices jittered using bounded random offsets to reduce repeated sampling of identical temporal positions; extracted frames were organized using a structured naming convention (animal ID, moment, and timestamp) and accompanied by structured metadata for tracking and reproducibility.

Manual annotations follow a consistent guideline: bounding boxes are drawn tightly around the head/face region while including ears, eyes, and muzzle; guidance for partial profiles and multi-animal frames is captured alongside the extracted frame metadata.

### E. IoT environment and deployment implications

Although collected in a controlled postoperative study, the videos include realistic variability in head pose, movement, and background clutter, motivating robust face localization, standardized crop generation, and temporal aggregation \[15\]–\[17\].

Key dataset statistics and derived assets used in the pipeline are summarized in **Table I**.

**Table I.** Summary of the UCAPS video corpus and derived assets used for model development and evaluation, including subject/video counts, acquisition properties, label availability, derived sequence/crop counts, and the held-out test set (Animals 14 and 17).
| Item | Value |
|------|-------|
| Subjects | 20 animals |
| Videos | 138 facial videos |
| Video properties | 1920×1080, 24 fps |
| Total video size / duration | ≈16.39 GB; ≈1.11 hours |
| Postoperative moments | M0–M4 |
| Scored animals / usable video–score pairs | 19 animals; 130 pairs |
| Face-crop sequences (10 s windows, 2 s overlap) | 392 sequences |
| Total extracted face crops | 92,494 crops |
| Held-out test set | Animals 14 and 17; 35 sequences |
| Face detector annotation set | 4,265 labeled frames (2,955 train / 810 val / 389 test) |

## V. Methodology (End-to-End Pipeline)

To maintain a clear narrative flow while ensuring reproducibility, the full pipeline from raw videos to temporal prediction is described. Consistent with prior computer-vision practice, **face detection is treated as an enabling preprocessing stage**, and the main methodological emphasis is placed on the temporal learning models for pain inference \[12\], \[13\].

### A. Overview

Given an input facial video recorded for a specific animal and postoperative moment, face localization is first performed to identify the head/face region per frame. The detected regions are then used to construct standardized face-crop sequences by resizing crops to a fixed spatial resolution and segmenting each video into fixed-duration windows. Finally, spatiotemporal representations are learned from these sequences to predict pain-related targets, with classification as the primary focus and regression treated as a secondary study.

An overview of the full processing pipeline is shown in **Fig. 3**.

**Fig. 3.** End-to-end processing pipeline from raw video to pain inference. Frames are processed by a face detector to localize the head/face region; standardized face crops are generated and segmented into fixed-length overlapping sequences; and temporal learning models produce (i) dual-head classification outputs (binary pain and 3-class intensity-moment grouping) with (ii) a supporting regression variant predicting facial action units and a total facial score.

### B. Face detection model training (preprocessing stage)

To support robust face localization under farm recording conditions, two single-stage object detectors were trained on a manually annotated dataset of **4,265 frames**. The dataset was split into **2,955** training images (≈70.8%), **810** validation images (≈19.5%), and **389** test images (≈9.4%). The two configurations were selected to explicitly represent the common deployment trade-off between accuracy and computational cost: **You Only Look Once (YOLO) v11s** as a higher-accuracy detector and **YOLOv8n** as a lightweight alternative with fewer parameters, which is attractive when considering future resource-constrained (edge) deployment.

Both detector configurations were evaluated and accuracy/efficiency trade-offs are reported. **YOLOv11s** achieved the best overall detection performance (mean average precision (mAP)@0.5 = 99.4%, precision 98.7%, recall 97.5%). **YOLOv8n** achieved slightly lower accuracy (mAP@0.5 = 98.9%) while offering a faster runtime and smaller model footprint, motivating its use when prioritizing throughput and efficiency in large-scale crop generation. The choice of a lightweight YOLOv8 variant for cattle facial-feature pipelines is supported by recent peer-reviewed work in cattle identification using facial cues \[13\], while recent benchmarking work in cattle monitoring highlights the practical benefits of evaluating newer YOLO families for accuracy–efficiency trade-offs \[14\].

Quantitative detection performance is reported in **Table II**, while representative qualitative examples are provided in **Fig. 4**.

**Table II.** Face detector performance on the annotated detection test set. Metrics include mean average precision (mAP) at IoU 0.5 and 0.5:0.95, precision, and recall, together with model size/parameter counts and per-image inference time to characterize the accuracy–efficiency trade-off between YOLOv11s (accuracy-priority) and YOLOv8n (edge/efficiency-priority) for downstream crop generation.
| Detector | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Parameters | Model size | Inference speed | Intended use |
|----------|---------|--------------|-----------|--------|------------|------------|-----------------|--------------|
| YOLOv11s (conservative aug.) | 0.994 | 0.859 | 0.987 | 0.975 | 9.4M | 19.2 MB | 5.0 ms/img | Accuracy-priority crop extraction |
| YOLOv8n (minimal aug.) | 0.989 | 0.891 | 0.981 | 0.956 | 3.0M | 5.9 MB | 2.9 ms/img | Efficiency/edge-priority crop extraction |

**Fig. 4.** Qualitative face-detection results under farm-recording conditions. Representative frames illustrate successful localization across viewpoint, illumination changes, and background clutter, along with common failure modes such as partial occlusion, extreme head pose, and multi-animal scenes. These examples motivate the use of robust detection thresholds and standardized crop generation prior to temporal modeling.

### C. Face crop sequence generation

After training, the face detector was applied to the full set of 138 videos to generate standardized face crops and temporal sequences for learning. This procedure produced 392 sequences and 92,494 individual face crops. Each sequence spans 10 s (240 frames at 24 fps), with a 2 s overlap between consecutive sequences to preserve temporal continuity. Each detected face crop is resized to 224×224 pixels and detections below a confidence threshold of 0.6 are discarded. These choices balance temporal coverage with manageable sequence length while preserving spatial detail in the facial region.

### D. Temporal pain inference models

Pain inference is formulated as temporal learning from fixed-length facial video segments. Following common practice in video-based affect recognition, early prototypes explored a high-capacity spatiotemporal design to jointly model appearance and short-term motion, and subsequent iterations emphasized parameter efficiency for improved generalization under limited data \[3\]–\[5\].

Three model stages are described for clarity:

First, a **spatiotemporal baseline** uses a 3D convolutional encoder to capture local motion cues over short frame windows, followed by a **bidirectional LSTM** with an attention aggregator to capture longer temporal dependencies. This stage optimizes a hybrid objective comprising continuous pain regression and binary pain/no-pain classification, enabling joint learning of intensity and categorical pain presence.

Second, an **efficient temporal backbone** is adopted for the main experiments. Each frame is processed independently by a compact four-block 2D convolutional neural network (CNN) (channel progression 3→32→64→128→256 with batch normalization, ReLU activations, max pooling, and global average pooling), producing a 256-dimensional per-frame embedding. A single-layer, unidirectional long short-term memory network (LSTM) (hidden size 128) models temporal structure across embeddings, and an attention layer aggregates the sequence into a fixed-length context vector. This design is parameter-efficient and aligns with the need to learn robust representations from a small, animal-wise dataset.

Third, the primary model extends the efficient backbone with **dual classification heads** that share the same temporal context. The first head performs **binary pain classification** (no pain vs pain), with label mapping M0/M1→0 and M2/M3/M4→1. The second head performs **3-class intensity-moment classification**, grouping moments into clinically meaningful categories: (i) no pain (M0/M1), (ii) acute pain (M2), and (iii) residual pain (M3/M4). Both heads are shallow multilayer perceptrons (Linear→ReLU→Dropout→Linear) applied to the shared context vector. A regression variant of the same backbone is included as a supporting experiment, predicting the seven UNESP‑Botucatu facial action units and using their sum as the Total Facial Scale.

The model stages and their outputs are summarized in **Table III**, and the core temporal architecture is depicted in **Fig. 5**.

**Table III.** Summary of temporal modeling stages used in the study. Each stage is characterized by its spatial encoder, temporal aggregation module, output targets, and experimental role. The primary contribution is the efficient CNN–LSTM–attention backbone with dual classification heads; a regression variant of the same backbone supports continuous prediction of facial action units and total facial scale.
| Model stage | Spatial encoder | Temporal module | Outputs | Primary objective |
|------------|------------------|-----------------|---------|-------------------|
| Spatiotemporal baseline | 3D CNN | BiLSTM + attention | regression + binary class | exploratory baseline |
| Efficient temporal backbone (supporting) | 2D CNN | LSTM + attention | regression (7 facial AUs + total) | supporting experiment |
| Efficient temporal backbone (primary) | 2D CNN | LSTM + attention | binary class + 3-class intensity-moment | primary contribution |

**Fig. 5.** Temporal learning architecture used for pain inference. Each frame is encoded by a 2D convolutional neural network (CNN) backbone into a per-frame embedding; a long short-term memory network (LSTM) models temporal dependencies; an attention mechanism aggregates the sequence into a context vector; and task-specific heads output either dual-head classification (binary pain and 3-class intensity-moment) or supporting regression targets (facial action units and total facial scale).

### E. Loss functions and data-driven weighting

For the dual-head classifier, optimization combines binary cross-entropy with logits for the pain/no-pain task and cross-entropy for the 3-class intensity-moment task. Both tasks apply **moment-weighted learning** to emphasize clinically critical and empirically challenging regimes. The peak-pain moment (M2) is assigned the highest weight (4.0×), with intermediate emphasis on M3 (2.0×) and a smaller boost for M4 (1.2×), while M0 and M1 use baseline weight (1.0×). Class imbalance is addressed via a positive-class weight for the binary task and inverse-frequency class weights for the 3-class task, computed from the training fold distribution.

For the regression variant, a weighted objective incorporates both (i) **moment weights** (as above) and (ii) **feature weights** derived from dataset statistics, assigning higher emphasis to facial features with stronger association to the Total Facial Scale. A consistency term encourages agreement between the directly predicted total score and the total computed from the predicted facial action units.

### F. Training procedure

All temporal models were trained for up to 60 epochs with early stopping and adaptive learning-rate scheduling. Standard image augmentations were applied to improve generalization under variable illumination and viewpoint (horizontal flips, affine transforms, color jitter, and Gaussian blur). Training was performed in Google Colab environments using NVIDIA T4 and NVIDIA L4 GPUs.

## VI. Experimental Setup and Evaluation Protocol

Evaluation is performed using animal-wise splitting to prevent identity leakage between training and evaluation. Two animals (Animals 14 and 17) are reserved as a fixed held-out test set (35 sequences) and are not used for model fitting or model selection. The remaining 18 animals are evaluated using a leave-two-animals-out protocol comprising 9 folds; in each fold, two animals are held out for validation and the remaining 16 animals are used for training. This design ensures that each animal in the cross-validation pool is used for validation exactly once.

For the dual-head classifier, performance is reported using accuracy, precision, recall, and F1-score. For the 3-class task, weighted averaging is used to account for class imbalance. Performance is additionally analyzed by postoperative moment and by test animal to assess generalization patterns.

For the regression model, performance is reported using mean absolute error (MAE), root mean square error (RMSE), coefficient of determination (R²), and Pearson correlation (r) with corresponding significance testing. Moment-wise and feature-wise performance breakdowns are reported to connect model behavior to the underlying pain-scale components.

The cross-validation protocol and the resulting sequence distribution by postoperative moment for train/validation (averaged across folds) and the held-out test set are summarized in **Fig. 6**.

**Fig. 6.** Animal-wise cross-validation protocol and sequence distribution derived from the generated face-crop clips. Left: leave-two-animals-out cross-validation over 9 folds on the 18-animal pool (blue cells indicate the two validation animals per fold), with Animals 14 and 17 held out as a fixed test set (orange). Right: number of 10 s sequences per postoperative moment (M0–M4) for the training split (average across folds), validation split (average across folds), and the held-out test set.

## VII. Results

### A. Dual-head classification results (primary)

On the held-out test animals (14 and 17; 35 sequences), the dual-head ensemble achieves strong binary pain detection, with **accuracy 0.9143** and **F1-score 0.9333**, and perfect recall (**1.0000**), indicating that all pain cases are detected. The 3-class intensity-moment task is substantially more challenging, with **accuracy 0.6000** and weighted **F1-score 0.5741**. The per-class report indicates high precision for the no-pain class (1.00), perfect recall for the acute-pain class (1.00), and low recall for the residual-pain class (0.17), suggesting confusion between late recovery and other moments.

Aggregate held-out test performance across tasks is reported in **Table IV**, and confusion matrices are shown in **Fig. 7**.

**Table IV.** Held-out test performance on unseen animals (Animals 14 and 17; N=35 sequences). Reported metrics include accuracy/precision/recall/F1 for binary pain classification, weighted accuracy/F1 for 3-class intensity-moment classification, and MAE/RMSE/\(R^2\)/Pearson correlation for Total Facial Scale regression. This table provides the primary summary of final generalization performance under the fixed held-out evaluation protocol.
| Task | Metric | Value |
|------|--------|-------|
| Binary pain classification | Accuracy | 0.9143 |
| Binary pain classification | F1 | 0.9333 |
| Binary pain classification | Precision | 0.8750 |
| Binary pain classification | Recall | 1.0000 |
| 3-class intensity-moment classification | Accuracy | 0.6000 |
| 3-class intensity-moment classification | F1 (weighted) | 0.5741 |
| 3-class intensity-moment classification | Precision (weighted) | 0.7338 |
| 3-class intensity-moment classification | Recall (weighted) | 0.6000 |
| Regression (Total Facial Scale) | MAE | 1.6527 |
| Regression (Total Facial Scale) | RMSE | 2.0205 |
| Regression (Total Facial Scale) | R² | 0.3125 |
| Regression (Total Facial Scale) | r (p-value) | 0.5696 (p=0.0004) |

**Fig. 7.** Confusion matrices on the held-out test animals (Animals 14 and 17; N=35 sequences). Cells report counts with row-normalized fractions in parentheses. (a) Binary pain/no-pain classification (no pain: M0/M1; pain: M2/M3/M4), emphasizing high sensitivity to pain with three false positives and no false negatives. (b) 3-class intensity-moment classification (no pain: M0/M1; acute pain: M2; residual pain: M3/M4), highlighting the dominant error mode in which residual-pain sequences are misclassified as acute pain.

### B. Regression results (supporting experiment)

On the held-out test animals (14 and 17; 35 sequences), the regression ensemble attains **MAE 1.6527**, **RMSE 2.0205**, and **R² 0.3125**, with Pearson correlation **r = 0.5696** (p = 0.0004). Fold-level test performance varies (best fold \(R^2 = 0.5322\), \(r = 0.7481\); worst fold \(R^2 = -0.2304\)), and the ensemble improves robustness relative to single folds, consistent with the benefits of model averaging under limited data.

The moment-wise test behavior is heterogeneous. Declining pain (M3) exhibits positive \(R^2\) (0.4209) and very strong correlation (r = 0.9620), while peak pain (M2) remains challenging and shows negative correlation (r = -0.5715) with negative \(R^2\). Feature-wise analysis indicates that orbital tightening and ear posture features are among the most reliably predicted components on the test set, supporting the feature-weighting rationale used in training.

Supporting regression visualizations (moment-wise behavior and feature-wise explained variance) are provided in **Fig. 8**.

**Fig. 8.** Supporting regression evaluation on the held-out test animals (Animals 14 and 17; N=35 sequences) for Total Facial Scale and its facial components. (a) Moment-wise mean absolute error (MAE; bars, left axis) and Pearson correlation (line, right axis) across postoperative moments (M0–M4), with sample counts annotated above bars. (b) Feature-wise coefficient of determination (\(R^2\)) on the held-out test set for seven facial components, shown with a zero-reference line to indicate features with negative explained variance. The figure highlights degraded agreement around peak pain (M2) and stronger predictability for orbital tightening and ear posture components.

## VIII. Discussion

Results on held-out animals indicate that temporal modeling of face-crop sequences can support robust binary pain detection, achieving high accuracy and F1-score with perfect recall. This suggests that the learned representations reliably capture salient pain-associated facial patterns despite variability across animals and recording conditions. In contrast, the 3-class intensity-moment task remains substantially more challenging, with particularly low recall for residual pain. This pattern is consistent with known challenges highlighted by recent surveys: late-stage recovery and intermediate affective states can be visually subtle, and facial indicators may overlap across adjacent postoperative moments \[3\]–\[5\].

From a pipeline perspective, the high face detection accuracy and stable crop generation provide a strong foundation for downstream learning. In practice, the choice between the higher-accuracy detector and the lighter detector can be guided by the operational constraints of the application. Where throughput and model size are critical (e.g., prospective edge deployment), the lightweight detector provides a strong accuracy–efficiency compromise, while the higher-accuracy detector is appropriate when maximizing localization quality is the priority.

The supporting regression results indicate that the efficient temporal backbone can also learn continuous targets tied to validated pain-scale components, achieving meaningful correlation and positive \(R^2\) on held-out animals. The observed moment-wise difficulties in regression—particularly around the acute postoperative period—motivate moment-weighted objectives and suggest that future work should prioritize improved discrimination around transitional and peak-pain regimes.

From an IoT systems perspective, the proposed workflow aligns with edge-first monitoring principles: compute-intensive perception is performed near the camera source, while only compact metadata need be transmitted for routine monitoring, improving resilience under bandwidth and reliability constraints discussed in smart-agriculture surveys \[10\] and livestock deployment contexts \[9\]. In addition, IoT architectures that combine wearable and environmental sensors with camera-based inference are increasingly recognized as necessary to move from proxy measurements toward affective-state monitoring \[8\], \[15\]–\[17\].

## IX. Limitations

Several limitations constrain the conclusions that can be drawn from the current experiments. First, dataset scale is limited in the number of animals and the distribution across postoperative moments; although animal-wise evaluation reduces leakage, it increases variance and can amplify class imbalance. Second, the intensity-moment categories compress multiple physiological states into a small label set, and some categories (notably residual pain) may exhibit heterogeneous facial patterns that are difficult to separate with limited training data. Third, results are derived from a single dataset and acquisition protocol; generalization to other farms, breeds, and camera setups remains to be established, as emphasized in prior surveys \[3\]–\[5\]. Fourth, while the pipeline is designed with efficiency considerations in mind, no dedicated edge-hardware deployment experiments are reported in this manuscript.

For IEEE IoT-J positioning, additional system-level evaluation is required beyond component accuracy: network overhead under concurrent camera streams, protocol configuration and reliability, and end-to-end latency breakdown across edge, fog, and cloud stages. Security and privacy protections are described architecturally, but are not experimentally evaluated; nevertheless, secure transport and access control are important given the protocol threat landscape in IoT deployments \[18\].

## X. Conclusion and Future Work

This work presents an end-to-end pipeline for bovine pain inference from facial video, combining face detection, standardized temporal sequence construction, and efficient temporal deep learning. Dual-head classification achieves strong binary pain detection on unseen animals, while intensity-moment classification remains challenging, particularly for residual pain discrimination. A supporting regression model further demonstrates that the same compact temporal backbone can predict validated facial action unit components and total pain scale scores.

Future work will focus on improving intensity-moment discrimination through data expansion and targeted modeling (e.g., calibration, better handling of residual pain subclasses, uncertainty-aware prediction), broader external validation across acquisition conditions, and system-level IoT evaluation. Recommended directions include profiling end-to-end latency and throughput on representative edge devices, estimating and benchmarking bandwidth reductions under metadata-first communication, evaluating scalability across multiple camera nodes, and incorporating security-by-design mechanisms for encrypted transport and access control consistent with IoT protocol/security guidance \[18\].

## References (peer-reviewed journals; published after 2020)

\[1\] T. Tschoner, K. R. Mueller, Y. Zablotski, and M. Feist, “Pain assessment in cattle by use of numerical rating and visual analogue scales—A systematic review and meta-analysis,” *Animals*, vol. 14, no. 2, Art. no. 351, 2024, doi: `10.3390/ani14020351`.

\[2\] M. C. Evangelista, B. P. Monteiro, and P. V. Steagall, “Measurement properties of grimace scales for pain assessment in nonhuman mammals: A systematic review,” *Pain*, vol. 163, pp. e697–e714, 2022, doi: `10.1097/j.pain.0000000000002474`.

\[3\] S. Broomé, M. Feighelstein, A. Zamansky, G. Carreira Lencioni, P. H. Andersen, F. Pessanha, M. Mahmoud, H. Kjellström, and A. A. Salah, “Going deeper than tracking: A survey of computer-vision based recognition of animal pain and emotions,” *International Journal of Computer Vision*, vol. 131, pp. 572–590, 2023, doi: `10.1007/s11263-022-01716-3`.

\[4\] C. Fischer-Tenhagen, J. Meier, and A. Pohl, “Do not look at me like that”: Is the facial expression score reliable and accurate to evaluate pain in large domestic animals? a systematic review,” *Frontiers in Veterinary Science*, vol. 9, Art. no. 1002681, 2022, doi: `10.3389/fvets.2022.1002681`.

\[5\] L. Chiavaccini, A. Gupta, and G. Chiavaccini, “From facial expressions to algorithms: A narrative review of animal pain recognition technologies,” *Frontiers in Veterinary Science*, vol. 11, Art. no. 1436795, 2024, doi: `10.3389/fvets.2024.1436795`.

\[6\] L. Ginger, L. Aubé, D. Ledoux, M. Borot, C. David, M. Bouchon, M. Leach, D. Durand, and A. D. B. des Roches, “A six-step process to explore facial expressions performances to detect pain in dairy cows with lipopolysaccharide-induced clinical mastitis,” *Applied Animal Behaviour Science*, vol. 264, Art. no. 105951, 2023, doi: `10.1016/j.applanim.2023.105951`.

\[7\] D. Ledoux, I. Veissier, B. Meunier, V. Gelin, C. Richard, H. Kiefer, H. Jammes, G. Foucras, and A. de Boyer Des Roches, “Combining accelerometers and direct visual observations to detect sickness and pain in cows of different ages submitted to systemic inflammation,” *Scientific Reports*, vol. 13, Art. no. 1977, 2023, doi: `10.1038/s41598-023-27884-x`.

\[8\] M. Zhang, X. Wang, H. Feng, Q. Huang, X. Xiao, and X. Zhang, “Wearable Internet of Things enabled precision livestock farming in smart farms: A review of technical solutions for precise perception, biocompatibility, and sustainability monitoring,” *Journal of Cleaner Production*, vol. 312, Art. no. 127712, 2021, doi: `10.1016/j.jclepro.2021.127712`.

\[9\] I. Bergier, M. Papa, R. Silva, and P. M. Santos, “Cloud/edge computing for compliance in the Brazilian livestock supply chain,” *Science of The Total Environment*, vol. 761, Art. no. 143276, 2021, doi: `10.1016/j.scitotenv.2020.143276`.

\[10\] Y. Kalyani and R. Collier, “A systematic survey on the role of cloud, fog, and edge computing combination in smart agriculture,” *Sensors*, vol. 21, no. 17, Art. no. 5922, 2021, doi: `10.3390/s21175922`.

\[11\] S. Zhang, K. Sailunaz, and S. Neethirajan, “Micro-Expression-Based Facial Analysis for Automated Pain Recognition in Dairy Cattle: An Early-Stage Evaluation,” *AI*, vol. 6, no. 9, Art. no. 199, 2025, doi: `10.3390/ai6090199`.

\[12\] R. M. Tomacheuski, A. R. Oliveira, P. H. E. Trindade, F. A. Oliveira, C. P. Candido, F. J. Teixeira Neto, P. V. Steagall, and S. P. L. Luna, “Reliability and validity of UNESP-Botucatu Cattle Pain Scale and Cow Pain Scale in *Bos taurus* and *Bos indicus* bulls to assess postoperative pain of surgical orchiectomy,” *Animals*, vol. 13, no. 3, Art. no. 364, 2023, doi: `10.3390/ani13030364`.

\[13\] W. Zhang, W. Wang, Y. Wang, S. Samat, and X. Chen, “An Improved YOLOv8n-Based Method for Multi-Object Individual Cattle Recognition Using Facial Features in Feeding Passages,” *Agriculture*, vol. 15, no. 24, Art. no. 2536, 2025, doi: `10.3390/agriculture15242536`.

\[14\] Z. Wu, Z. Zhang, L. Chen, Y. Xiao, Y. Gao, and T. Huang, “Benchmarking YOLOv8-v13 Architectures for Intelligent Real-Time Cattle Monitoring and Data-Driven Farm Management in Precision Livestock Farming,” *JoVE (Journal of Visualized Experiments)*, no. 225, 2025, doi: `10.3791/69490`.

\[15\] A. H. Stygar, Y. Gómez, G. V. Berteselli, E. Dalla Costa, E. Canali, J. K. Niemi, P. Llonch, and M. Pastell, “A Systematic Review on Commercially Available and Validated Sensor Technologies for Welfare Assessment of Dairy Cattle,” *Frontiers in Veterinary Science*, vol. 8, Art. no. 634338, 2021, doi: `10.3389/fvets.2021.634338`.

\[16\] F. M. Tangorra, E. Buoio, A. Calcante, A. Bassi, and A. Costa, “Internet of Things (IoT): Sensors Application in Dairy Cattle Farming,” *Animals*, vol. 14, no. 21, Art. no. 3071, 2024, doi: `10.3390/ani14213071`.

\[17\] S. Neethirajan, I. Reimert, and B. Kemp, “Measuring Farm Animal Emotions—Sensor-Based Approaches,” *Sensors*, vol. 21, no. 2, Art. no. 553, 2021, doi: `10.3390/s21020553`.

\[18\] A. Gerodimos, L. Maglaras, M. A. Ferrag, N. Ayres, and I. Kantzavelou, “IoT: Communication protocols and security threats,” *Internet of Things and Cyber-Physical Systems*, vol. 3, pp. 1–13, 2023, doi: `10.1016/j.iotcps.2022.12.003`.


