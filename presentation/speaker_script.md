## Speaker script (8–9 minutes, 9 slides)

**How to use this:** read it once, then rehearse with a timer. The phrasing is intentionally short and “sayable.”  
**Target total:** ~8:30 (matches `presentation/deck_outline.md` timing).

---

## Slide 1 — Title + who I am + why this matters (0:55)

Hi everyone—my name is **Shivam Patel**. I’m an **MCS Computer Science** student at **Dalhousie University**, and I’m in my **[YEAR]**.  
Today I’ll share my project on **bovine pain monitoring from facial video**.

Quick motivation: pain and welfare issues are not only an ethical problem—they’re also expensive.  
For example, one study estimated average costs per lameness case around **\$121–\$216 per case**, depending on the diagnosis.  

The core problem is: on-farm pain assessment is often **intermittent**, **subjective**, and hard to do at **scale**.  
So my goal is to build a practical pipeline that can turn **video into a pain alert**, in a way that’s compatible with smart/digital agriculture.

---

## Slide 2 — Ground reality + knowledge gap (0:55)

The dataset comes from the UNESP‑Botucatu cattle pain scale study, with postoperative measurements across moments M0 to M4.  
For each animal and moment, there are facial action unit scores and a Total Facial Scale, which represent pain-related facial indicators.

The model input is not a single image. It is a short sequence of face crops taken from video.

There are two main classification targets:
- Binary pain detection: pain vs no-pain
- Three-class intensity–moment: no pain, acute pain, and residual pain

This slide’s figure shows how pain indicators change across moments, especially around the acute pain window.



---

## Slide 3 — Dataset & labels (Brazil collaboration; what “pain” means here) (0:55)

Here is the full workflow shown left to right.

What goes in is facial video for a specific animal and moment.  
The pipeline outputs pain decisions: binary pain/no-pain, a 3-class stage, and optionally continuous pain-score support.

The most important point is consistency.  
First the pipeline creates clean and consistent face-crop sequences.  
Then the temporal model runs on those sequences.

If the face crops are unstable—wrong region, occlusions, or jitter—the temporal model will learn noise, and accuracy will drop.


## Slide 4 — System pipeline overview (0:55)

The first technical step is face detection in each frame, because the model should focus on the facial region only.

Two detectors were trained:
- YOLOv11s for higher accuracy
- YOLOv8n as a lighter, more efficient option

On the detection test set, YOLOv11s achieves mAP@0.5 = 0.994, with strong precision and recall.  
YOLOv8n is slightly lower at mAP@0.5 = 0.989, but it is much smaller—3.0M parameters vs 9.4M—and faster per image.

So the takeaway is: YOLOv11s is the accuracy-first option for the cleanest crops, and YOLOv8n is the efficiency-first option when compute is limited.



## Slide 5 — Face detection (YOLO) for robust cropping (0:55)

After detection, each sequence is a fixed number of face crops over time.

The temporal model has three parts:
1) a compact CNN that extracts a feature vector from each frame,  
2) an LSTM that models the time sequence, and  
3) an attention layer that focuses on the most informative frames.

The main model is a dual-head classifier.  
One head predicts binary pain/no-pain.  
The second head predicts the 3-class intensity–moment stage.

This design gives two clinically interpretable outputs from the same backbone, which is useful for decision support.



---

## Slide 6 — Temporal learning (core algorithm) (1:00)

Results are reported using held-out test animals to evaluate generalization to unseen individuals.

For binary pain detection, performance is strong:
- Accuracy 0.9143
- F1-score 0.9333
- Recall 1.0000

For the 3-class intensity–moment task, the problem is harder and accuracy is 0.6000.  
This is expected because classes are closer to each other: acute vs residual pain can look subtle in short clips.

A supporting regression model predicts facial action units and Total Facial Scale.  
For Total Facial Scale, the model achieves MAE 1.6527, (R^2 = 0.3125), and correlation (r = 0.5696).

So the practical takeaway today is: this pipeline is already strong as a **pain/no‑pain alert**, and the next push is improving staging and continuous scoring.

## Slide 8 — Deployment (IoT) + Security (must slide) (0:55)

In practice, the value is decision support for farmers and veterinarians:
- earlier detection of pain can trigger faster intervention and better welfare outcomes,
- automated monitoring reduces labor and makes monitoring more scalable,
- and the output can integrate into farm dashboards or herd-management software.

The long-term goal is not to replace veterinarians, but to provide consistent, repeatable monitoring that flags animals for attention.



## Slide 8 — Deployment (IoT) + Security (must slide) (0:55)

Digital livestock systems face real cybersecurity threats—ransomware, insecure IoT devices, weak authentication, and cloud misconfiguration.  
In livestock settings, a cyber incident can also become an animal welfare problem because feeding and care routines cannot pause.

So secure data handling needs to be part of the design:
- Data minimization: send only compact predictions and metadata; share video only when needed.
- Encryption: TLS for data in transit; encryption at rest for stored clips and labels.
- Access control: device identity, role-based access control, and multi-factor authentication for dashboards.
- Resilience: backups, logging, and an incident response plan following Identify–Protect–Detect–Respond–Recover.

These points align with the 2025 Frontiers cybersecurity roadmap for digital livestock farming.



## Slide 9 — Green AI / Sustainable AI (must slide) (0:55)

Green AI is about delivering the same welfare benefit with less compute, less energy, and less bandwidth.

This project already makes some frugal choices:
- using an efficient detector option YOLOv8n,  
- using a compact temporal backbone,  
- fixed-length sampling instead of processing full-length videos continuously,  
- and metadata-first outputs rather than streaming raw video.

Next steps that can be measured and reported:
- quantization and possibly pruning/distillation for edge deployment,
- report Joules per inference and estimated CO(_2)e, alongside accuracy,
- and do system-level accounting: compare compute emissions to the welfare and operational benefits.

That frames the system as not only accurate, but also responsible and deployable.



---

**Motivation stat reference for Slide 1 (on-slide small text):** Cha et al., *Preventive Veterinary Medicine*, 2010 (PMID: 20801533).


