## 8–9 minute presentation deck (9 slides total)

**Constraints (as provided):** 8–9 minutes strict • 8–9 slides total • no thank-you slide • no table of contents • jump in directly.  
**Style:** one key message per slide • ≤5 bullets/slide • font ≥24 pt • prefer diagrams/figures over text.

---

## Slide 1 — Title + who I am + why this matters (0:55)

- **Introduce yourself (10 s)**: Shivam Patel • MCS Computer Science • Dalhousie University • *(Year: fill in)*.
- **Motivation (1 data point)**: pain-related conditions are costly; e.g., lameness case costs reported around **\$121–\$216 per case** (Prev. Vet. Med., 2010).
- **Problem**: on-farm pain monitoring is **intermittent**, **subjective**, and **hard to scale**.
- **Goal**: build an automated **facial‑video** pain monitoring pipeline for smart/digital agriculture.

**Visual suggestion (non‑AI):** one real high‑res photo/frame of a cow + big hook text: “Pain is expensive — and hard to measure at scale.”  
**Talk track (0:50–1:00):** Brief intro, then 2 sentences on why pain measurement matters (welfare + economics), then state your one‑line project goal.

---

## Slide 2 — Ground reality + knowledge gap (why cameras/AI) (0:55)

- **Today’s reality**: pain scoring often needs trained observers; it’s not continuous, and it’s hard to do for every animal.
- **Knowledge gap**: we need objective, repeatable pain cues that can be measured continuously with low friction.
- **Why face video**: cameras are cheap; facial indicators are non-contact and can be captured during routine handling.
- **What we measure (simple)**: eye/ear/face tension changes over time → the model learns patterns from short clips.

**Visual suggestion (PowerPoint-built, non‑AI):**
- Use a **real cow face frame** (from your dataset) and overlay **3–5 callouts** with arrows: “eyes (orbital tightening)”, “ears (posture)”, “muzzle/nostrils”, “head position”.
- Add one small caption: “Facial cues are part of validated pain scales.”
**Talk track (0:50–1:00):** Slowly point to the callouts and say “this is what the model is trying to learn.”

---

## Slide 3 — System pipeline overview (end-to-end workflow)

- **What goes in**: facial video clips (per animal, per moment).
- **What comes out**: pain decisions (binary pain/no‑pain and 3‑class stage) + optional continuous pain score support.
- **Key idea**: first make clean, consistent face-crop sequences; then run the temporal model (bad crops → bad predictions).

**Visual suggestion:** single diagram (Video → YOLO face boxes → face-crop sequence → CNN+LSTM+Attn → {Binary, 3-class, Regression}).  
**Talk track (0:50–1:00):** Point to the diagram left→right in one sweep, then use the bullets to emphasize standardization and error propagation (why the pipeline design is critical).

---

## Slide 4 — Dataset & labels (Brazil collaboration; what “pain” means here) (0:55)

- **Where data comes from**: UNESP‑Botucatu (Brazil) postoperative pain study in bulls (*Bos taurus* and *Bos indicus*).
- **Moments**: M0–M4 (baseline → acute pain → recovery trend).
- **Model inputs**: facial video → fixed-length **face‑crop sequences**.
- **Targets (keep simple)**:
  - **Binary**: no‑pain vs pain
  - **3‑class stage**: no pain / acute pain / residual pain

**Visual suggestion (use your existing figure):** insert `paper/figures/exports/fig1_moment_trajectory_ieee.pdf` (or PNG export) to show the M2 peak visually.  
**Talk track (0:50–1:00):** Explain M0–M4 slowly and connect M2 to “acute pain” in one clear sentence.

---

## Slide 5 — Face detection (YOLO) for robust cropping (0:55)

- **Step**: detect bovine face in each frame → stable crop → sequence construction.
- **Detectors trained**: **YOLOv8n** (edge‑efficient) and **YOLOv11s** (higher accuracy).
- **Reason for two models**: accuracy–efficiency trade-off for future edge deployments.

**Visual suggestion (non‑AI):**
- Left: 1–2 real frames with **YOLO face boxes** (export from your own detections; crisp).
- Right: a **PowerPoint table** (not an image) with the key Table II numbers (mAP@0.5, params, model size, ms/img) for YOLOv11s vs YOLOv8n.
**Talk track (0:50–1:00):** Emphasize: “This step is boring but essential—bad crops create bad learning.”

---

## Slide 6 — Temporal learning (core algorithm) (1:00)

- **Input**: \(T\) face crops (fixed-length sequence).
- **Backbone**: efficient CNN feature extractor + LSTM + attention (temporal aggregation).
- **Dual-head classifier**:
  - Head A: binary pain vs no‑pain
  - Head B: 3‑class stage (no / acute / residual)

**Visual suggestion (PowerPoint-built):** simple block diagram (Sequence → CNN → LSTM → Attention → {Binary head, 3‑class head}).  
**Talk track (0:55–1:05):** Walk through it step-by-step, using plain language (“we compress each frame, then learn how it changes over time”).

---

## Slide 7 — Results + practical takeaway (1:10)

- **Held-out test animals**: binary pain detection is **strong** (Acc **0.9143**, F1 **0.9333**, Recall **1.0000**).
- **3‑class intensity–moment**: harder (Acc **0.6000**) → class overlap + subtle facial differences.
- **Regression (supporting)**: Total Facial Scale MAE **1.6527**, \(R^2=0.3125\), \(r=0.5696\).
- **So what?**: the system is best today as a **pain/no‑pain alert**; staging intensity is the next improvement target.
- **Impact (1 line)**: practical decision support → faster intervention and scalable monitoring.

**Visual suggestion:** insert `paper/figures/exports/fig7_confusion_matrices_v2_5.png` (left) + `paper/figures/exports/fig8_regression_summary_v2_4.png` (right).  
**Talk track (1:05–1:15):** One sentence interpreting each metric; be honest that 3‑class + regression are harder but meaningful.

---

## Slide 8 — Deployment (IoT) + Security (must slide) (0:55)

- **Simple IoT flow**: camera → edge box (runs model) → dashboard/cloud (alerts + history).
- **What we send**: mostly **small predictions + timestamps**; upload video only when needed.
- **Security (plain language)**:
  - **Encrypt data in transit** (TLS) and **encrypt stored data** (at rest)
  - **Login + access control** (only authorized users/devices can view data)
  - **Keep less data** (short local retention; share video only for low-confidence / audits)
  - **Logging + backups** (so incidents can be investigated and recovered)

**Visual suggestion (non‑AI, PowerPoint-built):** a 3-box diagram (Camera → Edge → Cloud/Dashboard) with a small **lock icon** on the arrows and a small callout “metadata-first; video on-demand”.  
**Reference (small footer):** Neethirajan, *Frontiers in Big Data*, 2025 (cybersecurity roadmap for digital livestock).  
**Talk track (0:50–1:00):** One line on the IoT flow, then 3–4 simple security actions; keep it practical (“what we would do in the app”).

---

## Slide 9 — Green AI / Sustainable AI (must slide) (0:55)

**Keep it simple: “same welfare benefit, less compute.”**

- **What we already do**:
  - Use a **smaller model option** (YOLOv8n) when resources are limited
  - Process **short clips** (not full videos all the time)
  - Send **predictions, not streams** (metadata-first communication)
- **What we would do next**:
  - Make models smaller/faster (e.g., **INT8 quantization**)
  - Report **energy per inference** (and estimate CO\(_2\)e) alongside accuracy
  - Run inference only when the **face crop quality is good** (avoid wasted compute)

**Nice visual for Green AI (non‑AI, PowerPoint-built):**
- A **“Compute & Bandwidth Budget”** slide with 3 big icons + arrows:
  - **Compute ↓** (smaller model, quantization)
  - **Bandwidth ↓** (metadata-first; video on-demand)
  - **Carbon ↓** (measure joules/inference → estimate CO\(_2\)e)
- On the right, a tiny **2-row mini-table**: YOLOv11s vs YOLOv8n (**params**, **ms/img**, **model size**) to show the efficiency trade-off.

**References (on-slide, short):** illuminem “Green AI arithmetic…”; livestock AI energy standardization roadmap (SSRN); Green AI for livestock perception (TechRxiv); sustainable digital livestock computing (Preprints/SSRN).  
**Talk track (0:50–1:00):** Give 2–3 measurable commitments: quantize, measure, report energy/CO\(_2\)e.

---

## Rehearsal timing target (8:30 total)

- Slide 1: 0:55  
- Slide 2: 0:55  
- Slide 3: 0:55  
- Slide 4: 0:55  
- Slide 5: 0:55  
- Slide 6: 1:00  
- Slide 7: 1:10  
- Slide 8: 0:55  
- Slide 9: 0:55  

**Tip:** rehearse once at 9:00, then cut 1 sentence per slide until you consistently hit 8:30.


