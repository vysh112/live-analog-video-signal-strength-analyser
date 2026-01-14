# FPV Video Signal Quality Estimation Framework

## Overview

This repository presents a **video-based signal quality estimation framework** designed for **analog FPV (First Person View) video links**, commonly used in UAVs, ground robots, and teleoperation systems.

Unlike digital communication systems that expose explicit metrics such as RSSI or SNR, **analog FPV links degrade gradually** and express link quality implicitly through **visual artifacts**. This project introduces a **non-invasive, vision-based method** to quantify link quality directly from recorded or live video streams.

The framework computes a **continuous signal strength score (0–100%)** derived from multiple perceptual and statistical image metrics and supports:

- Real-time on-screen signal strength estimation  
- Comparative analysis between two simultaneous FPV links  
- Visualization of degradation behavior over time  

This work is intended for **research, benchmarking, and experimental validation**, and is suitable for academic citation.

---

## Repository Structure

```
fpv-video-signal-quality-estimation/
│
├── video_strength_OSD.py
│   └── Real-time FPV signal strength estimation with on-screen display
│
├── video_strength_matplotlib.py
│   └── Dual-video comparative analysis with live and final plots
│
├── requirements.txt
└── README.md
```

---

## Installation & Requirements

### Python Version
- Python **3.8+** recommended

### Required Libraries

```bash
pip install opencv-python numpy matplotlib
```

---

## How to Run

### 1. Single Video Signal Strength (On-Screen Display)

Edit the video path inside `video_strength_OSD.py`:

```python
video_path = "path/to/your/video.ts"
```

Run:
```bash
python video_strength_OSD.py
```

Controls:
- `q` → Quit  
- `1–5` → Jump to predefined timestamps  

---

### 2. Dual Video Comparative Analysis

Edit both video paths in `video_strength_matplotlib.py`:

```python
video1 = "path/to/video1.ts"
video2 = "path/to/video2.ts"
```

Run:
```bash
python video_strength_matplotlib.py
```

This launches:
- A side-by-side FPV preview  
- Live plots of all quality metrics  
- Final comparative plots after playback ends  

---

## Methodology and Signal Strength Model

### Conceptual Foundation

Analog FPV degradation manifests through:
- Increased noise
- Reduced contrast
- Loss of color information
- Line interference
- Flicker
- Entropy collapse or explosion during static

This framework quantifies these effects per-frame and fuses them into a **single normalized signal strength indicator**.

---

## Computed Metrics (Per Frame)

### 1. Noise Level (High-Frequency Artifacts)

Computed as the variance of the Laplacian of the grayscale image.  
Captures salt-and-pepper noise and RF interference.

---

### 2. Contrast Score

Measured as the standard deviation of grayscale pixel intensities.  
Low contrast indicates washout or heavy noise.

---

### 3. Color Saturation

Computed as the mean saturation channel value in HSV color space.  
Analog FPV color loss strongly correlates with weak signal reception.

---

### 4. Line Interference (FFT-Based)

Detected using frequency-domain analysis to identify dominant horizontal and vertical interference patterns.

---

### 5. Flicker Index

Calculated as the mean absolute difference between consecutive frames.  
Captures temporal instability and brightness oscillations.

---

### 6. Entropy Score

Based on Shannon entropy of the grayscale histogram.  
Measures information richness and helps detect static or noise-dominated frames.

---

## Signal Strength Aggregation

Each metric is:
1. Clamped to empirically selected bounds  
2. Normalized to a 0–100 scale  
3. Inverted where lower values indicate better quality  
4. Weighted based on perceptual relevance  

### Weighted Fusion Model

```
Signal Strength =
  0.20 × Noise Score
+ 0.20 × Contrast Score
+ 0.15 × Saturation Score
+ 0.15 × Line Interference Score
+ 0.15 × Flicker Score
+ 0.15 × Entropy Score
```

A fail-safe condition caps the signal strength during near-total feed loss (grey static detection).

---

## Script-Specific Purpose

### video_strength_OSD.py

Provides **real-time visual feedback** of FPV signal quality directly on the video feed using traffic-light color coding and metric overlays.

**Use cases:**
- Field testing  
- Operator feedback  
- Qualitative validation  

---

### video_strength_matplotlib.py

Enables **quantitative comparison** between two FPV links with synchronized metric plots and final publication-ready graphs.

**Use cases:**
- Thesis experiments  
- Relay vs direct link comparison  
- Antenna and transmission benchmarking  

---

## Applications

- Analog FPV relay evaluation  
- UAV communication robustness analysis  
- Disaster robotics teleoperation  
- Human-in-the-loop navigation under degraded links  
- Post-flight RF performance analysis  

---

## Academic Relevance

This framework demonstrates that **visual degradation artifacts can act as a proxy for RF link quality**, enabling signal estimation without hardware telemetry. It is particularly relevant for low-cost or legacy analog FPV systems.

---

## Citation Recommendation

If referencing this repository in academic work:

“A video-based FPV signal quality estimation framework was implemented using multi-metric perceptual analysis, fusing noise, contrast, saturation, entropy, flicker, and frequency-domain interference into a normalized signal strength index.”
