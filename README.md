# MERSA — Multimodal Emotion Recognition System for Older Adults

**INF2102 — Final Programming Project — PUC-Rio 2026.1 — 3WA**

## Author
Filza Javed | filzajaved@tecgraf.puc-rio.br

## Description
MERSA detects the emotional state of older adult users through multimodal 
analysis of facial expressions and voice, and generates adaptive interface 
recommendations accordingly. It is designed for older adults aged 60+ and 
is part of ongoing PhD research on emotion-aware intelligent interfaces.

## Requirements
- Python 3.11
- DeepFace
- OpenCV
- PyAudio
- librosa
- soundfile
- scikit-learn
- joblib

## Installation
All data is stored locally — no personal data sent to external servers.
Future user studies will be submitted to Plataforma Brasil (CEP/CONEP).

## How it Works
MERSA uses a three-stage pipeline:
1. **Face Analysis** DeepFace detects facial landmarks and classifies emotion into 7 categories (happy, sad, angry, fear, surprise, neutral, disgust)
2. **Voice Analysis** librosa extracts acoustic features (energy, pitch, zero-crossing rate, MFCCs, chroma, mel-spectrogram) from a 5-second microphone recording, classified using a trained Random Forest model (see Model Comparison below)
3. **Fusion** — combines face (60%) and voice (40%) scores for a final emotion decision

### Face Module
Tested on 140 real, labeled images from the official FER2013 "PrivateTest" partition (20 images per emotion, 7 emotions).

| Model | Accuracy | How Measured |
|-------|----------|--------------|
| **DeepFace (used in MERSA)** | **52.86%** (74/140) | Run live on the full test set |
| Alternative CNN (gitshanks/fer2013) | 37.86% (53/140) | Run live on the same test set, for comparison |

DeepFace outperforms the comparison model on this test set. Full per-class precision/recall/specificity and confusion matrices are available in the project report.

### Voice Module
Tested on the complete official RAVDESS speech set (1,440 files, 24 actors, 8 emotions).

| Version | Accuracy | Method |
|---------|----------|--------|
| v1 (original) | 13.89% (200/1,440) | Fixed rule-based thresholds on 3 features |
| **v2 (current)** | **40.00%** (120/300, held-out actors) | Trained Random Forest on 156 features, actor-independent test split |

The original rule-based voice module performed close to chance level (~14.3% for 7 classes). It was replaced with a trained classifier, nearly tripling accuracy. Full methodology and per-class results are available in the project report.
## Data & Privacy
All data is processed and stored locally — nothing is transmitted externally. 
This prototype uses only open datasets (FER2013, RAVDESS); direct testing 
with real users will require prior approval from Plataforma Brasil.
