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
1. **Face Analysis** — DeepFace detects facial landmarks and classifies emotion into 7 categories (happy, sad, angry, fear, surprise, neutral, disgust)
2. **Voice Analysis** — librosa extracts acoustic features (energy, pitch, zero-crossing rate, MFCCs, chroma, mel-spectrogram) from a 5-second microphone recording, classified using a trained Random Forest model (see Model Comparison below)
3. **Fusion** — combines face (60%) and voice (40%) scores for a final emotion decision

## Adaptive Interface Recommendations
Based on the detected emotion, MERSA suggests:
- Font size adjustment
- Colour theme change
- Help prompt activation

## Session Logging
Every analysis is saved to `mersa_session_log.json` for longitudinal wellbeing monitoring. All data is processed and stored locally — no facial images, audio recordings, or derived emotion data are transmitted to external servers, cloud services, or third parties. Session logs use anonymous session identifiers only; no personally identifying metadata is collected or linked.

## Test Results

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

## Ethical Aspects and Data Collection
MERSA processes two types of sensitive personal data during operation: facial images (via webcam) and voice recordings (via microphone). Both are collected only during active use of the application, with the user's knowledge and direct involvement.

All data is processed and stored **locally on the user's own device**. No facial images, audio recordings, or derived emotion data are transmitted externally at any point in the current prototype. This local-first design minimizes privacy risk, given the sensitivity of biometric data and the vulnerability of the target population (older adults).

The current version is a technical proof-of-concept, validated using open, publicly available datasets (FER2013, RAVDESS) rather than data from human participants. As this research moves toward direct testing with real users — particularly older adults and, as part of my broader PhD research, individuals with Alzheimer's disease or other forms of dementia — I will submit the project for review and approval to **Plataforma Brasil**, the Brazilian national platform for the ethical review of research involving human subjects, before any such data collection.

## Future Work
- Real-time camera integration
- Further improve the voice classifier (data augmentation, larger training set, deep learning architecture)
- Android mobile application
- Behavioural modality (touch patterns, navigation)
- Validation with the target population (older adults, dementia patients) following ethics approval

## License
MIT License
