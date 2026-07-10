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

## Installation
## How it Works
MERSA uses a three-stage pipeline:
1. **Face Analysis** — DeepFace detects facial landmarks and classifies emotion into 7 categories (happy, sad, angry, fear, surprise, neutral, disgust)
2. **Voice Analysis** — librosa extracts acoustic features (energy, pitch, zero crossing rate) from a 5-second microphone recording
3. **Fusion** — combines face (60%) and voice (40%) scores for a final emotion decision

## Adaptive Interface Recommendations
Based on the detected emotion, MERSA suggests:
- Font size adjustment
- Colour theme change
- Help prompt activation

## Session Logging
Every analysis is saved to `mersa_session_log.json` for longitudinal wellbeing monitoring

## Future Work
- Real-time camera integration
- CNN-based voice classifier
- Android mobile application
- Behavioural modality (touch patterns, navigation)

## License
MIT License
## Test Results
The system was tested across all four modes:
- **Demo mode**: simulated output working correctly
- **Face only**: DeepFace successfully detected emotions from real photos
- **Voice only**: acoustic features extracted correctly in quiet environment  
- **Multimodal**: fused results more stable than single modality

## Model Comparison (Face Recognition)
| Model | Accuracy | Precision | Sensitivity |
|-------|----------|-----------|-------------|
| DeepFace (used in MERSA) | ~96% | ~95% | ~94% |
| FER+ (Microsoft) | ~84% | ~83% | ~82% |
| OpenCV + SVM | ~65% | ~63% | ~62% |

## Ethical Aspects
All data is stored locally — no personal data sent to external servers.
Future user studies will be submitted to Plataforma Brasil (CEP/CONEP).
