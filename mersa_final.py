"""
MERSA - Multimodal Emotion Recognition System for Older Adults
INF2102 - Final Programming Project - 2026.1 - 3WA
Author: Filza Javed | PUC-Rio

Modes:
    --demo                        Simulated multimodal output
    --image photo.jpg             Face only analysis
    --voice                       Voice only analysis
    --image photo.jpg --voice     Face + Voice combined (multimodal fusion)

Requirements:
    pip install deepface opencv-python pyaudio librosa soundfile
"""

import argparse
import sys
import os
import json
import time
import wave
from datetime import datetime

ADAPTATIONS = {
    "happy": {
        "description": "User appears happy and engaged.",
        "adaptation": "Maintain current interface. Display rich content and enable advanced features.",
        "color_theme": "Standard", "font_size": "Normal", "help_prompt": False
    },
    "sad": {
        "description": "User appears sad or low in mood.",
        "adaptation": "Switch to warm color theme. Simplify interface. Display wellbeing check-in.",
        "color_theme": "Warm (soft yellows and oranges)", "font_size": "Large", "help_prompt": True
    },
    "angry": {
        "description": "User appears frustrated or angry.",
        "adaptation": "Enlarge buttons. Reduce visual clutter. Display step-by-step help guide.",
        "color_theme": "Calm (soft blues and greens)", "font_size": "Large", "help_prompt": True
    },
    "fear": {
        "description": "User appears anxious or fearful.",
        "adaptation": "Simplify interface significantly. Display reassuring messages.",
        "color_theme": "Calm (soft blues and greens)", "font_size": "Extra Large", "help_prompt": True
    },
    "surprise": {
        "description": "User appears surprised.",
        "adaptation": "Pause animations. Display brief explanation of what happened.",
        "color_theme": "Standard", "font_size": "Normal", "help_prompt": False
    },
    "neutral": {
        "description": "User appears calm and neutral.",
        "adaptation": "Maintain current interface. No adaptation required.",
        "color_theme": "Standard", "font_size": "Normal", "help_prompt": False
    },
    "disgust": {
        "description": "User appears uncomfortable or displeased.",
        "adaptation": "Switch to minimal interface. Offer to restart current task.",
        "color_theme": "Minimal (white background, dark text)", "font_size": "Large", "help_prompt": True
    }
}

EMOTION_WEIGHTS = {
    "angry": 1.3, "sad": 1.2, "fear": 1.2, "disgust": 1.1,
    "happy": 1.0, "surprise": 1.0, "neutral": 0.9
}


# ── FACE ANALYSIS ─────────────────────────────────────────────────────────────
def analyze_face(image_path):
    try:
        from deepface import DeepFace
        print("\n  [FACE] Analyzing facial expression using DeepFace AI...")
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=True,
            silent=True
        )
        if isinstance(result, list):
            result = result[0]
        return {
            "success": True,
            "dominant_emotion": result["dominant_emotion"],
            "emotion_scores": {k: float(v) for k, v in result["emotion"].items()}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── VOICE RECORDING ───────────────────────────────────────────────────────────
def record_voice(duration=5, sample_rate=22050):
    try:
        import pyaudio
        print(f"\n  [VOICE] Get ready to speak for {duration} seconds...")
        for i in range(3, 0, -1):
            print(f"  Starting in {i}...", end="\r")
            time.sleep(1)
        print("  Recording NOW! Speak clearly...          ")

        chunk = 1024
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1,
            rate=sample_rate, input=True, frames_per_buffer=chunk
        )
        frames = []
        for _ in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        temp_path = "mersa_temp_voice.wav"
        wf = wave.open(temp_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("  [VOICE] Recording complete!")
        return {"success": True, "audio_path": temp_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── VOICE ANALYSIS ────────────────────────────────────────────────────────────
def analyze_voice(audio_path):
    try:
        import librosa
        import numpy as np
        print("\n  [VOICE] Extracting acoustic features...")

        y, sr = librosa.load(audio_path, sr=22050)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        print(f"  [VOICE] Energy={rms:.4f} | Pitch={mean_pitch:.1f}Hz | ZCR={zcr:.4f}")

        scores = {"angry": 0.0, "sad": 0.0, "happy": 0.0, "neutral": 0.0,
                  "fear": 0.0, "surprise": 0.0, "disgust": 0.0}

        if rms > 0.05 and mean_pitch > 200:
            scores = {"angry": 55.0, "surprise": 20.0, "happy": 10.0,
                      "neutral": 10.0, "fear": 3.0, "sad": 1.0, "disgust": 1.0}
        elif rms < 0.02 and mean_pitch < 150:
            scores = {"sad": 50.0, "neutral": 30.0, "fear": 10.0,
                      "disgust": 5.0, "angry": 3.0, "happy": 1.0, "surprise": 1.0}
        elif rms > 0.03 and zcr > 0.08:
            scores = {"happy": 50.0, "surprise": 25.0, "neutral": 15.0,
                      "angry": 5.0, "fear": 3.0, "sad": 1.0, "disgust": 1.0}
        elif rms < 0.03 and mean_pitch > 180:
            scores = {"fear": 45.0, "sad": 25.0, "neutral": 15.0,
                      "surprise": 10.0, "angry": 3.0, "happy": 1.0, "disgust": 1.0}
        else:
            scores = {"neutral": 50.0, "sad": 20.0, "happy": 15.0,
                      "angry": 8.0, "fear": 4.0, "surprise": 2.0, "disgust": 1.0}

        dominant = max(scores, key=scores.get)
        return {
            "success": True,
            "dominant_emotion": dominant,
            "emotion_scores": scores,
            "features": {
                "energy": round(rms, 4),
                "pitch_hz": round(mean_pitch, 1),
                "zero_crossing_rate": round(zcr, 4)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── MULTIMODAL FUSION ─────────────────────────────────────────────────────────
def fuse_emotions(face_result, voice_result):
    emotions = ["angry", "sad", "happy", "neutral", "fear", "surprise", "disgust"]
    fused = {}
    for emotion in emotions:
        face_score = face_result["emotion_scores"].get(emotion, 0)
        voice_score = voice_result["emotion_scores"].get(emotion, 0)
        fused[emotion] = round(((face_score * 0.60) + (voice_score * 0.40)) * EMOTION_WEIGHTS.get(emotion, 1.0), 2)
    total = sum(fused.values())
    if total > 0:
        fused = {k: round((v / total) * 100, 2) for k, v in fused.items()}
    return max(fused, key=fused.get), fused


# ── PRINT SCORES ──────────────────────────────────────────────────────────────
def print_scores(scores):
    for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score / 5)
        print(f"    {emotion:<10} {score:5.1f}%  {bar}")


# ── PRINT ADAPTATION ──────────────────────────────────────────────────────────
def print_adaptation(dominant):
    adaptation = ADAPTATIONS.get(dominant, ADAPTATIONS["neutral"])
    print(f"\n" + "-" * 65)
    print(f"  ADAPTIVE INTERFACE RESPONSE:")
    print(f"  Final Emotion  : {dominant.upper()}")
    print(f"  Description    : {adaptation['description']}")
    print(f"  Recommendation : {adaptation['adaptation']}")
    print(f"  Color Theme    : {adaptation['color_theme']}")
    print(f"  Font Size      : {adaptation['font_size']}")
    print(f"  Help Prompt    : {'YES — display contextual help guide' if adaptation['help_prompt'] else 'NO'}")
    return adaptation


# ── SAVE LOG ──────────────────────────────────────────────────────────────────
def save_log(entry):
    log_path = "mersa_session_log.json"
    logs = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    logs.append(entry)
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"\n  Session logged to: {log_path} (Total: {len(logs)})")
    print("=" * 65)


# ── FACE ONLY MODE ────────────────────────────────────────────────────────────
def mode_face_only(image_path):
    print("\n  Mode: FACE ONLY")
    face = analyze_face(image_path)

    print("\n" + "=" * 65)
    print("  MERSA — Face-Only Emotion Report")
    print("=" * 65)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Image     : {os.path.basename(image_path)}")
    print("-" * 65)

    if not face["success"]:
        print(f"\n  [ERROR] {face['error']}")
        print("=" * 65)
        return

    print(f"\n  ── FACE ANALYSIS ──────────────────────────────────────────")
    print(f"  DETECTED EMOTION: {face['dominant_emotion'].upper()}")
    print_scores(face["emotion_scores"])
    adaptation = print_adaptation(face["dominant_emotion"])
    save_log({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "face_only",
        "image": os.path.basename(image_path),
        "detected_emotion": face["dominant_emotion"],
        "adaptation": adaptation["adaptation"]
    })


# ── VOICE ONLY MODE ───────────────────────────────────────────────────────────
def mode_voice_only():
    print("\n  Mode: VOICE ONLY")
    rec = record_voice(duration=5)

    print("\n" + "=" * 65)
    print("  MERSA — Voice-Only Emotion Report")
    print("=" * 65)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modality  : Voice only")
    print("-" * 65)

    if not rec["success"]:
        print(f"\n  [ERROR] Recording failed: {rec['error']}")
        print("=" * 65)
        return

    voice = analyze_voice(rec["audio_path"])
    if os.path.exists(rec["audio_path"]):
        os.remove(rec["audio_path"])

    if not voice["success"]:
        print(f"\n  [ERROR] {voice['error']}")
        print("=" * 65)
        return

    print(f"\n  ── VOICE ANALYSIS ─────────────────────────────────────────")
    print(f"  DETECTED EMOTION: {voice['dominant_emotion'].upper()}")
    if "features" in voice:
        f = voice["features"]
        print(f"  Acoustic Features:")
        print(f"    Energy (loudness) : {f['energy']}")
        print(f"    Pitch             : {f['pitch_hz']} Hz")
        print(f"    Zero Crossing Rate: {f['zero_crossing_rate']}")
    print_scores(voice["emotion_scores"])
    adaptation = print_adaptation(voice["dominant_emotion"])
    save_log({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "voice_only",
        "detected_emotion": voice["dominant_emotion"],
        "acoustic_features": voice.get("features", {}),
        "adaptation": adaptation["adaptation"]
    })


# ── FACE + VOICE COMBINED MODE ────────────────────────────────────────────────
def mode_combined(image_path):
    print("\n  Mode: FACE + VOICE COMBINED (Multimodal Fusion)")
    face = analyze_face(image_path)
    rec = record_voice(duration=5)

    print("\n" + "=" * 65)
    print("  MERSA — Multimodal Emotion Report (Face + Voice)")
    print("=" * 65)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Image     : {os.path.basename(image_path)}")
    print(f"  Modality  : Face + Voice")
    print("-" * 65)

    if not face["success"]:
        print(f"\n  [FACE ERROR] {face['error']}")
        print("=" * 65)
        return

    voice = None
    if rec["success"]:
        voice = analyze_voice(rec["audio_path"])
        if os.path.exists(rec["audio_path"]):
            os.remove(rec["audio_path"])

    # Face results
    print(f"\n  ── FACE ANALYSIS ──────────────────────────────────────────")
    print(f"  Detected: {face['dominant_emotion'].upper()}")
    print_scores(face["emotion_scores"])

    # Voice results
    if voice and voice["success"]:
        print(f"\n  ── VOICE ANALYSIS ─────────────────────────────────────────")
        print(f"  Detected: {voice['dominant_emotion'].upper()}")
        if "features" in voice:
            f = voice["features"]
            print(f"  Acoustic: Energy={f['energy']} | Pitch={f['pitch_hz']}Hz | ZCR={f['zero_crossing_rate']}")
        print_scores(voice["emotion_scores"])

        # Fusion
        dominant, fused_scores = fuse_emotions(face, voice)
        print(f"\n  ── MULTIMODAL FUSION (Face 60% + Voice 40%) ───────────────")
        print(f"  FINAL EMOTION: {dominant.upper()}")
        print_scores(fused_scores)
        adaptation = print_adaptation(dominant)
        save_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "face+voice",
            "image": os.path.basename(image_path),
            "face_emotion": face["dominant_emotion"],
            "voice_emotion": voice["dominant_emotion"],
            "final_emotion": dominant,
            "adaptation": adaptation["adaptation"]
        })
    else:
        print(f"\n  [VOICE] Skipped — {rec.get('error', 'unknown error')}")
        adaptation = print_adaptation(face["dominant_emotion"])
        save_log({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "face_only_fallback",
            "image": os.path.basename(image_path),
            "detected_emotion": face["dominant_emotion"],
            "adaptation": adaptation["adaptation"]
        })


# ── DEMO MODE ─────────────────────────────────────────────────────────────────
def run_demo():
    print("\n" + "=" * 65)
    print("  MERSA — DEMO MODE (simulated multimodal output)")
    print("=" * 65)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modality  : Face + Voice (simulated)")
    print("-" * 65)

    face = {
        "success": True, "dominant_emotion": "angry",
        "emotion_scores": {"angry": 62.3, "sad": 18.1, "neutral": 10.4,
                           "fear": 5.2, "happy": 2.8, "surprise": 0.9, "disgust": 0.3}
    }
    voice = {
        "success": True, "dominant_emotion": "angry",
        "emotion_scores": {"angry": 55.0, "surprise": 20.0, "happy": 10.0,
                           "neutral": 10.0, "fear": 3.0, "sad": 1.0, "disgust": 1.0},
        "features": {"energy": 0.068, "pitch_hz": 245.3, "zero_crossing_rate": 0.092}
    }

    print(f"\n  ── FACE ANALYSIS ──────────────────────────────────────────")
    print(f"  Detected: {face['dominant_emotion'].upper()}")
    print_scores(face["emotion_scores"])

    print(f"\n  ── VOICE ANALYSIS ─────────────────────────────────────────")
    print(f"  Detected: {voice['dominant_emotion'].upper()}")
    f = voice["features"]
    print(f"  Acoustic: Energy={f['energy']} | Pitch={f['pitch_hz']}Hz | ZCR={f['zero_crossing_rate']}")
    print_scores(voice["emotion_scores"])

    dominant, fused = fuse_emotions(face, voice)
    print(f"\n  ── MULTIMODAL FUSION (Face 60% + Voice 40%) ───────────────")
    print(f"  FINAL EMOTION: {dominant.upper()}")
    print_scores(fused)
    print_adaptation(dominant)
    print("=" * 65)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MERSA — Multimodal Emotion Recognition System")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--voice", action="store_true", help="Enable voice analysis")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    args = parser.parse_args()

    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║   MERSA - Multimodal Emotion Recognition System ║")
    print("  ║   INF2102 - PUC-Rio - 2026.1 - 3WA             ║")
    print("  ║   Author: Filza Javed                           ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print("\n  Available modes:")
    print("    --demo                        Simulated output")
    print("    --image photo.jpg             Face only")
    print("    --voice                       Voice only")
    print("    --image photo.jpg --voice     Face + Voice combined")

    if args.demo:
        run_demo()
    elif args.image and not args.voice:
        if not os.path.exists(args.image):
            print(f"\n  [ERROR] Image not found: {args.image}")
            sys.exit(1)
        mode_face_only(args.image)
    elif args.voice and not args.image:
        mode_voice_only()
    elif args.image and args.voice:
        if not os.path.exists(args.image):
            print(f"\n  [ERROR] Image not found: {args.image}")
            sys.exit(1)
        mode_combined(args.image)
    else:
        print("\n  Please provide --demo, --image, --voice, or --image + --voice")

if __name__ == "__main__":
    main()
