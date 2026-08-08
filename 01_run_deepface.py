# Runs DeepFace on the labeled FER2013 test set and saves predictions.
# Used to get the accuracy numbers reported in the project report / README.

import os
import json
from deepface import DeepFace

DATA_DIR = "fer_test"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

results = []

for emotion in EMOTIONS:
    folder = os.path.join(DATA_DIR, emotion)
    for fname in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, fname)

        try:
            analysis = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):  # newer deepface versions return a list
                analysis = analysis[0]
            pred = analysis['dominant_emotion']
        except Exception as e:
            print(f"failed on {fname}: {e}")
            pred = "ERROR"

        results.append({"file": fname, "true_label": emotion, "predicted_label": pred})
        print(f"{emotion:10s} {fname:30s} -> {pred}")

os.makedirs("results", exist_ok=True)
with open("results/deepface_results.json", "w") as f:
    json.dump(results, f, indent=2)

correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])
print(f"\n{correct}/{len(results)} correct = {correct/len(results)*100:.2f}%")
