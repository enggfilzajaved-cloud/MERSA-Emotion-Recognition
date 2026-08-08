# Runs the original (v1) rule-based analyze_voice() from mersa_final.py
# against the full RAVDESS speech set to see how well it actually does.
# Spoiler: not great, which is why I built v2 (see train_classifier_v2.py).

import os
import json
import importlib.util

spec = importlib.util.spec_from_file_location("mersa_final", "mersa_final.py")
mersa = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mersa)
except SystemExit:
    pass  # mersa_final.py has a CLI entrypoint, ignore it here
analyze_voice = mersa.analyze_voice

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
# MERSA doesn't have a "calm" category so I'm folding it into neutral
TO_MERSA_LABEL = {
    "neutral": "neutral", "calm": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "fearful": "fear", "disgust": "disgust", "surprised": "surprise",
}

folder = "ravdess_speech_1440"
files = sorted(os.listdir(folder))
print(f"{len(files)} files to process")

results = []
for i, fname in enumerate(files):
    emo_code = fname.split("-")[2]
    true_label = TO_MERSA_LABEL[RAVDESS_EMOTIONS[emo_code]]

    r = analyze_voice(os.path.join(folder, fname))
    pred = r.get("dominant_emotion", "ERROR")

    results.append({"file": fname, "true_label": true_label, "predicted_label": pred})

    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(files)}")

os.makedirs("results", exist_ok=True)
with open("results/voice_results.json", "w") as f:
    json.dump(results, f, indent=2)

correct = sum(r["true_label"] == r["predicted_label"] for r in results)
print(f"\n{correct}/{len(results)} = {correct/len(results)*100:.2f}%")
