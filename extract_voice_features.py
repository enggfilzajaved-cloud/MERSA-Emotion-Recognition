# Pulls acoustic features out of the RAVDESS wav files so I can train
# something better than the old rule-based thresholds. Kept the original
# 3 features (rms/pitch/zcr) and added mfcc/chroma/mel since those are
# what most speech emotion papers use.

import os
import json
import numpy as np
import librosa

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
TO_MERSA_LABEL = {
    "neutral": "neutral", "calm": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "fearful": "fear", "disgust": "disgust", "surprised": "surprise",
}

folder = "ravdess_speech_1440"
files = sorted(os.listdir(folder))

rows = []
for i, fname in enumerate(files):
    parts = fname.replace(".wav", "").split("-")
    emo_code = parts[2]
    actor = int(parts[6])
    label = TO_MERSA_LABEL[RAVDESS_EMOTIONS[emo_code]]

    y, sr = librosa.load(os.path.join(folder, fname), sr=22050)

    rms = float(np.mean(librosa.feature.rms(y=y)))
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[mags > np.median(mags)]
    pitch = float(np.mean(pitch_vals)) if len(pitch_vals) else 0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist()
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr), axis=1).tolist()
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1).tolist()

    rows.append({
        "file": fname, "actor": actor, "label": label,
        "rms": rms, "pitch": pitch, "zcr": zcr,
        "mfcc": mfcc, "chroma": chroma, "mel": mel,
    })

    if (i + 1) % 200 == 0:
        print(f"{i+1}/{len(files)}")

with open("features.json", "w") as f:
    json.dump(rows, f)

print("done, saved to features.json")
