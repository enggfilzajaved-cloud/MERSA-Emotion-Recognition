# Same test as 01_run_deepface.py but using a different pretrained model
# (from github.com/gitshanks/fer2013) so I could compare DeepFace against
# something else on the exact same images.
#
# Note: this model's own training script doesn't respect FER2013's
# official train/test split (it just does a random 90/10 split on the
# whole dataset), so there's a chance it saw some of these test images
# during its own training. Flagging this as a limitation in the report.

import os
import json
import numpy as np
import tf_keras
from PIL import Image

with open('fer2013-master/fer.json') as f:
    model = tf_keras.models.model_from_json(f.read())
model.load_weights('fer2013-master/fer.h5')

# same order as the original repo used for training
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

DATA_DIR = "fer_test"
results = []

for emotion in LABELS:
    folder = os.path.join(DATA_DIR, emotion)
    for fname in sorted(os.listdir(folder)):
        img = Image.open(os.path.join(folder, fname)).convert('L').resize((48, 48))
        x = np.array(img).astype('float32') / 255.0
        x = x.reshape(1, 48, 48, 1)

        probs = model.predict(x, verbose=0)[0]
        pred = LABELS[np.argmax(probs)]

        results.append({"file": fname, "true_label": emotion, "predicted_label": pred})
        print(f"{emotion:10s} {fname:30s} -> {pred}")

os.makedirs("results", exist_ok=True)
with open("results/altmodel_results.json", "w") as f:
    json.dump(results, f, indent=2)

correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])
print(f"\n{correct}/{len(results)} correct = {correct/len(results)*100:.2f}%")
