# Computes accuracy / precision / recall / specificity from whatever
# results file you point it at. Used for both the DeepFace and
# alternative CNN results.

import json
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main(results_path):
    with open(results_path) as f:
        data = json.load(f)

    y_true = [r["true_label"] for r in data]
    y_pred = [r["predicted_label"] for r in data]

    acc = accuracy_score(y_true, y_pred)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"accuracy: {acc*100:.2f}% ({correct}/{len(y_true)})\n")

    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    total = len(y_true)
    print("specificity per class:")
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) else 0
        print(f"  {label:10s} {spec*100:.1f}%")

    print("\nconfusion matrix (rows=true, cols=pred):", LABELS)
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:10s}", list(row))


if __name__ == "__main__":
    # run twice, once per model:
    # python 03_compute_metrics.py results/deepface_results.json
    # python 03_compute_metrics.py results/altmodel_results.json
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("=== DeepFace ===")
        main("results/deepface_results.json")
        print("\n=== Alternative CNN ===")
        main("results/altmodel_results.json")
