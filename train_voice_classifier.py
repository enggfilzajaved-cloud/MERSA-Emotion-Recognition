# Trains the v2 voice classifier. Random Forest, actor-independent split
# so I'm not just testing on voices the model already heard.
#
# Actors 1-19 for training, 20-24 held out for testing. This matters --
# a random split would let the model partially learn to recognize the
# speaker instead of the emotion, which inflates accuracy in a way that
# doesn't actually generalize.

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

with open("features.json") as f:
    data = json.load(f)

X, y, actors = [], [], []
for row in data:
    X.append([row["rms"], row["pitch"], row["zcr"]] + row["mfcc"] + row["chroma"] + row["mel"])
    y.append(row["label"])
    actors.append(row["actor"])

X = np.array(X)
y = np.array(y)
actors = np.array(actors)
print("feature vector length:", X.shape[1])

train = actors <= 19
test = actors > 19
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
print(f"train: {len(X_train)}, test: {len(X_test)} (held-out actors)")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\ntest accuracy: {acc*100:.2f}% ({(y_pred==y_test).sum()}/{len(y_test)})\n")

labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=labels)
print("confusion matrix:", labels)
for i, row in enumerate(cm):
    print(f"  {labels[i]:10s}", list(row))

joblib.dump(clf, "mersa_voice_classifier.joblib")
joblib.dump(scaler, "mersa_voice_scaler.joblib")
print("\nsaved model + scaler")
