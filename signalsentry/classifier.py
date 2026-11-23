"""Module 4: dataset builder, RandomForest classifier, per-SNR evaluation."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from signalsentry.features import extract_features
from signalsentry.generator import DEFAULT_FS, generate_signal

KNOWN = ["BPSK", "QPSK", "16QAM"]          # 8PSK is never used for training
MODEL_PATH = Path("models/classifier.joblib")
RS = DEFAULT_FS / 8

# Train and test use different seeds AND different offset ranges / symbol counts.
TRAIN_CFG = dict(seed_base=1_000, offset_max=40e3, n_symbols=4000)
TEST_CFG = dict(seed_base=900_000, offset_max=70e3, n_symbols=3000)
TRAIN_SNR_RANGE = (-5.0, 20.0)
TEST_SNR_LEVELS = [-5, 0, 5, 10, 15, 20]


def make_samples(mods, snrs, per_cell, cfg):
    """Features for every (modulation, snr) cell. Returns X, y, snr arrays."""
    rng = np.random.default_rng(cfg["seed_base"])
    X, y, s = [], [], []
    seed = cfg["seed_base"]
    for mod in mods:
        for snr in snrs:
            for _ in range(per_cell):
                seed += 1
                off = rng.uniform(-cfg["offset_max"], cfg["offset_max"])
                iq, _ = generate_signal(mod, cfg["n_symbols"], snr, off, seed=seed)
                try:
                    f = extract_features(iq, DEFAULT_FS, RS)
                except ValueError:                      # nothing detected: skip
                    continue
                X.append(f); y.append(mod); s.append(snr)
    return np.array(X), np.array(y), np.array(s)


def train_model():
    rng = np.random.default_rng(0)
    snrs = rng.uniform(*TRAIN_SNR_RANGE, 40)             # 40 random SNRs in the range
    X, y, _ = make_samples(KNOWN, snrs, 15, TRAIN_CFG)   # 3 x 40 x 15 = 1800 samples
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0)
    clf.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    return clf, X, y


def load_model():
    return joblib.load(MODEL_PATH)


def predict(clf, features):
    """Return (label, probability dict) for one feature vector."""
    p = clf.predict_proba(features.reshape(1, -1))[0]
    return clf.classes_[int(np.argmax(p))], dict(zip(clf.classes_, p))


if __name__ == "__main__":
    t0 = time.time()
    clf, Xtr, ytr = train_model()
    print(f"trained on {len(ytr)} samples in {time.time() - t0:.0f}s, classes {[str(c) for c in clf.classes_]}")

    Xte, yte, ste = make_samples(KNOWN, TEST_SNR_LEVELS, 60, TEST_CFG)
    pred = clf.predict(Xte)
    print(f"\nTEST (different seeds/offsets/length), n={len(yte)}, overall acc {np.mean(pred == yte):.3f}")
    print("\nAccuracy per SNR (wideband dB):")
    for snr in TEST_SNR_LEVELS:
        m = ste == snr
        row = "  ".join(f"{c}:{np.mean(pred[m & (yte == c)] == c):.2f}" for c in KNOWN)
        print(f"{snr:4d} dB  all {np.mean(pred[m] == yte[m]):.3f}   {row}")

    print("\nConfusion matrix (rows = true, cols = predicted), order", KNOWN)
    print(confusion_matrix(yte, pred, labels=KNOWN))

    X8, _, s8 = make_samples(["8PSK"], TEST_SNR_LEVELS, 40, TEST_CFG)
    p8 = clf.predict_proba(X8)
    top = p8.max(axis=1)
    lab = clf.classes_[p8.argmax(axis=1)]
    print("\n8PSK (never trained) -> what the classifier says:")
    for snr in TEST_SNR_LEVELS:
        m = s8 == snr
        counts = {c: int(np.sum(lab[m] == c)) for c in KNOWN}
        print(f"{snr:4d} dB  predicted {counts}  mean top-prob {top[m].mean():.2f}  "
              f"frac top-prob>=0.70: {np.mean(top[m] >= 0.70):.2f}")