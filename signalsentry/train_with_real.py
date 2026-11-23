"""Optional experiment: retrain the classifier with labelled REAL files added.

Writes a NEW model (models/classifier_with_real.joblib); the original model is never touched.
Prints what changes, so you can see whether adding the real file helps or hurts.

  python -m signalsentry.train_with_real --real-weight 1
"""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from signalsentry.classifier import KNOWN, MODEL_PATH
from signalsentry.features import extract_features
from signalsentry.validator import analyze

FEATS = ["abs_C40", "C42", "env_var"]


def load_synth(path):
    rows = list(csv.DictReader(open(path)))
    X = np.array([[float(r[c]) for c in FEATS] for r in rows])
    return X, np.array([r["modulation"] for r in rows]), np.array([float(r["snr_db"]) for r in rows])


def load_real(csv_path, real_dir, chunks):
    X, y, files, meta = [], [], [], {}
    for r in csv.DictReader(open(csv_path)):
        if r["label"] in ("", "unknown"):
            continue
        if r["label"] not in KNOWN:
            print(f"skipping {r['file']}: label {r['label']} is not one of {KNOWN} (cannot add a new class here)")
            continue
        iq = np.fromfile(Path(real_dir) / r["file"], dtype=np.complex64)
        fs, rs = float(r["fs_hz"]), float(r["rs_hz"])
        meta[r["file"]] = (iq, fs, rs, r["label"])
        n = len(iq) // chunks
        for c in range(chunks):
            try:
                X.append(extract_features(iq[c * n:(c + 1) * n], fs, rs))
            except ValueError:
                continue
            y.append(r["label"]); files.append(r["file"])
    return np.array(X), np.array(y), np.array(files), meta


def acc_table(clf, X, y, snr):
    p = clf.predict(X)
    return (np.mean(p == y), np.mean((p == y)[snr == -5]), np.mean((p == y)[snr >= 5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-csv", default="data/dataset/real_examples.csv")
    ap.add_argument("--real-dir", default="data/dataset/real")
    ap.add_argument("--train-csv", default="data/dataset/train_features.csv")
    ap.add_argument("--test-csv", default="data/dataset/test_features.csv")
    ap.add_argument("--chunks", type=int, default=6, help="split each real file into this many pieces")
    ap.add_argument("--real-weight", type=float, default=1.0, help="weight of each real row (synthetic rows = 1)")
    ap.add_argument("--out", default="models/classifier_with_real.joblib")
    a = ap.parse_args()

    Xs, ys, _ = load_synth(a.train_csv)
    Xt, yt, st = load_synth(a.test_csv)
    Xr, yr, fr, meta = load_real(a.real_csv, a.real_dir, a.chunks)
    if len(yr) == 0:
        raise SystemExit("no labelled real rows found (needs a --label from add_real)")
    print(f"synthetic training rows: {len(ys)}; real rows: {len(yr)} from {len(meta)} file(s), weight {a.real_weight}")
    print("NOTE: pieces of one file are NOT independent examples.")

    old = joblib.load(MODEL_PATH)
    new = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0)
    new.fit(np.vstack([Xs, Xr]), np.concatenate([ys, yr]),
            sample_weight=np.concatenate([np.ones(len(ys)), np.full(len(yr), a.real_weight)]))
    Path(a.out).parent.mkdir(exist_ok=True)
    joblib.dump(new, a.out)

    print("\nSynthetic TEST accuracy (overall | -5 dB | >= 5 dB)")
    for name, m in [("original", old), ("with real", new)]:
        o, lo, hi = acc_table(m, Xt, yt, st)
        print(f"  {name:10s} {o:.3f} | {lo:.3f} | {hi:.3f}")

    print("\nReal files, as seen by each model (the file IS in the new model's training data)")
    for fn, (iq, fs, rs, label) in meta.items():
        for name, m in [("original", old), ("with real", new)]:
            r = analyze(iq, fs, rs, m)
            hard = sorted({f["rule"] for f in r["flags"] if f["severity"] == "HARD"})
            print(f"  {fn} (label {label}) {name:10s} classifier: {r.get('predicted')} | {r['verdict']} ({r['score']}) {hard}")
    print("\nWith the real file in the training data you CANNOT use it as a test any more.")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()