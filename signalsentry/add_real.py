"""Add a REAL recording to the separate real-examples table (never to train/test).

Example:
  python -m signalsentry.add_real data/real/ao73.iq --fs 9600 --rs 1200 --label BPSK ^
      --label-source "SatNOGS DB lists FUNcube-1 as 1200-baud BPSK" --note "audio, not raw IQ"
Leave out --label if the true modulation is not known (it is stored as 'unknown').
"""
import argparse
import csv
import datetime
import json
import shutil
from pathlib import Path

import numpy as np

from signalsentry.validator import analyze

COLS = ["file", "added", "label", "label_source", "fs_hz", "rs_hz", "n_samples",
        "centre_hz", "bandwidth_hz", "snr_db", "abs_C40", "C42", "env_var",
        "predicted", "top_prob", "verdict", "score", "hard_flags", "note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--fs", type=float)
    ap.add_argument("--rs", type=float)
    ap.add_argument("--label", default="unknown", help="true modulation, only if you know it reliably")
    ap.add_argument("--label-source", default="", help="where the label comes from")
    ap.add_argument("--note", default="")
    ap.add_argument("--csv", default="data/dataset/real_examples.csv")
    ap.add_argument("--copy-to", default="data/dataset/real")
    a = ap.parse_args()

    path = Path(a.file)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    fs = a.fs or meta.get("sample_rate_hz")
    rs = a.rs or meta.get("symbol_rate_hz")
    if fs is None or rs is None:
        raise SystemExit("pass --fs and --rs (no sidecar found)")
    if a.label != "unknown" and not a.label_source:
        raise SystemExit("a label needs --label-source (where do you know it from?)")

    iq = np.fromfile(path, dtype=np.complex64)
    r = analyze(iq, fs, rs)
    m = r["measurements"]
    f = r.get("features", {})
    row = {
        "file": path.name, "added": datetime.date.today().isoformat(), "label": a.label,
        "label_source": a.label_source, "fs_hz": fs, "rs_hz": rs, "n_samples": len(iq),
        "centre_hz": round(m.get("centre_hz", float("nan")), 1),
        "bandwidth_hz": round(m.get("bandwidth_hz", float("nan")), 1),
        "snr_db": round(m.get("snr_db", float("nan")), 2),
        "abs_C40": round(f.get("abs_C40", float("nan")), 4), "C42": round(f.get("C42", float("nan")), 4),
        "env_var": round(f.get("env_var", float("nan")), 5),
        "predicted": r.get("predicted", ""), "top_prob": max(r["probabilities"].values()) if "probabilities" in r else "",
        "verdict": r["verdict"], "score": r["score"],
        "hard_flags": ";".join(sorted({x["rule"] for x in r["flags"] if x["severity"] == "HARD"})),
        "note": a.note,
    }
    out = Path(a.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    Path(a.copy_to).mkdir(parents=True, exist_ok=True)
    shutil.copy(path, Path(a.copy_to) / path.name)
    print(f"added {path.name} to {out}")
    print(f"label: {a.label} | classifier: {row['predicted']} | verdict: {row['verdict']} ({row['score']}) | flags: {row['hard_flags']}")


if __name__ == "__main__":
    main()