"""Module 6: how good is the validator? Counts caught errors and false alarms."""
import json
from pathlib import Path

import numpy as np

from signalsentry.classifier import KNOWN, load_model
from signalsentry.generator import DEFAULT_FS, generate_signal
from signalsentry.validator import PROB_MIN, analyze

FS, RS = DEFAULT_FS, DEFAULT_FS / 8
SEED0 = 2_000_000          # separate from train (1000), calibration (500000), test (900000)
LEVELS = [-10, -8, -5, 0, 5, 10, 15, 20]
PER_CELL = 40
CROSS_CHECK_RULES = {"R2", "R3", "R4", "R7", "DETECT"}   # everything except the SNR gate R1


def one(clf, mod, snr, rng, seed, typed_rs=RS):
    off = rng.uniform(-70e3, 70e3)
    iq, _ = generate_signal(mod, 3000, snr, off, seed=seed)
    r = analyze(iq.astype(np.complex64), FS, typed_rs, clf)
    hard = {f["rule"] for f in r["flags"] if f["severity"] == "HARD"}
    top = max(r["probabilities"].values()) if "probabilities" in r else 0.0
    return {"correct": r.get("predicted") == mod, "detected": "predicted" in r,
            "full": r["verdict"] != "VALIDATED",              # full validator flags it
            "cross": bool(hard & CROSS_CHECK_RULES),          # flagged without using R1
            "r2": top < PROB_MIN}                             # baseline: confidence only


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def report(title, rows):
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    print(f"\n{title}: n={len(rows)}, classifier wrong {len(wrong)}, right {len(right)}")
    for name, key in [("full validator", "full"), ("R2/R3/R4/R7 (no SNR gate)", "cross"),
                      ("R2-only baseline", "r2")]:
        c = sum(r[key] for r in wrong)
        f = sum(r[key] for r in right)
        print(f"  {name:30s} caught {pct(c, len(wrong)):>12s}   false alarms {pct(f, len(right)):>12s}")
    return {"n": len(rows), "wrong": len(wrong), "right": len(right),
            **{k: {"caught": sum(r[k] for r in wrong), "false_alarms": sum(r[k] for r in right)}
               for k in ["full", "cross", "r2"]}}


def main():
    clf = load_model()
    rng = np.random.default_rng(SEED0)
    seed = SEED0
    A = {s: [] for s in LEVELS}
    for snr in LEVELS:
        for mod in KNOWN:
            for _ in range(PER_CELL):
                seed += 1
                A[snr].append(one(clf, mod, snr, rng, seed))

    print("A) KNOWN MODULATIONS, CORRECT SYMBOL RATE")
    print("\nPer SNR: classifier wrong (of which: signal not detected) | flagged by full validator")
    for snr in LEVELS:
        rows = A[snr]
        wrong = sum(not r["correct"] for r in rows)
        undet = sum(not r["detected"] for r in rows)
        print(f"  {snr:4d} dB: wrong {wrong:3d}/{len(rows)} (undetected {undet:3d}) | flagged {sum(r['full'] for r in rows):3d}/{len(rows)}")
    out = {}
    out["A_all"] = report("A all SNRs", [r for s in LEVELS for r in A[s]])
    out["A_detected_only"] = report("A, only files where a signal was detected",
                                    [r for s in LEVELS for r in A[s] if r["detected"]])
    out["A_ge5"] = report("A only SNR >= 5 dB", [r for s in LEVELS if s >= 5 for r in A[s]])

    B = []
    for snr in LEVELS:
        for _ in range(PER_CELL):
            seed += 1
            B.append(one(clf, "8PSK", snr, rng, seed))
    for r in B:
        r["correct"] = False                                  # classifier cannot be right on 8PSK
    print("\nB) 8PSK (UNFAMILIAR, never trained)")
    out["B_all"] = report("B all SNRs", B)
    out["B_ge5"] = report("B only SNR >= 5 dB", B[4 * PER_CELL:])

    C = []
    for typed in [100e3, 150e3]:
        for snr in [5, 10, 15, 20]:
            for mod in KNOWN:
                for _ in range(15):
                    seed += 1
                    C.append(one(clf, mod, snr, rng, seed, typed_rs=typed))
    flagged = sum(r["full"] for r in C)
    passed_wrong = sum((not r["full"]) and (not r["correct"]) for r in C)
    print("\nC) WRONG SYMBOL RATE TYPED (true 125 kHz, typed 100 or 150 kHz)")
    print("   The input is invalid, so the right behaviour is to flag every file.")
    print(f"   flagged {pct(flagged, len(C))}; passed {len(C) - flagged}, of which wrong answers: {passed_wrong}")
    out["C"] = {"n": len(C), "flagged": flagged, "passed": len(C) - flagged, "passed_wrong": passed_wrong}

    Path("data").mkdir(exist_ok=True)
    Path("data/validator_eval.json").write_text(json.dumps(out, indent=2))
    print("\nsaved data/validator_eval.json")


if __name__ == "__main__":
    main()