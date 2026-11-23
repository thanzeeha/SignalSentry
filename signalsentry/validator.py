"""Module 5: validation rules R1, R2, R4, R7, trust score, reasons."""
import argparse
import json
from pathlib import Path

import numpy as np

from signalsentry.classifier import KNOWN, load_model, predict
from signalsentry.features import THEORY, extract_features
from signalsentry.measure import measure

# ---- thresholds (tolerances calibrated on a separate synthetic calibration set) ----
SNR_MIN_DB = 0.0            # R1
PROB_MIN = 0.70             # R2
TOL_HIGH_SNR = 0.07         # R4: allowed |distance| in (|C40|, C42) space at SNR >= 0 dB
TOL_AT_MINUS5 = 0.10        #     grows linearly to this at -5 dB
HARD_FACTOR = 2.0           # distance > 2 x tol  -> hard fail
ENTROPY_MAX = 0.85          # R7: normalised probability entropy at/above this = "spread out"
REF_SYMBOLS = 3500          # tolerance was calibrated at this length
PASS_SCORE = 75.0
WEIGHTS = {"R1": 0.15, "R2": 0.20, "R3": 0.15, "R4": 0.30, "R7": 0.20}
BW_RATIO_EXPECTED = 1.163   # 99%-power bandwidth / symbol rate for RRC roll-off 0.35 (measured, calibration set)
BW_TOL = 0.10               # R3: allowed relative deviation (+-10%); synthetic scatter is only about 1%


def expected_cumulants(mod, snr_db, sps):
    """Theory values shrunk by the known noise bias: factor 1/(1+1/s)^2, s = symbol-level SNR."""
    s = sps * 10 ** (snr_db / 10)
    k = 1.0 / (1.0 + 1.0 / s) ** 2
    t40, t42 = THEORY[mod]
    return t40 * k, t42 * k


def tolerance(snr_db, n_symbols):
    if snr_db >= 0:
        tol = TOL_HIGH_SNR
    else:
        tol = TOL_HIGH_SNR + (TOL_AT_MINUS5 - TOL_HIGH_SNR) * min(-snr_db / 5.0, 1.0)
    return tol * max(1.0, np.sqrt(REF_SYMBOLS / max(n_symbols, 1)))


def _clip01(x):
    return float(min(max(x, 0.0), 1.0))


def analyze(iq, fs, symbol_rate, clf=None, beta=0.35):
    """Run the whole pipeline. Returns a dict with verdict, score, flags and measurements."""
    clf = clf or load_model()
    m = measure(iq, fs)
    if not m["detected"]:
        return {"verdict": "HUMAN REVIEW REQUIRED", "score": 0.0, "measurements": m,
                "flags": [{"rule": "DETECT", "severity": "HARD",
                           "reason": "No signal found above the noise floor, nothing to classify."}]}

    feats = extract_features(iq, fs, symbol_rate, m["centre_hz"], beta)
    label, probs = predict(clf, feats)
    p = np.array(list(probs.values()))
    top = float(p.max())
    ent = float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p)))

    snr = m["snr_db"]
    sps = fs / symbol_rate
    n_sym = int(len(iq) * symbol_rate / fs)
    tol = tolerance(snr, n_sym)
    c40, c42 = feats[0], feats[1]

    def dist(mod):
        e40, e42 = expected_cumulants(mod, snr, sps)
        return float(np.hypot(c40 - e40, c42 - e42))

    d_pred = dist(label)
    d_all = {mod: dist(mod) for mod in KNOWN}
    nearest = min(d_all, key=d_all.get)
    d_min = d_all[nearest]
    e40, e42 = expected_cumulants(label, snr, sps)

    flags = []

    def flag(rule, sev, reason):
        flags.append({"rule": rule, "severity": sev, "reason": reason})

    # R1: SNR
    if snr < SNR_MIN_DB:
        flag("R1", "HARD", f"Measured SNR {snr:.1f} dB is below {SNR_MIN_DB:.0f} dB; "
                           f"cumulant estimates are unreliable at this noise level.")
    # R2: classifier confidence
    if top < PROB_MIN:
        flag("R2", "HARD", f"Top class {label} has probability {top:.2f}, below {PROB_MIN:.2f}.")
    # R3: measured bandwidth must agree with the symbol rate the user typed
    bw_ratio = m["bandwidth_hz"] / symbol_rate
    bw_dev = abs(bw_ratio / BW_RATIO_EXPECTED - 1.0)
    if bw_dev > BW_TOL:
        flag("R3", "HARD", f"Measured 99% bandwidth {m['bandwidth_hz'] / 1e3:.2f} kHz is {bw_ratio:.2f}x the symbol rate "
                           f"you entered ({symbol_rate / 1e3:.2f} kHz); expected about {BW_RATIO_EXPECTED:.2f}x "
                           f"(allowed +-{BW_TOL:.0%}). The symbol rate may be wrong, or the signal is not "
                           f"single-carrier RRC; cumulants cannot be trusted.")
    # R4: cumulants vs predicted class
    if d_pred > HARD_FACTOR * tol:
        flag("R4", "HARD", f"Measured (|C40|={c40:.2f}, C42={c42:.2f}) is far from what {label} should give "
                           f"at this SNR ({e40:.2f}, {e42:.2f}): distance {d_pred:.2f} > {HARD_FACTOR * tol:.2f}.")
    elif d_pred > tol:
        flag("R4", "WARN", f"Measured cumulants are somewhat off for {label}: distance {d_pred:.2f} > {tol:.2f}.")
    # R7: unfamiliar signal
    if d_min > HARD_FACTOR * tol:
        flag("R7", "HARD", f"Unfamiliar signal: cumulants are far from every known class "
                           f"(nearest is {nearest}, distance {d_min:.2f} > {HARD_FACTOR * tol:.2f}).")
    if ent >= ENTROPY_MAX:
        flag("R7", "HARD", f"Unfamiliar signal: class probabilities are spread out (entropy {ent:.2f}).")

    # sub-scores 0..1 and weighted trust score 0..100
    sub = {
        "R1": _clip01(snr / 10.0),
        "R2": _clip01((top - 0.5) / 0.4),
        "R3": _clip01((1.5 * BW_TOL - bw_dev) / BW_TOL),
        "R4": _clip01((HARD_FACTOR * tol - d_pred) / ((HARD_FACTOR - 0.5) * tol)),
        "R7": min(_clip01((HARD_FACTOR * tol - d_min) / ((HARD_FACTOR - 0.5) * tol)),
                  _clip01((ENTROPY_MAX - ent) / 0.5)),
    }
    score = 100.0 * sum(WEIGHTS[r] * sub[r] for r in WEIGHTS)

    hard = any(f["severity"] == "HARD" for f in flags)
    if not hard and score < PASS_SCORE:
        flag("SCORE", "HARD", f"Trust score {score:.0f} is below {PASS_SCORE:.0f}.")
        hard = True
    verdict = "HUMAN REVIEW REQUIRED" if hard else "VALIDATED"

    return {"verdict": verdict, "score": round(score, 1), "predicted": label,
            "probabilities": {k: round(float(v), 3) for k, v in probs.items()},
            "features": {"abs_C40": float(c40), "C42": float(c42), "env_var": float(feats[2])},
            "measurements": m, "distance_to_predicted": d_pred, "nearest_known": nearest,
            "distance_to_nearest": d_min, "tolerance": tol,
            "bandwidth_ratio": float(bw_ratio), "subscores": sub, "flags": flags}


def print_report(r):
    print(f"VERDICT: {r['verdict']}   trust score: {r['score']:.0f}/100")
    if "predicted" in r:
        pr = ", ".join(f"{k} {v:.2f}" for k, v in r["probabilities"].items())
        print(f"classifier says: {r['predicted']}  ({pr})")
        m = r["measurements"]
        print(f"centre {m['centre_hz']:+.0f} Hz, bandwidth {m['bandwidth_hz']:.0f} Hz, SNR {m['snr_db']:.1f} dB")
    if not r["flags"]:
        print("no flags")
    for f in r["flags"]:
        print(f"  [{f['severity']}] {f['rule']}: {f['reason']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--fs", type=float)
    ap.add_argument("--rs", type=float)
    a = ap.parse_args()
    path = Path(a.file)
    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    fs = a.fs or meta.get("sample_rate_hz")
    rs = a.rs or meta.get("symbol_rate_hz")
    if fs is None or rs is None:
        raise SystemExit("no sidecar found: pass --fs and --rs")
    if "modulation" in meta:
        print(f"truth: {meta['modulation']} at {meta['snr_db']} dB")
    print_report(analyze(np.fromfile(path, dtype=np.complex64), fs, rs))