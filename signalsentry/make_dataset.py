"""Write the synthetic dataset to disk: feature tables (CSV) plus a small set of raw .iq examples.

Run:  python -m signalsentry.make_dataset
Uses the SAME seeds and settings as classifier.py, so the CSV rows are exactly what the model saw.
"""
import csv
from pathlib import Path

import numpy as np

from signalsentry.classifier import (KNOWN, RS, TEST_CFG, TEST_SNR_LEVELS, TRAIN_CFG,
                                     TRAIN_SNR_RANGE)
from signalsentry.features import extract_features
from signalsentry.generator import DEFAULT_FS, generate_signal, save_iq

OUT = Path("data/dataset")
COLS = ["file_id", "split", "modulation", "snr_db", "offset_hz", "seed", "n_symbols",
        "abs_C40", "C42", "env_var"]


def rows_for(split, mods, snrs, per_cell, cfg):
    rng = np.random.default_rng(cfg["seed_base"])
    seed = cfg["seed_base"]
    rows = []
    for mod in mods:
        for snr in snrs:
            for _ in range(per_cell):
                seed += 1
                off = rng.uniform(-cfg["offset_max"], cfg["offset_max"])
                iq, _ = generate_signal(mod, cfg["n_symbols"], float(snr), off, seed=seed)
                try:
                    f = extract_features(iq, DEFAULT_FS, RS)
                except ValueError:
                    continue                                   # nothing detected: skipped, like in training
                rows.append([f"{split}_{seed}", split, mod, round(float(snr), 3), round(off, 1), seed,
                             cfg["n_symbols"], round(float(f[0]), 5), round(float(f[1]), 5), round(float(f[2]), 5)])
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snrs_train = np.random.default_rng(0).uniform(*TRAIN_SNR_RANGE, 40)
    write_csv(OUT / "train_features.csv", rows_for("train", KNOWN, snrs_train, 15, TRAIN_CFG))
    write_csv(OUT / "test_features.csv", rows_for("test", KNOWN, TEST_SNR_LEVELS, 60, TEST_CFG))
    write_csv(OUT / "unfamiliar_8psk_features.csv", rows_for("8psk", ["8PSK"], TEST_SNR_LEVELS, 40, TEST_CFG))

    raw = OUT / "raw_examples"
    n = 0
    for i, mod in enumerate(KNOWN + ["8PSK"]):
        for snr in (0, 10, 20):
            for k in range(2):
                seed = 7_000_000 + 100 * i + 10 * snr + k
                off = float(np.random.default_rng(seed).uniform(-40e3, 40e3))
                iq, meta = generate_signal(mod, 4000, snr, off, seed=seed)
                save_iq(iq, meta, raw / f"{mod}_{snr}dB_{k}.iq")
                n += 1
    print(f"wrote {n} raw .iq examples (+ .json ground truth) to {raw}")


if __name__ == "__main__":
    main()