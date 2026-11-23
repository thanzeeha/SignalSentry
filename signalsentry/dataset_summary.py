"""Print dataset counts and save a picture of it: data/dataset/dataset_overview.png"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path("data/dataset")


def load(name):
    return list(csv.DictReader(open(D / name)))


def main():
    sets = {"train": load("train_features.csv"), "test": load("test_features.csv"),
            "8PSK (unfamiliar)": load("unfamiliar_8psk_features.csv")}
    for name, rows in sets.items():
        print(f"{name}: {len(rows)} rows  {dict(Counter(r['modulation'] for r in rows))}")
    print("raw example files:", len(list((D / 'raw_examples').glob('*.iq'))))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"BPSK": "tab:blue", "QPSK": "tab:green", "16QAM": "tab:orange", "8PSK": "tab:red"}
    for mod in ["BPSK", "QPSK", "16QAM"]:
        r = [x for x in sets["train"] if x["modulation"] == mod]
        ax[0].scatter([float(x["abs_C40"]) for x in r], [float(x["C42"]) for x in r], s=6, alpha=0.5,
                      c=colors[mod], label=f"{mod} (train)")
    r = sets["8PSK (unfamiliar)"]
    ax[0].scatter([float(x["abs_C40"]) for x in r], [float(x["C42"]) for x in r], s=8, alpha=0.6,
                  c=colors["8PSK"], marker="x", label="8PSK (never trained)")
    ax[0].set_xlabel("|C40|"); ax[0].set_ylabel("C42"); ax[0].set_title("Training data: the 3 classes in feature space")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    levels = sorted({float(x["snr_db"]) for x in sets["test"]})
    counts = [sum(1 for x in sets["test"] if float(x["snr_db"]) == l) for l in levels]
    ax[1].bar([str(int(l)) for l in levels], counts, color="tab:gray")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("files"); ax[1].set_title("Test set: files per SNR level")
    fig.tight_layout()
    fig.savefig(D / "dataset_overview.png", dpi=130)
    print("saved", D / "dataset_overview.png")


if __name__ == "__main__":
    main()