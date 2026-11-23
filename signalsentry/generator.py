"""Module 1: synthetic IQ signal generator for SignalSentry."""
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_FS = 1_000_000.0   # sample rate in Hz
DEFAULT_SPS = 8            # samples per symbol
DEFAULT_BETA = 0.35        # RRC roll-off
RRC_SPAN = 10              # filter length in symbols


def constellation(mod):
    """Return the unit-average-power symbol points for a modulation."""
    if mod == "BPSK":
        pts = np.array([1, -1], dtype=complex)
    elif mod == "QPSK":
        pts = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
    elif mod == "8PSK":
        pts = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    elif mod == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        pts = (lv[:, None] + 1j * lv[None, :]).ravel()
    else:
        raise ValueError(f"Unknown modulation: {mod}")
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def rrc_taps(beta=DEFAULT_BETA, sps=DEFAULT_SPS, span=RRC_SPAN):
    """Root-raised-cosine pulse-shaping filter, normalised to unit energy."""
    n = np.arange(-span * sps // 2, span * sps // 2 + 1)
    t = n / sps
    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def generate_signal(mod, n_symbols=4000, snr_db=10.0, freq_offset_hz=0.0,
                    fs=DEFAULT_FS, sps=DEFAULT_SPS, beta=DEFAULT_BETA, seed=0):
    """Return (iq complex128 array, metadata dict).

    SNR here = signal power / noise power over the WHOLE sampled band (fs wide).
    """
    rng = np.random.default_rng(seed)
    pts = constellation(mod)
    symbols = pts[rng.integers(0, len(pts), n_symbols)]

    up = np.zeros(n_symbols * sps, dtype=complex)      # zero-stuff: 1 symbol then sps-1 zeros
    up[::sps] = symbols
    h = rrc_taps(beta, sps)
    x = np.convolve(up, h, mode="full")
    start = len(h) // 2                                  # remove filter delay
    x = x[start:start + n_symbols * sps]

    x = x / np.sqrt(np.mean(np.abs(x) ** 2))             # signal power = 1

    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))

    phase = rng.uniform(0, 2 * np.pi)
    n = np.arange(len(x))
    carrier = np.exp(1j * (2 * np.pi * freq_offset_hz * n / fs + phase))
    iq = x * carrier + noise                             # offset+phase on signal, noise stays white

    meta = {
        "modulation": mod, "sample_rate_hz": fs, "samples_per_symbol": sps,
        "symbol_rate_hz": fs / sps, "rolloff": beta, "n_symbols": n_symbols,
        "snr_db": snr_db, "freq_offset_hz": freq_offset_hz,
        "phase_rad": float(phase), "seed": seed,
    }
    return iq, meta


def save_iq(iq, meta, path):
    """Save complex64 .iq and a .json sidecar with ground truth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iq.astype(np.complex64).tofile(path)
    meta = dict(meta, file=path.name)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_iq(path):
    """Read a complex64 .iq file back into a NumPy array."""
    return np.fromfile(path, dtype=np.complex64)


def make_demo_files(out_dir="data/demo"):
    cases = [("BPSK", 12, 20e3), ("QPSK", 12, -15e3), ("16QAM", 18, 10e3), ("8PSK", 15, 5e3)]
    for i, (mod, snr, off) in enumerate(cases):
        iq, meta = generate_signal(mod, snr_db=snr, freq_offset_hz=off, seed=5000 + i)
        save_iq(iq, meta, Path(out_dir) / f"{mod}_{snr}dB.iq")
        print(f"wrote {out_dir}/{mod}_{snr}dB.iq  ({len(iq)} samples, offset {off:+.0f} Hz)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["demo"])
    ap.add_argument("--out", default="data/demo")
    a = ap.parse_args()
    make_demo_files(a.out)