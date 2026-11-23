"""Module 2: PSD, detection, centre frequency, bandwidth (99% power), SNR."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

DETECT_MARGIN_DB = 3.0    # smoothed PSD peak must exceed noise floor by this much
EDGE_MARGIN_DB = 2.0      # bins above floor by this much count as "occupied"


def compute_psd(iq, fs, nperseg=1024):
    """Welch PSD, returned with frequency axis from -fs/2 to +fs/2 (linear power/Hz)."""
    f, p = welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)
    return np.fft.fftshift(f), np.fft.fftshift(p)


def noise_floor(p):
    """Median PSD bin. Robust as long as the signal fills less than half the band."""
    return float(np.median(p))


def _smooth(p, k=9):
    return np.convolve(p, np.ones(k) / k, mode="same")


def _largest_run(mask):
    """Boolean mask of the longest contiguous True stretch (ignores isolated noise spikes)."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.zeros_like(mask)
    splits = np.flatnonzero(np.diff(idx) > 1) + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    out = np.zeros_like(mask)
    out[best] = True
    return out


def measure(iq, fs):
    """Return a dict of measurements. 'detected' False means no signal found."""
    f, p = compute_psd(iq, fs)
    df = f[1] - f[0]
    floor = noise_floor(p)
    sm = _smooth(p)
    detected = bool(sm.max() > floor * 10 ** (DETECT_MARGIN_DB / 10))
    out = {"detected": detected, "noise_floor": floor}
    if not detected:
        return out

    occ = _largest_run(sm > floor * 10 ** (EDGE_MARGIN_DB / 10))
    excess = np.where(occ, np.clip(p - floor, 0, None), 0.0)

    centre = float(np.sum(f * excess) / np.sum(excess))

    cum = np.cumsum(excess) / np.sum(excess)
    lo = float(np.interp(0.005, cum, f))
    hi = float(np.interp(0.995, cum, f))

    total_power = float(np.sum(p) * df)
    noise_power = floor * fs
    sig_power = max(total_power - noise_power, 1e-12)
    snr_db = 10 * np.log10(sig_power / noise_power)

    out.update({"centre_hz": centre, "bandwidth_hz": hi - lo,
                "snr_db": float(snr_db), "f_low_hz": lo, "f_high_hz": hi})
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1])
    iq = np.fromfile(path, dtype=np.complex64)
    meta = json.loads(path.with_suffix(".json").read_text())
    m = measure(iq, meta["sample_rate_hz"])
    theory_bw = meta["symbol_rate_hz"] * (1 + meta["rolloff"])
    print(f"file: {path.name}")
    print(f"detected:   {m['detected']}")
    print(f"centre:     {m['centre_hz']:+10.0f} Hz   (true {meta['freq_offset_hz']:+.0f})")
    print(f"bandwidth:  {m['bandwidth_hz']:10.0f} Hz   (theory {theory_bw:.0f}, 99% power is a bit less)")
    print(f"SNR:        {m['snr_db']:10.2f} dB   (true {meta['snr_db']})")