"""Module 3: matched filter, offset removal, 1-sample-per-symbol, cumulants."""
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from signalsentry.generator import rrc_taps
from signalsentry.measure import measure

WORK_SPS = 8                 # internal samples per symbol
TRIM_SYMBOLS = 10            # drop filter transients at both ends

# Theory for unit-power, noise-free signals: (|C40|, C42)
THEORY = {
    "BPSK":  (2.00, -2.00),
    "QPSK":  (1.00, -1.00),
    "16QAM": (0.68, -0.68),
}


def to_symbol_samples(iq, fs, symbol_rate, centre_hz, beta=0.35):
    """Remove coarse offset, matched-filter, pick best timing phase, 1 sample/symbol."""
    n = np.arange(len(iq))
    x = iq * np.exp(-2j * np.pi * centre_hz * n / fs)           # move signal to 0 Hz

    sps = fs / symbol_rate
    if abs(sps - round(sps)) > 1e-6 or round(sps) != WORK_SPS:   # resample to 8 sps
        frac = Fraction(WORK_SPS * symbol_rate / fs).limit_denominator(1000)
        x = resample_poly(x, frac.numerator, frac.denominator)

    y = np.convolve(x, rrc_taps(beta, WORK_SPS), mode="same")    # matched filter
    y = y[TRIM_SYMBOLS * WORK_SPS: -TRIM_SYMBOLS * WORK_SPS]

    # timing: the phase (0..7) with the highest mean power is the symbol instant
    powers = [np.mean(np.abs(y[p::WORK_SPS]) ** 2) for p in range(WORK_SPS)]
    s = y[int(np.argmax(powers))::WORK_SPS]
    return s / np.sqrt(np.mean(np.abs(s) ** 2))                  # power = 1


def fine_offset_correct(s, symbol_rate, max_hz=3000.0):
    """Estimate leftover offset from the spectral line of s**4 and remove it."""
    N = len(s)
    nfft = 1 << (int(np.ceil(np.log2(N))) + 3)
    spec = np.abs(np.fft.fft(s ** 4, nfft))
    delta = np.fft.fftfreq(nfft, d=1 / symbol_rate) / 4          # s**4 line sits at 4*offset
    ok = np.abs(delta) < max_hz
    best = delta[ok][np.argmax(spec[ok])]
    return s * np.exp(-2j * np.pi * best * np.arange(N) / symbol_rate), float(best)


def cumulants(s):
    """C40, C42 and envelope variance of unit-power complex symbol samples."""
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    a2 = np.abs(s) ** 2
    m20 = np.mean(s ** 2)
    m40 = np.mean(s ** 4)
    m42 = np.mean(a2 ** 2)
    c40 = m40 - 3 * m20 ** 2
    c42 = m42 - np.abs(m20) ** 2 - 2 * np.mean(a2) ** 2
    return {"C40": c40, "C42": float(np.real(c42)), "env_var": float(np.var(np.abs(s)))}


def extract_features(iq, fs, symbol_rate, centre_hz=None, beta=0.35, fine=True):
    """Return feature vector [|C40|, C42, envelope variance] (and measured centre)."""
    if centre_hz is None:
        m = measure(iq, fs)
        if not m["detected"]:
            raise ValueError("no signal detected")
        centre_hz = m["centre_hz"]
    s = to_symbol_samples(iq, fs, symbol_rate, centre_hz, beta)
    if fine:
        s, _ = fine_offset_correct(s, symbol_rate)
    c = cumulants(s)
    return np.array([abs(c["C40"]), c["C42"], c["env_var"]])


if __name__ == "__main__":
    from signalsentry.generator import generate_signal

    print("Cumulants at 30 dB, 20000 symbols (offset 12 kHz):")
    print(f"{'mod':6s} {'|C40|':>7s} {'theory':>7s} {'C42':>7s} {'theory':>7s} {'envvar':>7s}")
    for i, mod in enumerate(["BPSK", "QPSK", "16QAM", "8PSK"]):
        iq, meta = generate_signal(mod, n_symbols=20000, snr_db=30, freq_offset_hz=12e3, seed=300 + i)
        f = extract_features(iq, meta["sample_rate_hz"], meta["symbol_rate_hz"])
        t = THEORY.get(mod, (0.0, -1.0))
        print(f"{mod:6s} {f[0]:7.3f} {t[0]:7.2f} {f[1]:7.3f} {t[1]:7.2f} {f[2]:7.4f}")

    print("\nSensitivity: QPSK 4000 symbols, 20 dB, deliberate error added to centre")
    print(f"{'error Hz':>9s} {'|C40| no fix':>13s} {'|C40| fixed':>12s} {'C42 no fix':>11s}")
    iq, meta = generate_signal("QPSK", n_symbols=4000, snr_db=20, freq_offset_hz=12e3, seed=400)
    for err in [0, 50, 150, 400, 800]:
        c = 12e3 + err
        a = extract_features(iq, meta["sample_rate_hz"], meta["symbol_rate_hz"], c, fine=False)
        b = extract_features(iq, meta["sample_rate_hz"], meta["symbol_rate_hz"], c, fine=True)
        print(f"{err:9d} {a[0]:13.3f} {b[0]:12.3f} {a[1]:11.3f}")