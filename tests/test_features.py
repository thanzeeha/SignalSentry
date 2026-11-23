import numpy as np

from signalsentry.features import THEORY, cumulants, extract_features
from signalsentry.generator import generate_signal

FS, RS = 1e6, 125e3


def feats(mod, snr, off, seed, n=20000, fine=True, centre=None):
    iq, _ = generate_signal(mod, n_symbols=n, snr_db=snr, freq_offset_hz=off, seed=seed)
    return extract_features(iq, FS, RS, centre, fine=fine)


def test_theory_values_high_snr():
    for i, mod in enumerate(["BPSK", "QPSK", "16QAM"]):
        f = feats(mod, 30, 12e3, 300 + i)
        assert abs(f[0] - THEORY[mod][0]) < 0.1, (mod, f)
        assert abs(f[1] - THEORY[mod][1]) < 0.1, (mod, f)


def test_8psk_c40_small_c42_minus_one():
    f = feats("8PSK", 30, 12e3, 303)
    assert f[0] < 0.2 and abs(f[1] + 1) < 0.1, f


def test_envelope_variance_separates_16qam():
    qam = feats("16QAM", 30, 0, 1)[2]
    qpsk = feats("QPSK", 30, 0, 2)[2]
    assert qam > 5 * qpsk, (qam, qpsk)


def test_c42_rotation_invariant():
    # raw symbols with arbitrary phase give identical C42
    rng = np.random.default_rng(0)
    s = (rng.integers(0, 2, 5000) * 2 - 1) + 1j * (rng.integers(0, 2, 5000) * 2 - 1)
    a = cumulants(s)["C42"]
    b = cumulants(s * np.exp(1j * 1.234))["C42"]
    assert abs(a - b) < 1e-9


def test_noise_bias_matches_formula():
    # wideband 0 dB -> about 9 dB at symbol level (x8 = sps), so C42 = -1/(1+1/8)^2 = -0.79
    f = feats("QPSK", 0, 12e3, 5)
    assert abs(f[1] - (-1 / (1 + 1 / 8) ** 2)) < 0.1, f


def test_fine_correction_rescues_c40():
    iq, _ = generate_signal("QPSK", n_symbols=4000, snr_db=20, freq_offset_hz=12e3, seed=400)
    bad = extract_features(iq, FS, RS, 12e3 + 400, fine=False)
    good = extract_features(iq, FS, RS, 12e3 + 400, fine=True)
    assert bad[0] < 0.5 and good[0] > 0.85, (bad, good)