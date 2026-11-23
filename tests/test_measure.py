import numpy as np

from signalsentry.generator import generate_signal
from signalsentry.measure import measure

THEORY_BW = 125_000 * 1.35


def test_noise_only_not_detected():
    rng = np.random.default_rng(0)
    n = 32000
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    assert measure(iq, 1e6)["detected"] is False


def test_centre_frequency():
    for off in [-60e3, -15e3, 0, 20e3, 80e3]:
        iq, meta = generate_signal("QPSK", snr_db=10, freq_offset_hz=off, seed=11)
        m = measure(iq, meta["sample_rate_hz"])
        assert m["detected"] and abs(m["centre_hz"] - off) < 3e3, (off, m)


def test_snr_accuracy():
    for snr in [0, 5, 10, 15, 20]:
        iq, meta = generate_signal("16QAM", snr_db=snr, freq_offset_hz=10e3, seed=12)
        m = measure(iq, meta["sample_rate_hz"])
        assert abs(m["snr_db"] - snr) < 1.0, (snr, m["snr_db"])


def test_bandwidth_close_to_theory():
    for snr in [5, 10, 20]:
        iq, meta = generate_signal("BPSK", snr_db=snr, freq_offset_hz=-20e3, seed=13)
        m = measure(iq, meta["sample_rate_hz"])
        assert 0.85 * THEORY_BW < m["bandwidth_hz"] < 1.05 * THEORY_BW, (snr, m["bandwidth_hz"])