import numpy as np
from scipy.signal import welch

from signalsentry.generator import (constellation, generate_signal, load_iq,
                                    rrc_taps, save_iq)


def test_constellations_unit_power():
    for m in ["BPSK", "QPSK", "8PSK", "16QAM"]:
        assert abs(np.mean(np.abs(constellation(m)) ** 2) - 1) < 1e-9


def test_rrc_unit_energy():
    assert abs(np.sum(rrc_taps() ** 2) - 1) < 1e-9


def test_snr_matches_request():
    # Noise-free signal power is 1, so total power = 1 + 10^(-snr/10).
    for snr in [0, 10, 20]:
        iq, _ = generate_signal("QPSK", n_symbols=20000, snr_db=snr, seed=1)
        expected = 1 + 10 ** (-snr / 10)
        assert abs(np.mean(np.abs(iq) ** 2) / expected - 1) < 0.03


def test_spectrum_peak_at_offset():
    iq, meta = generate_signal("QPSK", n_symbols=20000, snr_db=20, freq_offset_hz=50e3, seed=2)
    f, p = welch(iq, fs=meta["sample_rate_hz"], nperseg=1024, return_onesided=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    centroid = np.sum(f * p) / np.sum(p)
    assert abs(centroid - 50e3) < 3e3


def test_save_load_roundtrip(tmp_path):
    iq, meta = generate_signal("BPSK", n_symbols=500, seed=3)
    save_iq(iq, meta, tmp_path / "a.iq")
    back = load_iq(tmp_path / "a.iq")
    assert back.dtype == np.complex64 and len(back) == len(iq)
    assert np.allclose(back, iq, atol=1e-5)