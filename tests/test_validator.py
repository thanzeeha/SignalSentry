import numpy as np
import pytest

from signalsentry.classifier import load_model
from signalsentry.generator import generate_signal
from signalsentry.validator import analyze

FS, RS = 1e6, 125e3


@pytest.fixture(scope="module")
def clf():
    return load_model()      # run `python -m signalsentry.classifier` first


def run(clf, mod, snr, off, seed):
    iq, _ = generate_signal(mod, 4000, snr, off, seed=seed)
    return analyze(iq.astype(np.complex64), FS, RS, clf)


def rules(r, sev="HARD"):
    return {f["rule"] for f in r["flags"] if f["severity"] == sev}


def test_known_signals_validated(clf):
    for mod in ["BPSK", "QPSK", "16QAM"]:
        r = run(clf, mod, 15, 9e3, 777)
        assert r["verdict"] == "VALIDATED" and r["score"] >= 90, (mod, r["score"], r["flags"])


def test_8psk_flagged_unfamiliar(clf):
    r = run(clf, "8PSK", 15, 5e3, 778)
    assert r["verdict"] == "HUMAN REVIEW REQUIRED" and "R7" in rules(r), r["flags"]


def test_low_snr_hard_fail_r1(clf):
    r = run(clf, "QPSK", -3, 5e3, 779)
    assert r["verdict"] == "HUMAN REVIEW REQUIRED" and "R1" in rules(r)


def test_noise_only(clf):
    rng = np.random.default_rng(1)
    iq = ((rng.standard_normal(32000) + 1j * rng.standard_normal(32000)) / np.sqrt(2)).astype(np.complex64)
    r = analyze(iq, FS, RS, clf)
    assert r["verdict"] == "HUMAN REVIEW REQUIRED" and r["flags"][0]["rule"] == "DETECT"


def test_every_flag_has_reason(clf):
    r = run(clf, "8PSK", 10, 0, 780)
    assert r["flags"] and all(len(f["reason"]) > 20 for f in r["flags"])


def test_wrong_symbol_rate_flagged_by_r3(clf):
    iq, _ = generate_signal("QPSK", 4000, 12, 5e3, seed=781)
    iq = iq.astype(np.complex64)
    for typed in [100e3, 150e3]:
        r = analyze(iq, FS, typed, clf)
        assert r["verdict"] == "HUMAN REVIEW REQUIRED" and "R3" in rules(r), (typed, r["flags"])
    ok = analyze(iq, FS, 125e3, clf)
    assert "R3" not in rules(ok) and ok["verdict"] == "VALIDATED"