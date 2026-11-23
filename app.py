"""Module 7: SignalSentry Streamlit page. Run with: streamlit run app.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from signalsentry.classifier import MODEL_PATH, load_model
from signalsentry.measure import compute_psd
from signalsentry.validator import analyze

MAX_SAMPLES = 2_000_000
MIN_SYMBOLS = 500

st.set_page_config(page_title="SignalSentry", layout="wide")
st.title("SignalSentry")
st.caption("Checks an AI modulation guess against physical measurements of the signal.")
st.warning("Validated on SYNTHETIC data only (BPSK, QPSK, 16QAM; 8PSK held out as the unfamiliar signal). "
           "Behaviour on real recordings is not established.")


@st.cache_resource
def get_model():
    return load_model()


if not Path(MODEL_PATH).exists():
    st.error("Classifier not found. Run `python -m signalsentry.classifier` once, then reload.")
    st.stop()

demo_files = sorted(Path("data/demo").glob("*.iq"))
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    upload = st.file_uploader("IQ recording (.iq, complex64: float32 I then Q, interleaved)", type=["iq", "cf32", "bin"])
    demo = st.selectbox("...or use a demo file", ["(none)"] + [p.name for p in demo_files])
with col2:
    fs = st.number_input("Sample rate (Hz)", min_value=1_000.0, value=1_000_000.0, step=1_000.0, format="%.0f")
with col3:
    rs = st.number_input("Symbol rate (Hz)", min_value=100.0, value=125_000.0, step=1_000.0, format="%.0f")
st.caption("You must supply the symbol rate. A wrong value is usually caught by rule R3 (bandwidth check) but only if it is off by more than about 10%.")

run = st.button("Analyse", type="primary")

if run:
    if upload is not None:
        raw, name = upload.getvalue(), upload.name
    elif demo != "(none)":
        raw, name = (Path("data/demo") / demo).read_bytes(), demo
    else:
        st.error("Upload a file or pick a demo file first.")
        st.stop()

    if len(raw) % 8 != 0 or len(raw) == 0:
        st.error(f"File size ({len(raw)} bytes) is not a multiple of 8, so it is not complex64 IQ.")
        st.stop()
    iq = np.frombuffer(raw, dtype=np.complex64)
    if not np.all(np.isfinite(iq)):
        st.error("File contains NaN/inf values: wrong format or corrupted file.")
        st.stop()
    if len(iq) > MAX_SAMPLES:
        st.info(f"Using the first {MAX_SAMPLES:,} of {len(iq):,} samples.")
        iq = iq[:MAX_SAMPLES]
    sps = fs / rs
    if sps < 2:
        st.error(f"Sample rate / symbol rate = {sps:.2f}. Need at least 2 samples per symbol.")
        st.stop()
    if len(iq) / sps < MIN_SYMBOLS:
        st.error(f"Only {len(iq) / sps:.0f} symbols in the file; need at least {MIN_SYMBOLS}.")
        st.stop()
    if sps < 4:
        st.warning(f"Only {sps:.1f} samples per symbol: measurements may be unreliable.")

    with st.spinner("Analysing..."):
        r = analyze(iq, fs, rs, get_model())

    # ---- verdict ----
    if r["verdict"] == "VALIDATED":
        st.success(f"VALIDATED   |   trust score {r['score']:.0f}/100")
    else:
        st.error(f"HUMAN REVIEW REQUIRED   |   trust score {r['score']:.0f}/100")
    if r["flags"]:
        st.subheader("Reasons")
        for f in r["flags"]:
            (st.error if f["severity"] == "HARD" else st.warning)(f"**{f['rule']}** ({f['severity']}): {f['reason']}")
    else:
        st.write("No flags raised.")

    # ---- PSD plot ----
    m = r["measurements"]
    f, p = compute_psd(iq, fs)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(f / 1e3, 10 * np.log10(p + 1e-20), lw=0.8)
    ax.axhline(10 * np.log10(m["noise_floor"]), color="gray", ls="--", lw=0.8, label="noise floor")
    if m["detected"]:
        ax.axvspan(m["f_low_hz"] / 1e3, m["f_high_hz"] / 1e3, color="orange", alpha=0.2, label="99% power band")
        ax.axvline(m["centre_hz"] / 1e3, color="red", lw=0.8, label="centre")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(f"Power spectrum: {name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # ---- parameter table ----
    st.subheader("Measured parameters")
    if m["detected"]:
        rows = {
            "Centre frequency": f"{m['centre_hz'] / 1e3:+.2f} kHz",
            "Bandwidth (99% power)": f"{m['bandwidth_hz'] / 1e3:.1f} kHz",
            "SNR (whole-band)": f"{m['snr_db']:.1f} dB",
            "Classifier guess": f"{r['predicted']}  ({', '.join(f'{k} {v:.2f}' for k, v in r['probabilities'].items())})",
            "|C40|": f"{r['features']['abs_C40']:.3f}",
            "C42": f"{r['features']['C42']:.3f}",
            "Envelope variance": f"{r['features']['env_var']:.4f}",
            "Distance to predicted class": f"{r['distance_to_predicted']:.3f}  (tolerance {r['tolerance']:.3f})",
            "Nearest known class": f"{r['nearest_known']}  (distance {r['distance_to_nearest']:.3f})",
        }
        st.table({"Parameter": list(rows.keys()), "Value": list(rows.values())})
        st.caption("Rule scores (0-1): " + ", ".join(f"{k} {v:.2f}" for k, v in r["subscores"].items()))
    else:
        st.write("No signal detected above the noise floor.")