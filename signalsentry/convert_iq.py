"""Convert real SDR recordings into SignalSentry complex64 IQ files."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample_poly, welch


def normalize_audio(data):
    """Convert WAV integer or floating-point audio into float32."""
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype == np.uint8:
            return (data.astype(np.float32) - 128.0) / 128.0

        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max + 1))
        return data.astype(np.float32) / scale

    return data.astype(np.float32)


def read_raw(path, fmt, start=0, count=None):
    """Read complex samples from an SDR or WAV recording."""

    if fmt in ("wav", "wavreal"):
        sample_rate, data = wavfile.read(path)

        if data.ndim == 1:
            if fmt != "wavreal":
                raise SystemExit(
                    "Mono WAV requires --fmt wavreal"
                )

            # Select only the requested audio section.
            data = data[start:] if count is None else data[start:start + count]

            if len(data) == 0:
                return np.array([], dtype=np.complex64)

            audio = normalize_audio(data)

            # Convert real-valued audio into an analytic complex signal.
            iq = hilbert(audio)

            return iq.astype(np.complex64)

        if fmt == "wavreal":
            raise SystemExit(
                "wavreal requires a mono WAV recording"
            )

        if data.ndim != 2 or data.shape[1] != 2:
            raise SystemExit("wav must have 2 channels (I and Q)")

        data = data[start:] if count is None else data[start:start + count]

        if len(data) == 0:
            return np.array([], dtype=np.complex64)

        data = normalize_audio(data)
        return (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)

    if fmt == "cf32":
        return np.fromfile(
            path,
            dtype=np.complex64,
            count=-1 if count is None else count,
            offset=start * 8
        )

    spec = {
        "cu8": (np.uint8, 1),
        "cs8": (np.int8, 1),
        "cs16": (np.int16, 2),
    }

    if fmt not in spec:
        raise SystemExit(f"Unknown format: {fmt}")

    dt, nbytes = spec[fmt]

    a = np.fromfile(
        path,
        dtype=dt,
        count=-1 if count is None else 2 * count,
        offset=start * 2 * nbytes
    ).astype(np.float32)

    if fmt == "cu8":
        a = (a - 127.5) / 127.5
    elif fmt == "cs8":
        a = a / 128.0
    else:
        a = a / 32768.0

    a = a[:len(a) // 2 * 2]

    return (
        a[0::2] + 1j * a[1::2]
    ).astype(np.complex64)


def find_carrier(iq, fs):
    """Estimate the strongest frequency in a complex recording."""

    if len(iq) < 2:
        raise SystemExit("Not enough samples for carrier detection")

    signal = iq - np.mean(iq)

    frequencies, power = welch(
        signal,
        fs=fs,
        nperseg=min(65536, len(signal)),
        return_onesided=False,
        scaling="spectrum"
    )

    index = int(np.argmax(power))
    return float(frequencies[index])


def convert(
    path,
    fmt,
    fs,
    rs,
    skip_s=0.0,
    max_s=None,
    shift_hz=0.0,
    target_fs=None,
    remove_dc=False,
    auto_shift=False
):
    start = int(skip_s * fs)
    count = None if max_s is None else int(max_s * fs)

    iq = read_raw(path, fmt, start, count).astype(np.complex128)

    if len(iq) == 0:
        raise SystemExit("Nothing left after --skip-s/--max-s")

    if remove_dc or auto_shift:
        iq = iq - np.mean(iq)

    if auto_shift:
        detected = find_carrier(iq, fs)

        print(
            f"auto-shift: strongest signal found at {detected:.2f} Hz"
        )

        shift_hz = detected

    if shift_hz:
        t = np.arange(len(iq)) / fs
        iq = iq * np.exp(-2j * np.pi * shift_hz * t)

    if target_fs and abs(target_fs - fs) > 1e-6:
        if target_fs > fs:
            raise SystemExit(
                "--target-fs must not exceed the input sample rate"
            )

        ratio = Fraction(target_fs / fs).limit_denominator(2000)

        iq = resample_poly(
            iq,
            ratio.numerator,
            ratio.denominator
        )

        fs = fs * ratio.numerator / ratio.denominator

    return iq.astype(np.complex64), fs


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Convert SDR recordings into SignalSentry IQ files"
    )

    ap.add_argument("file")

    ap.add_argument(
        "--fmt",
        required=True,
        choices=["cu8", "cs8", "cs16", "cf32", "wav", "wavreal"]
    )

    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--rs", type=float, required=True)

    ap.add_argument("--skip-s", type=float, default=0.0)
    ap.add_argument("--max-s", type=float, default=None)

    ap.add_argument("--shift-hz", type=float, default=0.0)
    ap.add_argument("--auto-shift", action="store_true")

    ap.add_argument("--target-fs", type=float, default=None)
    ap.add_argument("--remove-dc", action="store_true")

    ap.add_argument("--out", required=True)

    a = ap.parse_args()

    if a.fs <= 0 or a.rs <= 0:
        ap.error("--fs and --rs must be positive")

    if a.skip_s < 0 or (a.max_s is not None and a.max_s <= 0):
        ap.error("--skip-s must be nonnegative and --max-s positive")

    iq, fs_out = convert(
        a.file,
        a.fmt,
        a.fs,
        a.rs,
        a.skip_s,
        a.max_s,
        a.shift_hz,
        a.target_fs,
        a.remove_dc,
        a.auto_shift
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    iq.tofile(out)

    meta = {
        "sample_rate_hz": fs_out,
        "symbol_rate_hz": a.rs,
        "source": Path(a.file).name,
        "n_samples": int(len(iq)),
        "note": "real-valued audio converted to analytic complex samples"
        if a.fmt == "wavreal"
        else "real recording, ground truth unknown"
    }

    out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2)
    )

    print(
        f"wrote {out}: {len(iq)} samples, "
        f"fs {fs_out:.0f} Hz, "
        f"{fs_out / a.rs:.2f} samples/symbol"
    )