# SignalSentry synthetic dataset

All data is SYNTHETIC (NumPy simulator in `signalsentry/generator.py`). Regenerate with
`python -m signalsentry.make_dataset`. Fixed seeds, so the files are identical every time.

| File | Rows | Purpose |
|---|---|---|
| train_features.csv | ~1800 | BPSK, QPSK, 16QAM, random SNR -5..20 dB, offsets up to +-40 kHz, 4000 symbols; the model was trained on this |
| test_features.csv | ~1080 | same 3 classes, SNR -5,0,5,10,15,20 dB, offsets up to +-70 kHz, 3000 symbols, different seeds |
| unfamiliar_8psk_features.csv | ~240 | 8PSK, never used for training; tests unfamiliar-signal detection |
| raw_examples/ | 24 files | complex64 `.iq` + `.json` ground truth, 2 per modulation per SNR (0, 10, 20 dB) |

Columns: file_id, split, modulation (label), snr_db, offset_hz, seed, n_symbols, abs_C40, C42, env_var
(the three features the classifier uses).

Not included: the larger validator-evaluation set (seeds from 2,000,000) is generated on the fly by
`python -m signalsentry.evaluate`. Real recording: `data/real/ao73.*` (one file, audio not raw IQ).