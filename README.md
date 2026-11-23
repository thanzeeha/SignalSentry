# SignalSentry: viva answers (honest versions)

## 60-second summary
An AI guess about a signal's modulation is only useful if you know when to distrust it.
SignalSentry measures the signal independently (bandwidth, SNR, higher-order statistics) and
checks the guess against those measurements. On synthetic data it caught 160/160 held-out 8PSK
signals at >= 5 dB where a confidence threshold caught 0. It is a prototype: synthetic data,
one real file, symbol rate typed by the user.

## Terms (plain words)
- **IQ**: each sample is a complex number (I = real, Q = imaginary) describing the radio wave.
- **PSD**: power at each frequency. **Welch**: average the spectra of many short chunks to smooth noise.
- **SNR**: signal power / noise power (here over the whole sampled band, so the matched filter
  later makes it about 9 dB better at symbol level with 8 samples/symbol).
- **RRC filter + matched filter**: shape pulses at the transmitter; the same filter at the
  receiver maximises SNR at the symbol instants.
- **Cumulant**: a statistic from powers of the samples; each constellation has fixed values
  (BPSK |C40|=2, C42=-2; QPSK 1, -1; 16QAM 0.68, -0.68; 8PSK 0, -1).

## Questions you will get
**Why does C40 need the fine offset fix?** C40 changes with rotation. After the PSD centroid, my
centre estimate was off by ~400-600 Hz and even 50 Hz error dropped |C40| from 1.0 to 0.05.
C42 ignores rotation, so it survives. I added an s^4 spectral-line search to remove the leftover.

**Why is 8PSK a good test?** Its C42 equals QPSK's (-1). Only |C40| differs (0 vs 1). The classifier
called it QPSK with 0.95 probability, so only a physics check can catch it.

**What does noise do to cumulants?** Shrinks them by (1 + 1/s)^-2 where s is symbol-level SNR.
Theory values are corrected by this factor before R4 compares.

**Why train over a range of SNRs / separate seeds?** So the model learns the noise bias, and
testing on the training generator would overstate accuracy. Test used different seeds, offsets
and length, but it is still the same simulator.

**Your accuracy is 97.8%. Is that good?** It is an average over easy cases. All errors are at
-5 dB. Report accuracy per SNR.

**Does the validator improve accuracy?** No. At >= 5 dB the classifier is never wrong. It
helps with unfamiliar signals (8PSK) and bad inputs (wrong symbol rate).

**Why does the validator reject everything below 0 dB?** R1 is a hard rule. That is policy, not
cleverness. At nominal 0 dB all files fail because measured SNR reads slightly below 0.

**What is the circularity problem?** The classifier and R4 use the same cumulants, so R4 cannot
catch errors that corrupt the cumulants themselves. Typing the wrong symbol rate did exactly
that (58/360 passed). R3 compares an independent measurement (bandwidth) with the typed rate and
brings that to 0/360, for rate errors of about 10% or more.

**How were the weights and thresholds chosen?** Weights by judgement, not fitted. Tolerances
from a separate calibration set. Both come from synthetic data, so real data will need
recalibration.

**What happened on the real file?** Flagged. Bandwidth 3.8 kHz vs 1.4 expected, signal cut off,
shape not RRC. The classifier said 16QAM, probably wrong (FUNcube-1 is BPSK). It was audio, not
raw IQ, and not a blind test. One file proves nothing about real-world performance.

**What would you do next?** Timing/roll-off estimation instead of typing the symbol rate; a real
labelled IQ dataset; recalibrate tolerances on it; handle multiple signals and Doppler.

## Do not say
- "97.8% accuracy" as a headline.
- "The validator caught all errors" (R1 rejects everything below 0 dB).
- "Works on real signals."
- That R3 improved low-SNR error detection.

## Demo order
1. `QPSK_12dB`: VALIDATED, 100/100.
2. `8PSK_15dB`: flagged as unfamiliar although the classifier says QPSK at 0.95.
3. A QPSK file at -3 dB: R1.
4. `QPSK_12dB` with symbol rate typed as 100000: R3.
5. `ao73.iq`: flagged, with the three disclosures (audio not IQ, not blind, cut-off signal).
Say at the start: results are on synthetic data.