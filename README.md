# rHFO_OPMs

Code to accompany the paper *High frequency oscillations measured with optically
pumped magnetometers from the human retina*.

The study records the flash-evoked electroretinogram (ERG) with
optically pumped magnetometers (OPMs) and electrodes simultaneously. Fourteen
OPMs were mounted in front of the eye, alongside two ERG electrodes (above and
below the eye). The analyses compare the low-frequency response (a-/b-wave) and
the oscillatory potentials (75–170 Hz) between the two modalities.

Data are not included in this repository.

## Requirements

- Python ≥ 3.10: `pip install -r requirements.txt`
- MATLAB (mutual-information analysis only), with
  [`mutualinformationx.m`](https://mikexcohen.com/book/) from M. X. Cohen,
  *Analyzing Neural Time Series Data* (MIT Press, 2014) on the path.

## Data layout

Set `OPM_ERG_DATA_DIR` to a folder with one sub-folder per participant:

```
<data dir>/
  PO13/
    PO13_raw.fif               OPM recording (trigger channel Input-1)
    PO13000.0100.flt(.hdr)     EEG: 4 interleaved float32 channels (E01-E03, analog trigger)
  PO10/                        PO10-PO12: EEG stored as one file per channel
    PO10.C01_0000_CH368.$$$    ...converted once with convert_split_eeg.py
```

Participants PO10–PO19 were recorded. PO12 is excluded from group analyses
because the OPMs were not initialised properly in that session.

## Pipeline

Run the Python scripts from `python/`. Results and figures go to `./results`
(change with `--out`); derivative data files go to the participant folders.

| Step | Script | Output |
|---|---|---|
| 1 | `convert_split_eeg.py` | EEG `.fif` for PO10–PO12 |
| 2 | `preprocess.py --band 110 145` | evoked files (0.5–45 Hz, 75–145 Hz, 110–145 Hz, 130–145 Hz envelope) and 1 Hz high-pass epochs |
| 3 | `compute_itc.py` | Morlet ITC and power, 110–170 Hz (20 log-spaced frequencies, n_cycles = f/10) |
| 3b | `compute_itc.py --fmax 145 --n-freqs 10 --divisor 9 --tag 9cycles` | ITC 110–145 Hz, used by `fig_itc_grand_average.py` |
| 4 | `itc_stats.py` | Group statistics on the ITC (Figure 5) |
| 5 | `export_mat.py` | `.mat` files for MATLAB |
| 6 | `matlab/mi_window.m`, `plot_mi_sensitivity_map.m` | MI between each sensor and the ERG, -10 to 120 ms (Figure 6a/b) |
| 7 | `matlab/mi_itc_over_time.m` | MI of ITC time courses over time and frequency (Figure 6c) |
| – | `matlab/mi_over_time.m` | MI of the evoked envelope in a sliding window (supplementary) |

Figures: `fig_evoked_low.py` (Figure 2), `fig_evoked_high.py` (Figure 3B),
`fig_best_opm_individual.py` (Figure 4B), `fig_itc_grand_average.py`,
`fig_amplitude_spectrum.py` (Figure 8). Supplementary comparisons:
`snr_compare.py`, `itpc_compare.py`.

### Preprocessing

Both systems sample at 1 kHz and receive the same flash trigger. Data are
epoched from −200 to 300 ms on each system's own trigger and merged. The first
10 flashes of each run are discarded, then epochs rejected by visual inspection
(`rejected_epochs.py`). Filters are applied to the continuous data before epoching.

### ITC statistics

For every channel and frequency, a within-participant Mann–Whitney U compares
each time point with the pre-stimulus period; the U values are signed by whether
the ITC lies above the baseline median. Across participants, the signed values at
each time point (−50 to 100 ms) are compared with all baseline values (−50 to 0 ms).
p-values are FDR-corrected over time.

### Mutual information

Histogram-based MI between each sensor and the under-eye ERG, on the 130–145 Hz
envelope of the evoked response, −10 to 120 ms. The bin count is the rounded
mean of Freedman–Diaconis estimates. Significance is a z-score against
surrogates in which the ERG bin sequence is split at a random point and its
segments swapped/reversed (2000 permutations per participant, 5000 for the group).
Figure 6c uses the same estimator on ITC time courses (4 bins, 1000 permutations,
27-sample sliding window), with a rank-sum test across participants against the
−50–0 ms baseline.

## Channel indices

Every evoked/epochs file has 16 channels: 14 OPMs (index 0–13), ERG above the eye
(14) and ERG below the eye (15). "OPM 13" (index 12) is the sensor with the
clearest response. Layout settings are in `python/config.py` and `matlab/study_config.m`.
