"""ITPC spectrum of OPM 13 (20-50 ms) against two null references.

  - scrambled: each trial circularly shifted by a random lag, ITPC averaged over -100 to -10 ms
  - pre-stimulus: 95th percentile of the real ITPC from -100 to 0 ms, pooled over participants
Uses the 1 Hz high-pass epochs; 50 linearly spaced frequencies from 110 to 300 Hz.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import config as cfg

FREQS = np.linspace(110, 300, 50)
N_CYCLES = FREQS / 20
WINDOW = (0.02, 0.05)
SCRAMBLED_WINDOW = (-0.1, -0.01)
PRESTIM_WINDOW = (-0.1, 0.0)
PERCENTILE = 95
SEED = 0


def itpc(trials, sfreq):
    """trials: (n_trials, n_times) -> ITPC (n_freqs, n_times)."""
    tfr = mne.time_frequency.tfr_array_morlet(trials[:, None, :], sfreq, FREQS, n_cycles=N_CYCLES,
                                              output="complex", zero_mean=True, verbose=False)
    return np.abs(np.mean(np.exp(1j * np.angle(tfr)), axis=0)).squeeze()


def circular_shift(trials, rng):
    return np.array([np.roll(tr, rng.integers(trials.shape[1])) for tr in trials])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)
    real, scrambled = [], []
    for pid in cfg.INCLUDED:
        epochs = mne.read_epochs(cfg.fname(pid, "hp1_epochs"), preload=True, verbose=False)
        trials = epochs.pick(["mag", "eeg"]).get_data()[:, cfg.BEST_OPM]
        real.append(itpc(trials, epochs.info["sfreq"]))
        scrambled.append(itpc(circular_shift(trials, rng), epochs.info["sfreq"]))
    real, scrambled = np.array(real), np.array(scrambled)   # (participant, freq, time)
    times = epochs.times

    def in_window(w):
        return (times >= w[0]) & (times <= w[1])

    mean_real = real.mean(axis=0)[:, in_window(WINDOW)].mean(axis=1)
    mean_scrambled = scrambled[:, :, in_window(SCRAMBLED_WINDOW)].mean(axis=(0, 2))
    pre = real[:, :, in_window(PRESTIM_WINDOW)]
    threshold = np.percentile(np.moveaxis(pre, 1, -1).reshape(-1, len(FREQS)), PERCENTILE, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(FREQS, mean_real, label="OPM 13")
    ax.plot(FREQS, mean_scrambled, linestyle="--", label="Scrambled")
    ax.plot(FREQS, threshold, color="green", linestyle=":", label=f"Pre-stimulus {PERCENTILE}th percentile")
    ax.axvspan(120, 160, color="gray", alpha=0.3, label="120-160 Hz")
    ax.set(xlabel="Frequency (Hz)", ylabel=f"Mean ITPC, {WINDOW[0]*1e3:.0f}-{WINDOW[1]*1e3:.0f} ms",
           title="OPM 13 ITPC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "itpc_opm13_vs_null.pdf")
    plt.show()
