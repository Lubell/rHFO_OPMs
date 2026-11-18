"""SNR of OPM 13 relative to the under-eye ERG, in the time and frequency domain.

Time domain: RMS of the evoked response (0-120 ms) / RMS of the baseline.
Frequency domain: Welch PSD of the signal window / PSD of the baseline window,
per epoch, then the geometric mean over epochs. The OPM/ERG ratio is shown raw and normalised to its geometric
mean in a 30-40 Hz reference band, with bootstrap 95% CIs across participants.
Uses the 1 Hz high-pass epochs, with no further filtering.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import config as cfg

SIGNAL = (0.0, 0.12)
BASELINE = (-0.1, -0.02)     # epochs start at -100 ms
REF_BAND = (30.0, 40.0)
N_FFT = 256
N_BOOT = 2000
SEED = 7
EPS = 1e-24


def rms(evoked, window):
    mask = (evoked.times >= window[0]) & (evoked.times <= window[1])
    return np.sqrt(np.mean(evoked.data[:, mask] ** 2, axis=1))


def psd(epochs, window):
    data = epochs.copy().crop(*window).get_data()
    n = data.shape[-1]
    p, f = mne.time_frequency.psd_array_welch(data, epochs.info["sfreq"], fmin=0, fmax=500, n_fft=N_FFT,
                                              n_per_seg=n, n_overlap=n // 2, average="mean", verbose=False)
    return f, p  # (epoch, channel, freq)


def analyse(pid):
    epochs = mne.read_epochs(cfg.fname(pid, "hp1_epochs"), preload=True, verbose=False)
    epochs.pick(["mag", "eeg"]).pick([cfg.BEST_OPM, cfg.ERG_UNDER])   # -> [OPM 13, ERG under]
    evoked = epochs.average()
    snr_t = rms(evoked, SIGNAL) / (rms(evoked, BASELINE) + 1e-12)
    f, p_sig = psd(epochs, SIGNAL)
    _, p_base = psd(epochs, BASELINE)
    snr_f = np.exp(np.mean(np.log(p_sig / (p_base + EPS) + EPS), axis=0))  # geometric mean over epochs
    return snr_t, f, snr_f


def bootstrap_ci(x, rng):
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [2.5, 50, 97.5], axis=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)

    res = [analyse(pid) for pid in cfg.INCLUDED]
    f = res[0][1]
    snr_t = np.array([r[0] for r in res])                     # (participant, [opm, erg])
    ratio = np.array([r[2][0] / (r[2][1] + EPS) for r in res])  # OPM / ERG, per frequency
    ref = (f >= REF_BAND[0]) & (f <= REF_BAND[1])
    normed = ratio / np.exp(np.mean(np.log(ratio[:, ref] + EPS), axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    for data, label, alpha in [(ratio, "Raw", 0.2), (normed, "Normalised", 0.3)]:
        lo, med, hi = 10 * np.log10(bootstrap_ci(data, np.random.default_rng(SEED)))
        ax.semilogx(f, med, label=f"{label} median")
        ax.fill_between(f, lo, hi, alpha=alpha, label=f"{label} 95% CI")
    ax.axvspan(*REF_BAND, alpha=0.1, label="Reference band")
    ax.axvline(145.0, linestyle="--", alpha=0.6, label="~145 Hz observed limit")
    ax.set(xlabel="Frequency (Hz)", ylabel="Relative SNR, OPM/ERG (dB)", title="OPM/ERG SNR comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "snr_frequency.pdf")

    ratio_t = 10 * np.log10(snr_t[:, 0] / snr_t[:, 1])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratio_t, "o", label="Participants")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    med, lo, hi = np.median(ratio_t), *np.percentile(ratio_t, [2.5, 97.5])
    ax.errorbar(len(ratio_t), med, yerr=[[med - lo], [hi - med]], fmt="s", color="red", label="Median, 95% range")
    ax.set(xlabel="Participant", ylabel="Time-domain SNR ratio, OPM/ERG (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "snr_time.pdf")
    plt.show()
