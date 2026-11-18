"""Figure 8: amplitude spectral density of OPM 13 and the under-eye ERG,
post-stimulus versus pre-stimulus baseline.

Spectra are computed per participant from the 1 Hz high-pass evoked response
(baseline -50-0 ms), Hanning-tapered and zero-padded to 2048 points, and scaled
as amplitude spectral density so that windows of different length are comparable.
Each channel is compared only with its own baseline. Frequency ranges where
mean - SEM (post) exceeds the mean baseline are printed; these are descriptive
and not corrected for multiple comparisons.

Spectra of trial-averaged data contain only phase-locked activity, so their level
is far below the sensor noise density and should not be compared with it.

Usage: python fig_amplitude_spectrum.py [--equal-windows]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from loading import read_evoked

BASE_WINDOW = (-0.050, -0.005)
N_FFT = 2048
ITPC_BAND = (120, 158)   # significant ITC in OPM 13 (Figure 5a, itc_stats.py)
CHANNELS = {
    "OPM 13": dict(idx=cfg.BEST_OPM, scale=1e15, unit="fT/√Hz", color="tab:blue"),
    "Under-eye ERG": dict(idx=cfg.ERG_UNDER, scale=1e6, unit="µV/√Hz", color="tab:green"),
}


def window_indices(times, window):
    return int(np.argmin(np.abs(times - window[0]))), int(np.argmin(np.abs(times - window[1])))


def amplitude_spectral_density(x, sfreq):
    w = np.hanning(len(x))
    return np.abs(np.fft.rfft(x * w, n=N_FFT)) * np.sqrt(2.0 / (sfreq * np.sum(w ** 2)))


def ranges(mask, freqs):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [(freqs[a], freqs[b - 1]) for a, b in zip(edges[::2], edges[1::2])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--equal-windows", action="store_true", help="5-50 ms post window instead of 5-100 ms")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    post_window = (0.005, 0.050) if args.equal_windows else (0.005, 0.100)

    post = {name: [] for name in CHANNELS}
    base = {name: [] for name in CHANNELS}
    for pid in cfg.INCLUDED:
        evoked = read_evoked(cfg.fname(pid, "hp1_evoked")).apply_baseline((-0.05, 0.0))
        sfreq = evoked.info["sfreq"]
        b0, b1 = window_indices(evoked.times, BASE_WINDOW)
        p0, p1 = window_indices(evoked.times, post_window)
        for name, ch in CHANNELS.items():
            x = evoked.data[ch["idx"]] * ch["scale"]
            base[name].append(amplitude_spectral_density(x[b0:b1], sfreq))
            post[name].append(amplitude_spectral_density(x[p0:p1], sfreq))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sfreq)
    n = len(cfg.INCLUDED)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    above = {}
    for ax, (name, ch) in zip(axes, CHANNELS.items()):
        P, B = np.array(post[name]), np.array(base[name])
        mp, sp = P.mean(0), P.std(0, ddof=1) / np.sqrt(n)
        mb, sb = B.mean(0), B.std(0, ddof=1) / np.sqrt(n)
        above[name] = (mp - sp) > mb
        print(f"{name}: post above baseline at " +
              ", ".join(f"{a:.1f}-{b:.1f} Hz" for a, b in ranges(above[name], freqs)))

        ax.plot(freqs, mp, color=ch["color"],
                label=f"{name}, {post_window[0]*1e3:.0f} to {post_window[1]*1e3:.0f} ms (mean ± SEM, N = {n})")
        ax.fill_between(freqs, mp - sp, mp + sp, color=ch["color"], alpha=0.3)
        ax.plot(freqs, mb, color="tab:orange", linestyle="--",
                label=f"Pre-stimulus baseline, {BASE_WINDOW[0]*1e3:.0f} to {BASE_WINDOW[1]*1e3:.0f} ms")
        ax.fill_between(freqs, mb - sb, mb + sb, color="tab:orange", alpha=0.2)
        ax.axvspan(*ITPC_BAND, color="gray", alpha=0.3,
                   label=f"Significant ITPC in OPM 13, {ITPC_BAND[0]}-{ITPC_BAND[1]} Hz")
        ax.set(title=f"Amplitude spectral density, {name}", ylabel=f"ASD ({ch['unit']})", yscale="log")
        ax.grid(True)
        ax.legend(fontsize=8)
    both = above["OPM 13"] & above["Under-eye ERG"]
    print("Both above baseline at " + ", ".join(f"{a:.1f}-{b:.1f} Hz" for a, b in ranges(both, freqs)))

    axes[-1].set(xlabel="Frequency (Hz)", xlim=(0, 500))
    fig.tight_layout()
    tag = "equal45ms" if args.equal_windows else "post95ms"
    fig.savefig(out / f"fig8_amplitude_spectrum_{tag}.pdf")
    plt.show()
