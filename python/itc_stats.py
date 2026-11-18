"""Group statistics on the ITC (Figure 5).

Two-step non-parametric test, per channel and frequency:
  1. Within participant: for every time point, the Mann-Whitney U of that ITC
     value against all pre-stimulus values (-120 to -1 ms). The U value is given
     a sign: negative if the ITC is below the median of the -50-0 ms baseline.
  2. Across participants: at each time point from -50 to 100 ms, the 9 signed
     U values are compared (Mann-Whitney) with all 9 x 51 baseline (-50-0 ms) values.
p-values are FDR corrected (Benjamini-Hochberg) over time, per channel and frequency.

Usage: python itc_stats.py [--tag 10_perBand --fmin 110 --fmax 170]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.stats import fdr_correction
from scipy.stats import mannwhitneyu

import config as cfg
from loading import read_tfr
from plotting import ms_xaxis, sensor_grid

T_CROP = (-0.12, 0.236)
BASELINE = (-0.05, 0.0)
T_TEST_END = 0.1
ALPHA = 0.05
YTICKS = [110, 115.2, 117.8, 123.6, 129.1, 135.2, 138.4, 144.8, 151.6, 158.7, 162.4, 170]


def load_itc(names):
    tfrs = [read_tfr(cfg.fname(pid, "itc", **names)).crop(*T_CROP) for pid in cfg.INCLUDED]
    return tfrs[0], np.stack([t.data for t in tfrs])  # (participant, channel, freq, time)


def u_against(values, reference):
    """Mann-Whitney U of each single value against the reference samples (last axis)."""
    v = values[..., :, None]
    r = reference[..., None, :]
    return (r < v).sum(-1) + 0.5 * (r == v).sum(-1)


def group_test(itc, times):
    i0 = np.flatnonzero(np.isclose(times, BASELINE[0]))[0]
    i1 = np.flatnonzero(np.isclose(times, BASELINE[1]))[0]
    base = slice(i0, i1 + 1)                 # -50 to 0 ms
    test = np.arange(i0, i1 + 1 + round(T_TEST_END * cfg.SFREQ))  # -50 to 100 ms

    # step 1: within participant, signed by position relative to the baseline median
    u = u_against(itc, itc[..., :i1])
    sign = np.where(itc >= np.median(itc[..., base], axis=-1, keepdims=True), 1, -1)
    u_signed = u * sign

    # step 2: across participants
    pooled_base = np.moveaxis(u_signed[..., base], 0, -1).reshape(*u.shape[1:3], 1, -1)
    at_t = np.moveaxis(u_signed[..., test], 0, -1)        # (channel, freq, time, participant)
    res = mannwhitneyu(at_t, pooled_base, axis=-1)
    p_fdr = np.stack([[fdr_correction(p, ALPHA)[1] for p in ch] for ch in res.pvalue])
    return test, res.statistic, p_fdr


def plot_grid(template, test, stat, p_fdr):
    data = np.zeros(template.data.shape)
    data[..., test] = stat
    mask = np.ones(template.data.shape, bool)
    mask[..., test] = p_fdr < ALPHA
    tfr = mne.time_frequency.AverageTFRArray(info=template.info, data=data, times=template.times,
                                             freqs=template.freqs, nave=len(cfg.INCLUDED))
    in_window = (template.times >= BASELINE[0] - 1e-9) & (template.times <= T_TEST_END + 1e-9)

    def draw(ax, ch):
        vmax = 2000 if ch >= cfg.N_OPM else 1500
        tfr.plot(picks=[ch], axes=ax, tmin=BASELINE[0], tmax=T_TEST_END, baseline=BASELINE,
                 colorbar=False, mask=mask[ch][:, in_window], mask_style="contour",
                 mask_alpha=0.5, mask_cmap="twilight_shifted", vlim=(-vmax, vmax), show=False)
        ms_xaxis(ax)
        ax.set_yticks([f for f in YTICKS if template.freqs[0] <= f <= template.freqs[-1]])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig = sensor_grid(draw)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (Hz)")
    return fig


def band_profile(times, freqs, test, stat, p_fdr, ch=cfg.BEST_OPM, t_end=0.07):
    """Mean statistic per frequency from the first significant time point after 0 ms to t_end."""
    t = times[test]
    sig_any = (p_fdr[ch] < ALPHA).any(axis=0) & (t >= 0)
    if not sig_any.any():
        print(f"channel {cfg.SENSOR_LABELS[ch]}: no significant time points after 0 ms")
        return None
    window = (t >= t[sig_any][0]) & (t <= t_end)
    mean_stat = stat[ch][:, window].mean(axis=1)
    sig_freq = (p_fdr[ch][:, window] < ALPHA).any(axis=1)
    print(f"OPM {cfg.SENSOR_LABELS[ch]}: window {t[window][0]*1e3:.0f}-{t[window][-1]*1e3:.0f} ms, "
          f"significant frequencies {np.round(freqs[sig_freq], 1)}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, mean_stat)
    ax.scatter(freqs[sig_freq], mean_stat[sig_freq], color="tab:orange", zorder=3, label="significant")
    ax.set(xlabel="Frequency (Hz)", ylabel="Mean group statistic",
           title=f"OPM {cfg.SENSOR_LABELS[ch]}, {t[window][0]*1e3:.0f}-{t_end*1e3:.0f} ms")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmin", default="110")
    ap.add_argument("--fmax", default="170")
    ap.add_argument("--tag", default="10_perBand")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    names = dict(lo=args.fmin, hi=args.fmax, tag=args.tag)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    template, itc = load_itc(names)
    test, stat, p_fdr = group_test(itc, template.times)
    np.savez(out / f"itc_stats_{args.fmin}_{args.fmax}_{args.tag}.npz", times=template.times[test],
             freqs=template.freqs, stat=stat, p_fdr=p_fdr, channels=np.array(template.ch_names))

    for ch in range(16):
        for f, freq in enumerate(template.freqs):
            sig = template.times[test][p_fdr[ch, f] < ALPHA]
            if sig.size:
                print(f"channel {cfg.SENSOR_LABELS[ch]}, {freq:.1f} Hz: {sig[0]*1e3:.0f} to {sig[-1]*1e3:.0f} ms")

    plot_grid(template, test, stat, p_fdr).savefig(out / "itc_stats_grid.pdf")
    fig = band_profile(template.times, template.freqs, test, stat, p_fdr)
    if fig:
        fig.savefig(out / "itc_stats_band_profile_opm13.pdf")
    plt.show()
