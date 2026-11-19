"""Statistics on the evoked high-frequency amplitude (Paper III revision, reviewer IDs 10, 11, 26).

Question: is the amplitude of the evoked 75-145 Hz response larger after the
flash than before it?

All choices are fixed here, before the script is run on the data (author
approval 2026-11-19):
  signal    each participant's 75-145 Hz evoked response (Figure 3 files)
  measure   RMS within a window (the band-limited average oscillates around
            zero, so a plain mean would cancel)
  post      20 to 50 ms: the oscillatory potential window from the literature
            (Robson et al., 2022: begins at about 19-20 ms, ends at about 45-50 ms),
            not chosen from these data
  baseline  -200 to -100 ms: the 75-145 Hz FIR filter is 177 samples long and
            zero-phase, so it spreads the response about 88 ms backwards in time;
            a baseline close to 0 ms would contain leaked response
  test      one-sided Wilcoxon signed-rank (post > baseline) across the
            9 included participants, exact p
  sensors   primary: OPM 13 (selected for its b-wave, not for its high-frequency
            response). Secondary: all 14 OPMs and the under-eye ERG (positive
            control), Benjamini-Hochberg FDR across these 15 tests
  reported  p, FDR-adjusted p, median post/baseline ratio, number of
            participants with an increase

Outputs (in --out):
  evoked_high_stats.csv              one row per sensor
  evoked_high_stats_participants.csv RMS values per participant and sensor
  evoked_high_stats.pdf/png          log2(post/baseline) per participant and sensor

Usage (from python/): python stats_evoked_high.py [--out results]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mne.stats import fdr_correction
from scipy.stats import wilcoxon

import config as cfg
from loading import read_evoked

POST = (0.020, 0.050)        # s, inclusive
BASE = (-0.200, -0.100)      # s, start inclusive, end exclusive
ALPHA = 0.05
PRIMARY = "OPM 13"

INK = "#52514e"
INCREASE = "#2a78d6"
NOT_SIG = "#a8a7a2"


def label(ch):
    return f"OPM {cfg.SENSOR_LABELS[ch]}" if ch < cfg.N_OPM else "ERG under"


def channels():
    """OPMs in sensor-label order, then the under-eye ERG."""
    return sorted(range(cfg.N_OPM), key=lambda ch: cfg.SENSOR_LABELS[ch]) + [cfg.ERG_UNDER]


def rms(x, times, win, end_inclusive):
    # MNE stores times in single precision (20 ms is 19.999997 ms), so compare whole milliseconds
    t = np.round(times * 1e3)
    lo, hi = round(win[0] * 1e3), round(win[1] * 1e3)
    m = (t >= lo) & ((t <= hi) if end_inclusive else (t < hi))
    return np.sqrt(np.mean(x[..., m] ** 2, axis=-1)), int(m.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)

    band = cfg.band_kw(cfg.HIGH_BAND)
    evokeds = {pid: read_evoked(cfg.fname(pid, "high_evoked", **band)) for pid in cfg.INCLUDED}
    times = next(iter(evokeds.values())).times
    if round(times[0] * 1e3) > round(BASE[0] * 1e3):
        raise ValueError(f"epochs start at {times[0]*1e3:.0f} ms; the baseline needs {BASE[0]*1e3:.0f} ms")
    chans = channels()
    data = np.array([ev.data[chans] for ev in evokeds.values()])       # participant x channel x time
    post, n_post = rms(data, times, POST, end_inclusive=True)
    base, n_base = rms(data, times, BASE, end_inclusive=False)
    ratio = post / base

    p = np.array([wilcoxon(post[:, i], base[:, i], alternative="greater", method="exact").pvalue
                  for i in range(len(chans))])
    w = np.array([wilcoxon(post[:, i], base[:, i], alternative="greater", method="exact").statistic
                  for i in range(len(chans))])
    sig, p_fdr = fdr_correction(p, alpha=ALPHA)
    n_part = len(evokeds)

    rows = []
    for i, ch in enumerate(chans):
        rows.append(dict(sensor=label(ch), channel_index=ch, n_participants=n_part,
                         n_increase=int((ratio[:, i] > 1).sum()), median_ratio=f"{np.median(ratio[:, i]):.3f}",
                         W=f"{w[i]:.0f}", p_one_sided=f"{p[i]:.4f}", p_fdr=f"{p_fdr[i]:.4f}",
                         significant_fdr=bool(sig[i])))
    with open(out / "evoked_high_stats.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "evoked_high_stats_participants.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["participant", "sensor", "rms_post", "rms_baseline", "ratio"])
        for j, pid in enumerate(evokeds):
            for i, ch in enumerate(chans):
                wr.writerow([pid, label(ch), f"{post[j, i]:.4g}", f"{base[j, i]:.4g}", f"{ratio[j, i]:.3f}"])

    print(f"Evoked {cfg.HIGH_BAND[0]:g}-{cfg.HIGH_BAND[1]:g} Hz, RMS {POST[0]*1e3:.0f} to {POST[1]*1e3:.0f} ms "
          f"({n_post} samples) vs {BASE[0]*1e3:.0f} to {BASE[1]*1e3:.0f} ms ({n_base} samples), "
          f"one-sided Wilcoxon signed-rank, N = {n_part}, BH-FDR over {len(chans)} sensors")
    for r in rows:
        flag = "  <- primary" if r["sensor"] == PRIMARY else ""
        print(f"  {r['sensor']:10s} increase {r['n_increase']}/{n_part}  median ratio {r['median_ratio']}  "
              f"W {r['W']:>3s}  p {r['p_one_sided']}  p_FDR {r['p_fdr']}  {'*' if r['significant_fdr'] else ''}{flag}")

    # figure: log2(post/baseline) per participant and sensor
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rng = np.random.default_rng(0)                       # jitter only
    for i in range(len(chans)):
        y = np.log2(ratio[:, i])
        x = i + rng.uniform(-0.15, 0.15, size=y.size)
        color = INCREASE if sig[i] else NOT_SIG
        ax.scatter(x, y, s=22, color=color, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(y)] * 2, color=INK, linewidth=2, zorder=4)
        if sig[i]:
            ax.text(i, np.max(y) + 0.15, "*", ha="center", fontsize=14, color=INK)
    ax.axhline(0, color=INK, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([label(ch).replace("OPM ", "") for ch in chans])
    ax.set_xlabel("Sensor (OPM label; ERG under = under-eye ERG)")
    ax.set_ylabel("log2(RMS post / RMS baseline)")
    ax.set_title(f"Evoked {cfg.HIGH_BAND[0]:g}-{cfg.HIGH_BAND[1]:g} Hz amplitude, "
                 f"{POST[0]*1e3:.0f}-{POST[1]*1e3:.0f} ms vs {BASE[0]*1e3:.0f} to {BASE[1]*1e3:.0f} ms "
                 f"(N = {n_part}; * FDR p < {ALPHA})", fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"evoked_high_stats.{ext}", dpi=200)
