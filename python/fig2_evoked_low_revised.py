"""Figure 2 (revised): low-frequency (0.5-45 Hz) flash-evoked responses, N = 9.

Replaces fig_evoked_low.py for the Paper III revision. Differences from the
published figure:
  - only the 9 included participants (PO12 is no longer in panels A and B)
  - the same sensor, OPM 13 (channel index 12), in every OPM panel
    (the published panel C plotted channel index 13, which is OPM 12)
  - 2 x 2 layout without a second y-axis: ERG on the left, OPM 13 on the right,
    individual participants on top, grand average below
  - inset zoom on the a-wave (0-40 ms) in the upper right of the OPM panels
    (Reviewer 1); the y-axis gets headroom so the inset covers no data
  - the grand-average baseline (-100 to 0 ms) is applied to every panel

It also prints, and writes to fig2_values.csv, the numbers the Results text needs:
  - a- and b-wave peak latency and amplitude of the grand averages
  - every OPM ranked by grand-average b-wave amplitude (checks "largest response")
  - the OPM with the largest b-wave in each participant (checks the published
    statement that each participant's best sensor was used)

Peak definitions: the a-wave is the minimum (most negative sample) 10-30 ms;
the b-wave is the sample with the largest absolute amplitude 40-120 ms, reported
with its sign (OPM b-waves can be negative).

Usage (from python/): python fig2_evoked_low_revised.py [--channel 12] [--out results]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import config as cfg
from loading import read_evoked

ERG_COLOR = "#eb6834"
OPM_COLOR = "#2a78d6"
INK = "#52514e"
BASELINE = (-0.1, 0.0)
A_WIN = (0.010, 0.030)
B_WIN = (0.040, 0.120)
INSET = (0.0, 0.040)
INSET_BOX = [0.62, 0.60, 0.35, 0.34]   # axes fraction: x0, y0, width, height
XLIM = (-0.1, 0.3)


def in_window(times, win):
    """Samples within win (inclusive). MNE stores times in single precision
    (10 ms is 9.9999998 ms), so compare whole milliseconds."""
    t = np.round(times * 1e3)
    return (t >= round(win[0] * 1e3)) & (t <= round(win[1] * 1e3))


def peak(times, x, win):
    """Largest absolute amplitude in the window (b-wave)."""
    m = in_window(times, win)
    i = np.flatnonzero(m)[np.argmax(np.abs(x[m]))]
    return times[i], x[i]


def trough(times, x, win):
    """Most negative sample in the window (a-wave)."""
    m = in_window(times, win)
    i = np.flatnonzero(m)[np.argmin(x[m])]
    return times[i], x[i]


def headroom(ax, times, curves, box=INSET_BOX):
    """Raise the y-limit so that no data under the inset box overlaps it."""
    xlo, xhi = ax.get_xlim()
    t0 = (xlo + box[0] * (xhi - xlo)) / 1e3 - 0.01          # a little margin, in s
    under = np.concatenate([c[times >= t0] for c in curves])
    lo, hi = ax.get_ylim()
    need = lo + (under.max() - lo) / (box[1] - 0.12)       # room for the inset tick labels
    ax.set_ylim(lo, max(hi, need))


def draw(ax, times, traces, ga, color, ylabel, letter, title, inset=False):
    for tr in traces:
        ax.plot(times * 1e3, tr, color=color, linewidth=0.8, alpha=0.45)
    if ga is not None:
        ax.plot(times * 1e3, ga, color=color, linewidth=2)
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8)
    ax.axhline(0, color=INK, linestyle="--", linewidth=0.8)
    ax.set_xlim(XLIM[0] * 1e3, XLIM[1] * 1e3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left", pad=4, x=0.08)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="bottom", ha="right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if inset:
        headroom(ax, times, list(traces) + ([ga] if ga is not None else []))
        m = in_window(times, INSET)
        ins = ax.inset_axes(INSET_BOX)
        for tr in traces:
            ins.plot(times[m] * 1e3, tr[m], color=color, linewidth=0.6, alpha=0.45)
        if ga is not None:
            ins.plot(times[m] * 1e3, ga[m], color=color, linewidth=1.5)
        ins.axhline(0, color=INK, linestyle="--", linewidth=0.6)
        ins.set_xlim(INSET[0] * 1e3, INSET[1] * 1e3)
        ins.set_xticks([0, 20, 40])
        ins.tick_params(labelsize=7)
        ins.set_title("a-wave", fontsize=7, pad=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", type=int, default=cfg.BEST_OPM, help="OPM channel index (12 = OPM 13)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ch = args.channel
    opm_name = f"OPM {cfg.SENSOR_LABELS[ch]}"

    evokeds = {pid: read_evoked(cfg.fname(pid, "low_evoked")).apply_baseline(BASELINE)
               for pid in cfg.INCLUDED}
    times = next(iter(evokeds.values())).times
    ga = mne.grand_average(list(evokeds.values()))
    erg = np.array([ev.data[cfg.ERG_UNDER] for ev in evokeds.values()]) * 1e6   # uV
    opm = np.array([ev.data[ch] for ev in evokeds.values()]) * 1e12             # pT
    ga_erg, ga_opm = ga.data[cfg.ERG_UNDER] * 1e6, ga.data[ch] * 1e12

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    n = len(evokeds)
    draw(axes[0, 0], times, erg, None, ERG_COLOR, "Amplitude (µV)", "A", f"Under-eye ERG, each participant (N = {n})")
    draw(axes[0, 1], times, opm, None, OPM_COLOR, "Field strength (pT)", "B", f"{opm_name}, each participant (N = {n})", inset=True)
    draw(axes[1, 0], times, [], ga_erg, ERG_COLOR, "Amplitude (µV)", "C", "Under-eye ERG, grand average")
    draw(axes[1, 1], times, [], ga_opm, OPM_COLOR, "Field strength (pT)", "D", f"{opm_name}, grand average", inset=True)
    for ax in axes[1]:
        ax.set_xlabel("Time (ms)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_revised_{opm_name.replace(' ', '')}.{ext}", dpi=200)

    # ---------------------------------------------------------------- numbers for the text
    rows = []
    for name, x, unit in (("Under-eye ERG", ga_erg, "uV"), (opm_name, ga_opm, "pT")):
        for wave, find, win in (("a-wave", trough, A_WIN), ("b-wave", peak, B_WIN)):
            t, a = find(times, x, win)
            rows.append(["grand-average peak", name, wave, f"{t*1e3:.0f} ms", f"{a:.3g} {unit}"])
    opm_b = [(c, peak(times, ga.data[c] * 1e12, B_WIN)) for c in range(cfg.N_OPM)]
    for rank, (c, (t, a)) in enumerate(sorted(opm_b, key=lambda r: -abs(r[1][1])), 1):
        rows.append(["GA b-wave rank", f"OPM {cfg.SENSOR_LABELS[c]} (index {c})", rank, f"{t*1e3:.0f} ms", f"{a:.3g} pT"])
    for pid, ev in evokeds.items():
        best = max(range(cfg.N_OPM), key=lambda c: abs(peak(times, ev.data[c], B_WIN)[1]))
        t, a = peak(times, ev.data[best] * 1e12, B_WIN)
        rows.append(["best b-wave OPM per participant", pid, f"OPM {cfg.SENSOR_LABELS[best]} (index {best})",
                     f"{t*1e3:.0f} ms", f"{a:.3g} pT"])
    with open(out / "fig2_values.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "what", "detail", "latency", "amplitude"])
        w.writerows(rows)
    for r in rows:
        print("  ".join(str(v) for v in r))
    plt.show()
