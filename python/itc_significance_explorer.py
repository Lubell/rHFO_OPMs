"""Explore when and where the group ITC statistics are significant.

Written for the Paper III revision. It answers two questions about the group
ITC test in itc_stats.py:

  1. Which criterion gives the published "OPMs 5, 12 and 13" for 110-145 Hz,
     when the test at FDR alpha 0.05 also flags OPMs 3, 8 and 10?
  2. Which sensors show significant ITC above 158 Hz in the 110-170 Hz analysis
     (the Discussion says only one sensor, OPM 2)?

It reuses load_itc() and group_test() from itc_stats.py unchanged, so the
p-values are the published ones. Nothing here changes the analysis; it only
tabulates and plots the existing statistics under several candidate criteria.

Sensors are named by their SENSOR LABEL (config.SENSOR_LABELS), as in the ITC
figures, never by channel index.

Direction: the group test is two-sided, so a significant point can be an ITC
decrease. A point counts as an increase when its U statistic exceeds the null
midpoint n_participants * n_baseline / 2.

Outputs (in --out):
  itc_sig_<band>_map.pdf/png      per-sensor time-frequency map, array layout:
                                  colour = signed -log10(p_FDR) where p < 0.05
                                  (blue = increase, orange = decrease), black
                                  contour = p < 0.02; lines at 0, 50 and 75 ms
  itc_sig_<band>_raster.pdf/png   one row per sensor: number of frequencies
                                  with a significant increase at each time point;
                                  orange ticks mark significant decreases
  itc_sig_<band>_table.csv        per sensor and frequency: counts, first and last
                                  significant time, longest run, minimum p
  itc_sig_<band>_criteria.csv     sensors passing each candidate criterion
  itc_sig_summary.txt             the printed summary

Usage (from python/, after compute_itc.py for both bands):
  python itc_significance_explorer.py
  python itc_significance_explorer.py --bands 110,145,9cycles
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import config as cfg
from itc_stats import ALPHA, BASELINE, group_test, load_itc
from plotting import sensor_grid

DEFAULT_BANDS = ["110,145,9cycles", "110,170,10_perBand"]
STRICT_ALPHA = 0.02            # the "p < .02" seen in an old notebook cell
HIGH_FREQ = 158.0              # Discussion: activity "above 158 Hz"
PUBLISHED_110_145 = {5, 12, 13}

INCREASE = "#2a78d6"           # blue
DECREASE = "#eb6834"           # orange
NOT_SIG = "#e9e8e4"            # light neutral gray
INK = "#52514e"

# Candidate criteria: (name, alpha, t_start_ms (exclusive), t_end_ms (inclusive),
# increases only, minimum consecutive significant ms within one frequency)
CRITERIA = [
    ("A  p<.05, t>0 (any direction)", ALPHA, 0, 100, False, 1),
    ("B  p<.05, 0-50 ms (any direction)", ALPHA, 0, 50, False, 1),
    ("C  p<.05, 0-50 ms, increases", ALPHA, 0, 50, True, 1),
    ("D  p<.05, 0-75 ms, increases", ALPHA, 0, 75, True, 1),
    ("E  p<.05, 0-50 ms, increases, run >= 5 ms", ALPHA, 0, 50, True, 5),
    ("F  p<.05, 0-50 ms, increases, run >= 10 ms", ALPHA, 0, 50, True, 10),
    ("G  p<.02, t>0 (any direction)", STRICT_ALPHA, 0, 100, False, 1),
    ("H  p<.02, 0-50 ms (any direction)", STRICT_ALPHA, 0, 50, False, 1),
    ("I  p<.02, 0-50 ms, increases", STRICT_ALPHA, 0, 50, True, 1),
    ("J  p<.02, 0-75 ms, increases", STRICT_ALPHA, 0, 75, True, 1),
]


def label(ch):
    lab = cfg.SENSOR_LABELS[ch]
    return f"OPM {lab}" if ch < cfg.N_OPM else str(lab)


def sensor_order():
    """Channel indices ordered by sensor label 1-14, then the two ERG electrodes."""
    opms = sorted(range(cfg.N_OPM), key=lambda ch: cfg.SENSOR_LABELS[ch])
    return opms + [cfg.ERG_UPPER, cfg.ERG_UNDER]


def longest_run(mask):
    """Longest run of True along the last axis."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def compute(names):
    template, itc = load_itc(names)
    test, stat, p_fdr = group_test(itc, template.times)
    t_ms = np.round(template.times[test] * 1e3).astype(int)
    n_part = itc.shape[0]
    n_base = n_part * int(np.sum((t_ms >= BASELINE[0] * 1e3) & (t_ms <= BASELINE[1] * 1e3)))
    increase = stat > n_part * n_base / 2
    return dict(freqs=template.freqs, t_ms=t_ms, stat=stat, p=p_fdr, inc=increase,
                n_part=n_part, n_base=n_base)


def sig_mask(r, alpha, t0, t1, inc_only, min_run):
    """(channel, freq, time) mask for one criterion."""
    window = (r["t_ms"] > t0) & (r["t_ms"] <= t1)
    m = (r["p"] < alpha) & window
    if inc_only:
        m &= r["inc"]
    if min_run > 1:
        keep = np.zeros_like(m)
        for ch in range(m.shape[0]):
            for f in range(m.shape[1]):
                row, start = m[ch, f], None
                for i, v in enumerate(np.append(row, False)):
                    if v and start is None:
                        start = i
                    elif not v and start is not None:
                        if i - start >= min_run:
                            keep[ch, f, start:i] = True
                        start = None
        m = keep
    return m


def write_table(r, path):
    post = r["t_ms"] > 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "channel_index", "freq_hz", "n_sig_post", "n_sig_increase_post",
                    "n_sig_decrease_post", "first_sig_ms", "last_sig_ms", "longest_run_ms",
                    "n_sig_0_50ms", "n_sig_0_75ms", "n_p02_post", "min_p_post"])
        for ch in sensor_order():
            for f, freq in enumerate(r["freqs"]):
                sig = (r["p"][ch, f] < ALPHA) & post
                if not sig.any():
                    continue
                t = r["t_ms"][sig]
                w.writerow([label(ch), ch, f"{freq:.1f}", int(sig.sum()),
                            int((sig & r["inc"][ch, f]).sum()), int((sig & ~r["inc"][ch, f]).sum()),
                            int(t[0]), int(t[-1]), longest_run(sig),
                            int((sig & (r["t_ms"] <= 50)).sum()), int((sig & (r["t_ms"] <= 75)).sum()),
                            int(((r["p"][ch, f] < STRICT_ALPHA) & post).sum()),
                            f"{r['p'][ch, f][post].min():.2g}"])


def evaluate_criteria(r, tag, path, log):
    rows = []
    for name, alpha, t0, t1, inc_only, min_run in CRITERIA:
        m = sig_mask(r, alpha, t0, t1, inc_only, min_run)
        passing = [ch for ch in sensor_order() if m[ch].any()]
        opm_labels = {cfg.SENSOR_LABELS[ch] for ch in passing if ch < cfg.N_OPM}
        detail = []
        for ch in passing:
            fs = r["freqs"][m[ch].any(axis=1)]
            ts = r["t_ms"][m[ch].any(axis=0)]
            detail.append(f"{label(ch)} [{fs.min():.1f}-{fs.max():.1f} Hz, {ts.min()}-{ts.max()} ms]")
        match = ""
        if tag.startswith("110_145"):
            match = "MATCHES 5, 12, 13" if opm_labels == PUBLISHED_110_145 else ""
        log(f"  {name:45s} OPMs {sorted(opm_labels)} {match}")
        for d in detail:
            log(f"      {d}")
        rows.append([name, " ".join(str(x) for x in sorted(opm_labels)), match, "; ".join(detail)])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["criterion", "opm_sensor_labels", "matches_published", "detail"])
        w.writerows(rows)


def high_freq_report(r, log):
    """Sensors with any significant point (p < .05, t > 0) above HIGH_FREQ."""
    hi = r["freqs"] > HIGH_FREQ
    if not hi.any():
        return
    log(f"  Significant points above {HIGH_FREQ:g} Hz (p < .05, t > 0 ms):")
    found = False
    for ch in sensor_order():
        m = (r["p"][ch][hi] < ALPHA) & (r["t_ms"] > 0)
        if m.any():
            found = True
            fs = r["freqs"][hi][m.any(axis=1)]
            ts = r["t_ms"][m.any(axis=0)]
            n_inc = int((m & r["inc"][ch][hi]).sum())
            log(f"      {label(ch)}: {fs.min():.1f}-{fs.max():.1f} Hz, {ts.min()}-{ts.max()} ms, "
                f"{int(m.sum())} points ({n_inc} increases); "
                f"below {HIGH_FREQ:g} Hz also significant: {bool(((r['p'][ch][~hi] < ALPHA) & (r['t_ms'] > 0)).any())}")
    if not found:
        log("      none")


def signed_map(r):
    with np.errstate(divide="ignore"):
        v = -np.log10(r["p"])
    v = np.where(r["inc"], v, -v)
    return np.where(r["p"] < ALPHA, v, np.nan)


def plot_map(r, title):
    vals = signed_map(r)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 3
    cmap = LinearSegmentedColormap.from_list("signed", [DECREASE, "#f6d2c3", NOT_SIG, "#c7dbf3", INCREASE])
    cmap.set_bad(NOT_SIG)
    fidx = np.arange(len(r["freqs"]))
    tick_idx = fidx[:: max(1, len(fidx) // 5)]
    holder = {}

    def draw(ax, ch):
        holder["im"] = ax.pcolormesh(r["t_ms"], fidx, vals[ch], cmap=cmap, vmin=-vmax, vmax=vmax,
                                     shading="nearest")
        strict = (r["p"][ch] < STRICT_ALPHA).astype(float)
        if strict.any():
            ax.contour(r["t_ms"], fidx, strict, levels=[0.5], colors="k", linewidths=0.8)
        for t in (0, 50, 75):
            ax.axvline(t, color=INK, linewidth=0.6, linestyle="--" if t else "-")
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{r['freqs'][i]:.0f}" for i in tick_idx])
        ax.set_xticks([-50, 0, 50, 100])
        ax.tick_params(labelsize=7)

    fig = sensor_grid(draw)
    for ax in fig.axes:
        t = ax.get_title()
        if t and t.isdigit():
            ax.set_title(f"OPM {t}", fontsize=9)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (Hz)")
    fig.subplots_adjust(top=0.92)
    fig.suptitle(title, fontsize=12, y=0.985)
    cb = fig.colorbar(holder["im"], ax=fig.axes, shrink=0.5, location="right")
    cb.set_label("signed -log10(p FDR), shown where p < 0.05\nblue: ITC increase, orange: decrease")
    return fig


def plot_raster(r, title):
    order = sensor_order()
    m = (r["p"] < ALPHA)
    n_inc = (m & r["inc"]).sum(axis=1)[order]           # (sensor, time)
    dec = (m & ~r["inc"]).any(axis=1)[order]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = LinearSegmentedColormap.from_list("count", [NOT_SIG, "#c7dbf3", INCREASE, "#0d3a73"])
    im = ax.pcolormesh(r["t_ms"], np.arange(len(order)), n_inc, cmap=cmap, vmin=0,
                       vmax=len(r["freqs"]), shading="nearest")
    yy, xx = np.nonzero(dec)
    ax.scatter(r["t_ms"][xx], yy, marker="|", s=40, color=DECREASE, linewidths=1.5,
               label="significant decrease")
    for t in (0, 50, 75):
        ax.axvline(t, color=INK, linewidth=0.8, linestyle="--" if t else "-")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([label(ch) for ch in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title(title, fontsize=11, loc="left")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f"number of frequencies (of {len(r['freqs'])}) with a significant increase")
    if dec.any():
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", nargs="+", default=DEFAULT_BANDS, help="fmin,fmax,tag")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    for spec in args.bands:
        lo, hi, tag = spec.split(",")
        r = compute(dict(lo=lo, hi=hi, tag=tag))
        key = f"{lo}_{hi}_{tag}"
        log(f"\n=== ITC {lo}-{hi} Hz ({tag}): {r['n_part']} participants, "
            f"{r['n_base']} pooled baseline values, FDR alpha {ALPHA} ===")
        log(f"  frequencies: {np.round(r['freqs'], 1).tolist()}")
        evaluate_criteria(r, key, out / f"itc_sig_{key}_criteria.csv", log)
        high_freq_report(r, log)
        write_table(r, out / f"itc_sig_{key}_table.csv")
        title = f"Group ITC significance, {lo}-{hi} Hz (N = {r['n_part']})"
        for kind, fig in (("map", plot_map(r, title)), ("raster", plot_raster(r, title))):
            for ext in ("pdf", "png"):
                fig.savefig(out / f"itc_sig_{key}_{kind}.{ext}", dpi=200)
            plt.close(fig)

    (out / "itc_sig_summary.txt").write_text("\n".join(lines) + "\n")
