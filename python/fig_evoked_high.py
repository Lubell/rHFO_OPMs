"""Figure 3B: grand-average oscillatory potentials (75-145 Hz), every OPM with the ERG overlaid."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne

import config as cfg
from loading import read_evoked

# Free subplot slots of the 4 x 5 grid, filled with OPM 0..13 in order (as in the published figure).
# Note this is not the mounted-array layout (config.GRID_POS) used in the other sensor-grid figures.
SLOTS = [1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18]
FLIPPED = range(1, 11)          # OPM polarity flipped for display
WINDOW = slice(100, 300)        # -100 to 99 ms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)

    band = cfg.band_kw(cfg.HIGH_BAND)
    ga = mne.grand_average([read_evoked(cfg.fname(pid, "high_evoked", **band)) for pid in cfg.INCLUDED])
    t = ga.times[WINDOW] * 1e3
    erg = ga.data[cfg.ERG_UNDER, WINDOW] * 1e7   # scaled to share the axis with the OPM trace

    fig, axs = plt.subplots(4, 5, figsize=(15, 10))
    for ax in axs.flat:
        ax.axis("off")
    for opm, slot in enumerate(SLOTS):
        ax = axs.flat[slot]
        ax.axis("on")
        sign = -1 if opm in FLIPPED else 1
        ax.plot(t, erg, color="blue", label="ERG")
        ax.plot(t, sign * ga.data[opm, WINDOW] * 1e15, color="orange", label="OPM (fT)")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.text(0.01, 0.99, "B", fontsize=24, fontweight="bold", va="top", ha="left")
    fig.savefig(out / "fig3b_oscillatory_potentials.pdf")
    plt.show()
