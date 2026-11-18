"""Figure 2: low-frequency (0.5-45 Hz) evoked responses.

A: under-eye ERG, every participant
B: OPM 13, every participant
C: grand average (included participants) of ERG and OPM on twin axes
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import config as cfg
from loading import read_evoked

XTICKS = np.arange(-200, 301, 50)


def panel(ax, letter):
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(XTICKS)
    ax.set_xlabel("Time (ms)")
    ax.text(0.01, 0.99, letter, transform=ax.transAxes, fontsize=24, fontweight="bold", va="top", ha="left")


def single_traces(evokeds, ch, scale, ylabel, letter):
    fig, ax = plt.subplots()
    for pid, ev in evokeds.items():
        ax.plot(ev.times * 1e3, ev.data[ch] * scale, label=pid)
    ax.set_ylabel(ylabel)
    panel(ax, letter)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)

    # Panels A and B show all recorded participants, as in the published figure
    evokeds = {pid: read_evoked(cfg.fname(pid, "low_evoked")) for pid in cfg.PARTICIPANTS}
    single_traces(evokeds, cfg.ERG_UNDER, 1e6, "Amplitude (µV)", "A").savefig(out / "fig2a_erg.pdf")
    single_traces(evokeds, cfg.BEST_OPM, 1e12, "Field strength (pT)", "B").savefig(out / "fig2b_opm.pdf")

    ga = mne.grand_average([evokeds[pid] for pid in cfg.INCLUDED])
    ga.apply_baseline((-0.1, 0))
    t = ga.times[:-1] * 1e3
    fig, ax1 = plt.subplots()
    ax1.plot(t, ga.data[cfg.ERG_UNDER, :-1] * 1e6, color="tab:red")
    ax1.set_ylabel("Amplitude (µV)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    # NOTE: panel C uses OPM index 13 (panels A/B use index 12), as in the published figure
    ax2.plot(t, ga.data[13, :-1] * 1e12, color="tab:blue")
    ax2.set_ylabel("Field strength (pT)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    panel(ax1, "C")
    fig.savefig(out / "fig2c_grand_average.pdf")
    plt.show()
