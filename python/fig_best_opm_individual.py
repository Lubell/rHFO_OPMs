"""Figure 4B: OPM 13 response (110-145 Hz) of each participant, -50 to 100 ms.

Needs the 110-145 Hz evoked files: python preprocess.py --band 110 145
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import config as cfg
from loading import read_evoked

WINDOW = slice(150, 300)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    out = Path(ap.parse_args().out)
    out.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    for pid in cfg.PARTICIPANTS:
        ev = read_evoked(cfg.fname(pid, "high_evoked", lo="110", hi="145"))
        ax.plot(ev.times[WINDOW] * 1e3, ev.data[cfg.BEST_OPM, WINDOW] * 1e15, label=pid)
    ax.set(xlabel="Time (ms)", ylabel="Field strength (fT)", title=f"OPM {cfg.SENSOR_LABELS[cfg.BEST_OPM]}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig4b_opm13_individual.pdf")
    plt.show()
