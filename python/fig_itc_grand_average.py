"""Grand-average ITC per sensor, z-scored against the -50-0 ms baseline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

import config as cfg
from loading import read_tfr
from plotting import ms_xaxis, sensor_grid

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmin", default="110")
    ap.add_argument("--fmax", default="145")
    ap.add_argument("--tag", default="9cycles")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    names = dict(lo=args.fmin, hi=args.fmax, tag=args.tag)
    tfrs = [read_tfr(cfg.fname(pid, "itc", **names)) for pid in cfg.INCLUDED]
    ga = mne.time_frequency.AverageTFRArray(info=tfrs[0].info, data=np.mean([t.data for t in tfrs], axis=0),
                                            times=tfrs[0].times, freqs=tfrs[0].freqs, nave=len(tfrs))

    def draw(ax, ch):
        vmax = 30 if ch >= cfg.N_OPM else 15
        ga.plot(picks=[ch], axes=ax, baseline=(-0.05, 0), mode="zscore", tmin=-0.05, tmax=0.1,
                vlim=(-vmax, vmax), colorbar=False, show=False)
        ms_xaxis(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig = sensor_grid(draw)
    fig.supxlabel("Time (ms)")
    fig.supylabel("Frequency (Hz)")
    fig.savefig(out / f"itc_grand_average_{args.fmin}_{args.fmax}_{args.tag}.pdf")
    plt.show()
