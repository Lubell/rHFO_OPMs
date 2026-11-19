"""Morlet-wavelet power and inter-trial phase coherence (ITC) per participant.

Computed on the unfiltered merged epochs. Frequencies are log-spaced and the
number of cycles scales with frequency: n_cycles = freqs / cycle_divisor.
Defaults give the files used in itc_stats.py (<pid>_itc_110_170_10_perBand.hdf5).
The older 110-145 Hz files used --fmax 145 --n-freqs 10 --divisor 9 --tag 9cycles.
Figure 5b (110-300 Hz): --fmax 300 --n-freqs 35 --divisor 15 --tag 15cycles6.

Usage: python compute_itc.py [--fmin 110 --fmax 170 --n-freqs 20 --divisor 10 --tag 10_perBand]
"""

import argparse

import numpy as np

import config as cfg
from loading import merged_epochs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", default=cfg.PARTICIPANTS)
    ap.add_argument("--fmin", type=float, default=110)
    ap.add_argument("--fmax", type=float, default=170)
    ap.add_argument("--n-freqs", type=int, default=20)
    ap.add_argument("--divisor", type=float, default=10)
    ap.add_argument("--tag", default="10_perBand")
    args = ap.parse_args()

    freqs = np.logspace(np.log10(args.fmin), np.log10(args.fmax), args.n_freqs)
    names = dict(lo=f"{args.fmin:g}", hi=f"{args.fmax:g}", tag=args.tag)
    for pid in args.participants:
        epochs = merged_epochs(pid)
        power, itc = epochs.compute_tfr("morlet", freqs, n_cycles=freqs / args.divisor,
                                        return_itc=True, average=True)
        itc.save(cfg.fname(pid, "itc", **names), overwrite=True)
        power.save(cfg.fname(pid, "power", **names), overwrite=True)
