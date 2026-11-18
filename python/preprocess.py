"""Make the per-participant evoked and epochs files used by all later analyses.

For each participant this writes (into the participant folder):
  <pid>_lowF_1_45.fif                    evoked, 0.5-45 Hz, baseline -50-0 ms (a-/b-wave)
  <pid>_highF_75_145.fif                 evoked, 75-145 Hz band-pass (oscillatory potentials)
  <pid>_highF_env_ev130_145.fif          130-145 Hz Hilbert envelope of the unfiltered evoked
  <pid>_highpassOnly_1Hz-epo.fif         epochs, 1 Hz high-pass, -100-150 ms
  <pid>_highpassOnly_1Hz_evoked-ave.fif  their average
Extra high bands (e.g. 110-145 for fig_best_opm_individual.py) can be added with --band.

Usage:
  python preprocess.py [--participants PO13 PO14] [--band 110 145]
  python preprocess.py --inspect PO13   # browse epochs to pick rejections
"""

import argparse

import config as cfg
from loading import merged_epochs


def inspect_epochs(pid):
    """Browse merged epochs (2 Hz high-pass) to choose the epochs listed in rejected_epochs.py."""
    epochs = merged_epochs(pid, reject=False)
    epochs.filter(l_freq=2, h_freq=None)
    epochs.plot(block=True)


def preprocess(pid, extra_bands=()):
    low = merged_epochs(pid, *cfg.LOW_BAND, baseline=(-0.05, 0))
    low.average().save(cfg.fname(pid, "low_evoked"), overwrite=True)

    for band in [cfg.HIGH_BAND, *extra_bands]:
        high = merged_epochs(pid, *band)
        high.average().save(cfg.fname(pid, "high_evoked", **cfg.band_kw(band)), overwrite=True)

    # envelope: filter the evoked response (not the trials), then Hilbert
    evoked = merged_epochs(pid).average()
    evoked.filter(*cfg.ENVELOPE_BAND).apply_hilbert(envelope=True)
    evoked.save(cfg.fname(pid, "env_evoked", **cfg.band_kw(cfg.ENVELOPE_BAND)), overwrite=True)

    hp = merged_epochs(pid, 1.0, None, tmin=-0.1, tmax=0.15)
    hp.save(cfg.fname(pid, "hp1_epochs"), overwrite=True)
    hp.average().save(cfg.fname(pid, "hp1_evoked"), overwrite=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", nargs="+", default=cfg.PARTICIPANTS)
    ap.add_argument("--band", nargs=2, type=float, action="append", default=[],
                    metavar=("LO", "HI"), help="additional high band-pass evoked to save")
    ap.add_argument("--inspect", metavar="PID")
    args = ap.parse_args()
    if args.inspect:
        inspect_epochs(args.inspect)
    else:
        for pid in args.participants:
            preprocess(pid, [tuple(b) for b in args.band])
