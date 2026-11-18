"""Export data to .mat files for the MATLAB mutual-information analyses.

Per participant, <data dir>/<pid>_<name>.mat with variable `evoked` (16 channels x 501 samples):
  highF_75_145        75-145 Hz band-pass
  highF_env_ev130_145 130-145 Hz envelope
Group: <data dir>/itc_110_145_9cycles.mat with variable `itc`
  (included participants x 16 channels x 10 freqs x 501 samples), from compute_itc.py step 3b.
"""

import numpy as np
import scipy.io

import config as cfg
from loading import read_evoked, read_tfr

EXPORTS = {
    "highF_75_145": ("high_evoked", cfg.band_kw(cfg.HIGH_BAND)),
    "highF_env_ev130_145": ("env_evoked", cfg.band_kw(cfg.ENVELOPE_BAND)),
}

if __name__ == "__main__":
    for pid in cfg.PARTICIPANTS:
        for name, (kind, kw) in EXPORTS.items():
            evoked = read_evoked(cfg.fname(pid, kind, **kw))
            scipy.io.savemat(cfg.data_dir() / f"{pid}_{name}.mat", {"evoked": evoked.data})

    itc_names = dict(lo="110", hi="145", tag="9cycles")
    itc = np.stack([read_tfr(cfg.fname(pid, "itc", **itc_names)).data for pid in cfg.INCLUDED])
    scipy.io.savemat(cfg.data_dir() / "itc_110_145_9cycles.mat", {"itc": itc})
