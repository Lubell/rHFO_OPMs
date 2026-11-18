"""Study-wide settings: participants, channels, epoching and file names."""

import os
from pathlib import Path


def data_dir() -> Path:
    """Root data folder, containing one sub-folder per participant (PO10/, PO11/, ...)."""
    path = os.environ.get("OPM_ERG_DATA_DIR")
    if not path:
        raise RuntimeError("Set the environment variable OPM_ERG_DATA_DIR to the data folder.")
    return Path(path)


def participant_dir(pid: str) -> Path:
    return data_dir() / pid


# ---------------------------------------------------------------- participants
PARTICIPANTS = [f"PO{i}" for i in range(10, 20)]
EXCLUDED = ["PO12"]  # OPMs were not initialised properly in this session
INCLUDED = [p for p in PARTICIPANTS if p not in EXCLUDED]

# PO10-PO12: EEG was stored as one file per channel and converted to .fif first
# (convert_split_eeg.py). PO13 onwards: interleaved 4-channel .flt file.
SPLIT_EEG_PARTICIPANTS = ["PO10", "PO11", "PO12"]

# ---------------------------------------------------------------- recording
SFREQ = 1000.0
OPM_STIM = "Input-1"     # trigger channel in the OPM recording
EEG_STIM = "Ana"         # analog trigger channel in the EEG recording
EEG_REF_CHANNEL = "3"    # reference electrode, noisy, dropped

# Channel order in every merged epochs/evoked file: 14 OPMs, then 2 ERG electrodes
N_OPM = 14
ERG_UPPER = 14           # electrode '1'
ERG_UNDER = 15           # electrode '2', below the eye; the ERG reference signal for most analyses
BEST_OPM = 12            # "OPM 13", the sensor with the clearest response

# Sensor labels and their position in the mounted array, as a 4 x 5 subplot index (1-based)
SENSOR_LABELS = [8, 6, 11, 5, 4, 1, 2, 7, 3, 9, 10, 14, 13, 12, "ERG upper", "ERG under"]
GRID_POS = [10, 8, 15, 7, 6, 2, 3, 9, 4, 13, 14, 19, 18, 17, 11, 16]

# ---------------------------------------------------------------- epoching
EPOCH_TMIN, EPOCH_TMAX = -0.2, 0.3
N_SKIP_EPOCHS = 10       # first 10 flashes of every run are discarded
MIN_EVENT_DURATION = 0.001

LOW_BAND = (0.5, 45.0)   # a-/b-wave
HIGH_BAND = (75.0, 145.0)  # oscillatory potentials
ENVELOPE_BAND = (130.0, 145.0)


# ---------------------------------------------------------------- derivative file names
def fname(pid: str, kind: str, **kw) -> Path:
    """Path of a derivative file. Names match those used for the published analysis."""
    names = {
        "low_evoked": f"{pid}_lowF_1_45.fif",
        "high_evoked": "{pid}_highF_{lo}_{hi}.fif",
        "env_evoked": "{pid}_highF_env_ev{lo}_{hi}.fif",
        "hp1_epochs": f"{pid}_highpassOnly_1Hz-epo.fif",
        "hp1_evoked": f"{pid}_highpassOnly_1Hz_evoked-ave.fif",
        "itc": "{pid}_itc_{lo}_{hi}_{tag}.hdf5",
        "power": "{pid}_power_{lo}_{hi}_{tag}.hdf5",
    }
    name = names[kind].format(pid=pid, **kw)
    return participant_dir(pid) / name


def band_kw(band):
    """(75.0, 145.0) -> dict(lo=75, hi=145), for file names."""
    return dict(lo=f"{band[0]:g}", hi=f"{band[1]:g}")
