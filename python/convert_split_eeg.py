"""Convert the per-channel EEG files of PO10-PO12 into one .fif file.

These sessions stored each EEG channel in its own float32 file
(<pid>.C01_0000_CH<n>.$$$). Later sessions use one interleaved .flt file,
which loading.load_eeg_raw reads directly.

Usage: python convert_split_eeg.py [PO10 PO11 PO12]
"""

import sys

import mne
import numpy as np

import config as cfg

# hardware channel number -> (name, scale to SI units)
CHANNELS = {368: ("1", 1e-10), 369: ("2", 1e-10), 370: ("3", 1e-10), 423: (cfg.EEG_STIM, 1e-2)}


def convert(pid):
    d = cfg.participant_dir(pid)
    data = np.array([np.fromfile(d / f"{pid}.C01_0000_CH{ch}.$$$", dtype=np.float32) * scale
                     for ch, (_, scale) in CHANNELS.items()])
    data[-1] = np.abs(data[-1])
    names = [name for name, _ in CHANNELS.values()]
    info = mne.create_info(names, cfg.SFREQ, ["eeg"] * 3 + ["stim"])
    raw = mne.io.RawArray(data, info)
    raw.drop_channels([cfg.EEG_REF_CHANNEL])
    raw.save(d / f"{pid}000.0100_raw.fif", overwrite=True)


if __name__ == "__main__":
    for pid in sys.argv[1:] or cfg.SPLIT_EEG_PARTICIPANTS:
        convert(pid)
