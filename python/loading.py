"""Load the raw OPM and EEG recordings and build merged, cleaned epochs."""

import re

import mne
import numpy as np

import config as cfg
from rejected_epochs import REJECTED


def read_flt(path):
    """Read an interleaved float32 .flt file. Returns (n_channels, n_samples) array."""
    header = open(str(path) + ".hdr", encoding="latin1").read()
    n_chan = int(re.search(r"number_of_channels=(\d+)", header).group(1))
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(-1, n_chan).T


def load_eeg_raw(pid):
    """EEG (ERG electrodes '1', '2') plus the analog trigger channel 'Ana'."""
    d = cfg.participant_dir(pid)
    if pid in cfg.SPLIT_EEG_PARTICIPANTS:
        raw = mne.io.read_raw_fif(d / f"{pid}000.0100_raw.fif", preload=True)
    else:
        data = read_flt(d / f"{pid}000.0100.flt") / 1e6   # uV -> V
        data[3] = np.abs(data[3])                          # trigger can be negative
        info = mne.create_info(["1", "2", "3", cfg.EEG_STIM], cfg.SFREQ, ["eeg"] * 3 + ["stim"])
        raw = mne.io.RawArray(data, info)
        raw.drop_channels([cfg.EEG_REF_CHANNEL])
    raw.anonymize()
    return raw


def load_opm_raw(pid):
    raw = mne.io.read_raw_fif(cfg.participant_dir(pid) / f"{pid}_raw.fif", preload=True)
    raw.anonymize()
    return raw


def _epochs(raw, stim, tmin, tmax, baseline):
    events = mne.find_events(raw, stim_channel=stim, output="onset", consecutive=False,
                             min_duration=cfg.MIN_EVENT_DURATION)
    return mne.Epochs(raw, events, tmin=tmin, tmax=tmax, baseline=baseline,
                      reject_by_annotation=False, preload=True)


def merged_epochs(pid, l_freq=None, h_freq=None, tmin=cfg.EPOCH_TMIN, tmax=cfg.EPOCH_TMAX,
                  baseline=None, reject=True):
    """OPM + EEG epochs for one participant.

    The continuous data are filtered (if l_freq/h_freq given), epoched on each
    system's own trigger, merged, and the first N_SKIP_EPOCHS plus the visually
    rejected epochs are dropped. Only data channels are kept, in the order
    14 OPMs, ERG '1', ERG '2'.
    """
    raws = [load_opm_raw(pid), load_eeg_raw(pid)]
    if l_freq is not None or h_freq is not None:
        for raw in raws:
            raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
    opm = _epochs(raws[0], cfg.OPM_STIM, tmin, tmax, baseline)
    eeg = _epochs(raws[1], cfg.EEG_STIM, tmin, tmax, baseline)
    # the systems are aligned only by trigger order, so both must see every flash
    if len(opm) != len(eeg):
        raise RuntimeError(f"{pid}: {len(opm)} OPM triggers but {len(eeg)} EEG triggers")
    epochs = opm.add_channels([eeg])
    epochs.drop(list(range(cfg.N_SKIP_EPOCHS)))
    if reject:
        epochs.drop(REJECTED[pid])
    return epochs.pick(["mag", "eeg"])


def read_evoked(path):
    return mne.read_evokeds(path, condition=0, baseline=None, verbose="error")


def read_tfr(path):
    tfr = mne.time_frequency.read_tfrs(path)
    return tfr[0] if isinstance(tfr, list) else tfr
