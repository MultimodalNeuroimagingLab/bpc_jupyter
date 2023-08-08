"""
.. _ex-rereference:

=====================
Rereference iEEG Data
=====================

Reference per headbox to low variance channels.
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#          Dora Hermes
#
# License: BSD-3-Clause
import os
import numpy as np
from pathlib import Path
import pandas as pd

import mne
import mne_bids
from pymef.mef_session import MefSession

dataset = 'ds003708'
root = Path('../..') / dataset
path = mne_bids.BIDSPath(
    subject='01', session='ieeg01', task='ccep', run='01',
    datatype='ieeg', root=root)
metadata = pd.read_csv(path.copy().update(suffix='events'), sep='\t')
metadata = metadata[metadata.status == 'good']
ms = MefSession(str(path.fpath) + '_ieeg.mefd', '')
ms_info = ms.read_ts_channel_basic_info()
ch_names = [ch['name'] for ch in ms_info]
info = mne.create_info(ch_names, ms_info[0]['fsamp'][0], 'seeg')
data = np.array(ms.read_ts_channels_sample(ch_names, [0, ms_info[0]['nsamp'][0]]))
raw = mne.io.RawArray(0.2658 * data / 1e6, info)
del data
ch_names_ordered = pd.read_csv(
    path.copy().update(suffix='channels'),
    sep='\t')['name']
raw = raw.reorder_channels(list(ch_names_ordered))
mne.export.export_raw(str(path.fpath) + '_ieeg.vhdr', raw)

'''
exclude = np.array([contact in sites.split('-') for sites in
                    metadata.electrical_stimulation_site])
sfreq = ms_info[0]['fsamp'][0]
epo_data = np.zeros((events.shape[0], len(ch_names), int((tmax - tmin) * sfreq)))
for i, e in enumerate(events[:, 0]):
    epo_data[i] = data[:, e + int(tmin * sfreq):e + int(tmax * sfreq)]

epo_data = epo_data[~exclude]
'''

# re-reference, exclude stimulation artifact for reference
raw = mne_bids.read_raw_bids(path)
raw.load_data()
events = metadata.onset * raw.info['sfreq']
mask = np.zeros((len(raw.ch_names), raw.times.size), dtype=bool)
for onset, site in zip(metadata.onset, metadata.electrical_stimulation_site):
    samp = int(onset * raw.info['sfreq'])
    start_i, end_i = int(samp - raw.info['sfreq']), int(samp + 3 * raw.info['sfreq'])
    for ch in site.split('-'):
        mask[raw.ch_names.index(ch), start_i:end_i] = True
stim_artifact = raw._data[mask].copy()  # save for later
raw._data[mask] = np.nan

# find channels with low variance
events = np.zeros((len(metadata), 3), dtype=int)
events[:, 0] = metadata.onset * raw.info['sfreq']
epochs = mne.Epochs(raw, events, tmin=0.01, tmax=2,
                    baseline=None, preload=True)
epochs.pick(['ecog', 'seeg'], exclude='bads')
# swap epochs with channels to flatten
erp_data = epochs.get_data(tmax=0.1).transpose([1, 0, 2])
baseline_data = epochs.get_data(tmin=0.5).transpose([1, 0, 2])
erp_var = np.nanvar(erp_data.reshape(erp_data.shape[0], -1), axis=1, ddof=1)
baseline_var = np.nanvar(baseline_data.reshape(
    baseline_data.shape[0], -1), axis=1, ddof=1)
erp_var_thresh = np.quantile(erp_var, 0.75, method='midpoint')
baseline_var_thresh = np.quantile(baseline_var, 0.75, method='midpoint')
low_var = (erp_var < erp_var_thresh) & (baseline_var < baseline_var_thresh)
ref_chs = [epochs.ch_names[i] for i, check in enumerate(low_var) if check]
del epochs
sel = mne.pick_types(raw.info, ecog=True, seeg=True)
for i in range(0, len(raw.ch_names), 64):
    max_i = min([i + 64, len(raw.ch_names) - 1])
    idxs = np.arange(i, max_i)[np.in1d(np.arange(i, max_i), sel)]
    ref_idxs = [i for i in idxs if raw.ch_names[i] in ref_chs]
    print(f"re-referencing block {i} to {[raw.ch_names[i] for i in ref_idxs]}")
    raw._data[idxs] -= np.nanmean(raw._data[ref_idxs], axis=0, keepdims=True)

raw._data[mask] = stim_artifact
del erp_data, baseline_data, mask, stim_artifact
for ext in ('vhdr', 'vmrk', 'eeg'):  # memory issues when not deleted
    os.remove(str(path.fpath) + f'_ieeg.{ext}')
mne.export.export_raw(str(path.fpath) + '_ieeg.vhdr', raw)
