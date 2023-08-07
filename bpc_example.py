"""
.. _ex-bpc:

==================================
Basis Profile Curve (BPC) analysis
==================================

In this example, we will show how to use basis profile curve (BPC)
analysis to quantify the shapes and strengths of connectivity from
electrical stimulation sites to a recording site.

To do: format citation  https://doi.org/10.1101/2021.01.24.428020
"""
# Authors: Alex Rockhill <aprockhill@mailbox.org>
#          Dora Hermes
#
# License: BSD-3-Clause

from pathlib import Path
import numpy as np

import openneuro
import mne
import mne_bids

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import NMF

"""
import os
# conversion
dataset = 'ds003708'
root = Path('..') / dataset
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
"""

# %%
# First, download the dataset.
dataset = 'ds003708'
root = Path('..') / dataset
openneuro.download(dataset=dataset, target_dir=root)

# %%
# Load the data and plot channel positions and events.
path = mne_bids.BIDSPath(
    subject='01', session='ieeg01', task='ccep', run='01', root=root)
raw = mne_bids.read_raw_bids(path)

trans = mne.transforms.Transform(fro='head', to='mri', trans=np.eye(4))  # identity
fig = mne.viz.plot_alignment(
    raw.info, trans=trans, subject='fsaverage', surfaces='pial')
mne.viz.set_3d_view(fig, azimuth=150)

xy, im = mne.viz.snapshot_brain_montage(fig, raw.info)
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(im)
for name, pos in xy.items():
    if pos[0] >= 0 and pos[1] >= 0:  # no NaN locations
        ax.text(*pos, name, ha='center', va='center', fontsize=4)

events, event_id = mne.events_from_annotations(raw)
mne.viz.plot_events(events, raw.info['sfreq'], event_id=event_id)

# %%
# Create epochs around stimulation, visualize data.
contact = 'LMS2'
tmin, tmax = -1, 2
bl_tmin, bl_tmax = -0.5, -0.05

# try ``baseline=None`` for no baseline correction to play around
metadata = pd.read_csv(path.update(suffix='events'), sep='\t')
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=(bl_tmin, bl_tmax), picks=[contact],
                    preload=True)
epochs.metadata = metadata  # contains stimulation location information
epochs = epochs[metadata.status == 'good']

# unpack each pair separated by a hyphen, only use trials where
# stimulation was delivered to channels other than the channel of
# interest
epochs.metadata['site1'], epochs.metadata['site2'] = np.array([
    sites.split('-') for sites in
    epochs.metadata.electrical_stimulation_site]).T
exclude = np.in1d(epochs.metadata.site1, contact) | \
    np.in1d(epochs.metadata.site2, contact)
epochs = epochs[~exclude]

epochs.plot_image(picks=[contact], cmap='viridis', vmin=-250, vmax=250)

# %%
# Calculate BPCs.
bpc_tmin, bpc_tmax = 0.015, 1

V = epochs.get_data(tmin=bpc_tmin, tmax=bpc_tmax)[:, 0]  # select only channel
V0 = V / np.linalg.norm(V, axis=0)  # L2 norm each trial
P = V0 @ V.T  # calculate internal projections

pairs = np.array(sorted(np.unique(epochs.metadata.electrical_stimulation_site)))
tmat = np.zeros((len(pairs), len(pairs)))
for i, pair1 in enumerate(pairs):
    mask = epochs.metadata.electrical_stimulation_site == pair1
    b = P[np.ix_(mask, mask)].ravel()
    tmat[i, i] = np.mean(b) * np.sqrt(len(b)) / np.std(b, ddof=1)
    for j, pair2 in enumerate(pairs[i + 1:]):
        b = P[np.ix_(epochs.metadata.electrical_stimulation_site == pair1,
                     epochs.metadata.electrical_stimulation_site == pair2)].ravel()
        tmat[i + j + 1, i] = np.mean(b) * np.sqrt(len(b)) / np.std(b, ddof=1)

# copy to lower
tmat = tmat + tmat.T - np.diag(np.diag(tmat))

fig, ax = plt.subplots()
im = ax.imshow(tmat, vmin=0, vmax=10)
ax.set_xticks(range(tmat.shape[0]))
ax.set_xticklabels(pairs, rotation=90, fontsize=6)
ax.set_xlabel('Stimulation Pair')
ax.set_yticks(range(tmat.shape[0]))
ax.set_yticklabels(pairs, fontsize=6)
ax.set_ylabel('Stimulation Pair')
ax.set_title(r'Significance Matrix $\Xi$', fontsize=15)
fig.colorbar(im, ax=ax)
fig.subplots_adjust(bottom=0.2)
fig.show()




H0 = 0 * H_min

for k in range(len(pair_types)):
    k_ind = np.argmax(H_min, axis=0)[k]
    H0[k_ind][k] = H_min[k_ind][k]

H0_ = H0 > 1 / (2 * np.sqrt(len(pair_types)))




"""
Calculate statistics for each basis curve, eliminating those where there is no significant representation
"""

B_struct = pyBPCs.curvesStatistics(B_struct, V, B, pair_types)


"""
Calculate projection weights
"""

B_struct = pyBPCs.projectionWeights(B_struct)

B_struct.pairs


colors = iter(cm.tab10(np.linspace(0, 1, 10)))

plt.figure(figsize=(10, 8), dpi=80)
for q in range(len(B_struct)): #  cycle through basis curves
    plt.plot(tt[tt_BPCs[0]:tt_BPCs[1]], B_struct.curve[q], color=next(colors), label=q)

plt.xlabel('Time from stimulation (s)')
plt.ylabel('Normalized weight of BPCs')
plt.title('Calculated BPCs',fontsize=15)
plt.legend()

plt.show()

marker_colors = [] # marker_color array
marker_sizes = [] # marker_size array

# all electrodes
xyz = df_electrodes[['x', 'y','z']]
xyz_list = xyz.values.tolist()
electrodeName_list = df_electrodes[['name']].name.values.tolist()
for cords in xyz_list:
    marker_colors.append('white')
    marker_sizes.append(7.5)

# el_interest
xyz_el_interest = df_electrodes.loc[df_electrodes.name == el_interest_nr.name.values[0]][['x', 'y','z']]
xyz_list.append(xyz_el_interest.values[0].tolist())
marker_colors.append('black')
marker_sizes.append(20)

pair_types['interpolated_locs'] = ''

for index, row in pair_types.iterrows():
    xyz_1 = xyz.iloc[row['ccep_num_1']].values.tolist()
    xyz_2 = xyz.iloc[row['ccep_num_2']].values.tolist()
    pair_types.at[index, 'interpolated_locs'] = np.mean((xyz_1, xyz_2), axis=0).tolist()

# non-significant stim pair sites    
interpolated_locs_list = pair_types.iloc[excluded_pairs[0]].interpolated_locs.values.tolist()
for cords in interpolated_locs_list:
    marker_colors.append('gray')
    marker_sizes.append(7.5)

xyz_all = [y for x in [xyz_list, interpolated_locs_list] for y in x]

# plot BPCs, colored
colors = iter(cm.tab10(np.linspace(0, 1, 10)))

for q in range(B_struct.shape[0]):
    # get electrodes for BPC
    xyz_BPC_list = pair_types.interpolated_locs[B_struct.pairs[q]].tolist()
    xyz_all = [y for x in [xyz_all, xyz_BPC_list] for y in x]
    plotweights = B_struct.plotweights[q]
    curr_color = next(colors)
    for ind_cords in range(len(xyz_BPC_list)):
        marker_colors.append(curr_color)
        marker_sizes.append(B_struct.plotweights[q][ind_cords] * 25)

# plot all
view = plotting.view_markers(marker_coords=xyz_all,marker_size=marker_sizes, # marker_labels=electrodeName_list,
                             marker_color=marker_colors, title='Spatial representation of BPCs rendered on an MNI brain') # Insert a 3d plot of markers in a brain
view.save_as_html('markers.html')


def curve_stats(B_struct, V, B, pair_types):
    B_struct['alphas'] = None
    B_struct['ep2'] = None
    B_struct['V2'] = None
    B_struct['errxproj'] = None

    for bb in range(B_struct.shape[0]):  # cycle through basis curves

        # alpha coefficient weights for basis curve bb into V
        al = B_struct.curve[bb] @ V
        # np.newaxis comes in handy when we want to explicitly convert a 1D array to either a row vector or a column vector!!!!
        # residual epsilon (error timeseries) for basis bb after alpha*B coefficient fit
        ep = V - B_struct.curve[bb][np.newaxis].T @ al[np.newaxis]
        errxproj = ep.T  @ ep  # calculate all projections of error
        V_selfproj = V.T @ V  # power in each trial

        B_struct['alphas'][bb] = []
        B_struct['ep2'][bb] = []
        B_struct['V2'][bb] = []
        B_struct['errxproj'][bb] = []

        pair_types = pair_types.reset_index(drop=True)

        # cycle through pair types represented by this basis curve
        for n in range(len(B_struct.pairs[bb])):
            ind = (B_struct.pairs[bb])[n]
            tmp_inds = pair_types['indices'][ind]  # indices for this pair type
            # alpha coefficient weights for basis curve bb into V
            (B_struct.alphas[bb]).append(al[tmp_inds])
            # self-submatrix of error projections
            a = errxproj[np.ix_(tmp_inds, tmp_inds)]
            (B_struct.ep2[bb]).append((np.diag(a)).T)  # sum-squared error
            # sum-squared individual trials
            (B_struct.V2[bb]).append(
                np.diag(V_selfproj[np.ix_(tmp_inds, tmp_inds)]).T)

            # gather all off-diagonal elements from self-submatrix
            b = []
            # for q=1:(size(a,2)-1), b=[b a(q,(q+1):end)]; end
            for q in range(a.shape[1]-1):
                b.extend(a[q, q+1:])
            # for q=2:(size(a,2)), b=[b a(q,1:(q-1))]; end
            for q in range(1, a.shape[1]):
                b.extend(a[q, :q-1])

            # systematic residual structure within a stim pair group for a given basis will be given by set of native normalized internal cross-projections
            B_struct.errxproj[bb] = b
    return B_struct


def projection_weights(B_struct):
    B_struct['p'] = None
    B_struct['plotweights'] = None
    for q in range(B_struct.shape[0]):  # cycle through basis curves
        B_struct['p'][q] = []
        B_struct['plotweights'][q] = []
        # cycle through pair types represented by this basis curve
        for n in range(B_struct.pairs[q].shape[0]):
            curr_alphas = B_struct.alphas[q][n]
            curr_ep2_5 = (B_struct.ep2[q][n])**0.5
            # alphas normalized by error magnitude
            B_struct.plotweights[q].append(np.mean(curr_alphas / curr_ep2_5))

            # significance alphas normalized by error magnitude
            t, pVal = stats.ttest_1samp((curr_alphas / curr_ep2_5), 0)
            (B_struct.p[q]).append(pVal)

    return B_struct


def kpca(X):
    F, S, _ = np.linalg.svd(X.T)  # Compute the eigenvalues and right eigenvectors
    ES = X @ F  # kernel trick
    # divide through to obtain unit-normalized eigenvectors
    E = ES / (np.ones((X.shape[0], 1)) @ S[np.newaxis])
    return E


def bpcs(V, stim_sites, cluster_dim=10, n_reruns=20, tol=1e-5,
         random_state=99, verbose=True):
    """Compute basis profile curves of an evoked response.

    Parameters
    ----------
    V : np.ndarray (n_epochs, n_samples)
        The voltage time course for the channel of interest.
    stim_sites : np.ndarray (n_epochs)
        The stimulation sites for each epoch.
    cluster_dim : int
        The maximum dimension of the clusters.
    n_reruns : int
        The number of reruns of non-negative matrix factorization
        to ensure convergence.
    tol : float
        The convergence tolerance threshold.
    random_state : int
        Reproducibility seed.
    verbose : bool
        Whether to print function status updates.

    Returns
    -------
    tmat : np.ndarray (n_pairs, n_pairs)
        The projection matrix.
    """
    V0 = V / np.linalg.norm(V, axis=0)  # L2 norm each trial
    P = V0 @ V.T  # calculate internal projections

    pairs = np.array(sorted(np.unique(stim_sites)))
    tmat = np.zeros((len(pairs), len(pairs)))
    for i, pair1 in enumerate(pairs):
        mask = stim_sites == pair1
        b = P[np.ix_(mask, mask)].ravel()
        tmat[i, i] = np.mean(b) * np.sqrt(len(b)) / np.std(b, ddof=1)
        for j, pair2 in enumerate(pairs[i + 1:]):
            b = P[np.ix_(stim_sites == pair1, stim_sites == pair2)].ravel()
            tmat[i + j + 1, i] = np.mean(b) * np.sqrt(len(b)) / np.std(b, ddof=1)

    # copy to lower
    tmat = tmat + tmat.T - np.diag(np.diag(tmat))

    t0 = tmat.copy()
    t0[t0 < 0] = 0
    t0 /= (np.max(t0))

    for n_components in range(cluster_dim, 1, -1):
        this_error = None
        for k in range(n_reruns):
            model = NMF(n_components=n_components, init='random', solver='mu',
                        tol=tol, max_iter=10000, random_state=random_state).fit(t0)
            if this_error is None or model.reconstruction_err_ < this_error:
                this_error = model.reconstruction_err_
                W = model.transform(t0)
                H = model.components_
        H /= np.linalg.norm(H, axis=1)[:, None]
        nmf_penalty = np.triu(H @ H.T, 1).sum()
        if verbose:
            print(f'Inner dimension: {n_components}, off diagonal score: {nmf_penalty}')
        if nmf_penalty <= 1:
            break

    # find significant pairs per BPC; must be > threshold and greater than other BPCs
    for bpc_idx in range(H.shape[0]):
        bpc_pair_idxs = np.where((H[bpc_idx] == np.max(H, axis=0)) &
                                 (H[bpc_idx] > 1 / (2 * np.sqrt(len(pairs)))))[0]
        bpc_trials = np.concatenate([np.where(stim_sites == pairs[bpc_pair_idx])[0]
                                     for bpc_pair_idx in bpc_pair_idxs])
        B = kpca(V[bpc_trials])
    return tmat


B = []
B_struct = pd.DataFrame(columns=['curve', 'pairs'])  # saving structure

for q in range(H0.shape[0]):
    cl_pg = np.where(H0_[q])  # cluster pair groups
    cl_inds = []  # cluster indices (e.g. concatenated trials from cluster pair groups)
    for k in cl_pg:
        for i in (pair_types['indices'][k]).values:
            cl_inds.extend(i)

    V_tmp = V[:, cl_inds]

    E = pyBPCs.kpca(V_tmp)
    B_tmp = E[:, 0]  # basis vector is 1st PC

    if np.mean(B_tmp.T @ V_tmp) < 0:
        B_tmp = -B_tmp

    curr_B = pd.DataFrame({'curve': [B_tmp], 'pairs': [cl_pg[0]]})
    B_struct = pd.concat([B_struct, curr_B], ignore_index=True)

# pairs not represented by any basis
excluded_pairs = np.where(1 - (np.sum(H0_, axis=0)))