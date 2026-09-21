import mne
import os
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import zscore, ttest_rel, ranksums, pearsonr, spearmanr
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# First cohort analysis

fma_u0 = [30,38,
          27,27,
          20,23,
          39,41,
          23,24,
          39,43,
          47,50,
          27,30,
          21,22] #baseline and after treatment
fma_u0 = np.array(fma_u0).reshape(-1, 2)
fma_pre = fma_u0[:, 0]
fma_post = fma_u0[:, 1]
fma_delta = fma_post - fma_pre

side = ['R','R','L','R','L','L','R','R','L']

fea = 'fEI'

vns_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datapath = os.path.join(vns_root, 'RepairStudy', 'results_epoch85_baseline')

session_dirs = [
    name for name in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, name)) and name.startswith('sub-')
]
subject_ids = sorted({name.rsplit('_', 1)[0] for name in session_dirs})
sub = [sid.replace('sub-', '') for sid in subject_ids]

first_file = os.path.join(datapath, subject_ids[0] + '_pre', fea + '.npy')
first_index = np.load(first_file)
if fea == 'MSE':
    n_channels = first_index.shape[0]
else:
    n_channels = first_index.shape[-1]

index_all = np.full([len(subject_ids), 2, n_channels], np.nan)

for sid, subject_id in enumerate(subject_ids):
    for cond_idx, session in enumerate(['pre', 'post']):
        index_path = os.path.join(datapath, f'{subject_id}_{session}', fea + '.npy')
        index = np.load(index_path)
        if fea == 'MSE':
            index_all[sid, cond_idx, :] = np.nanmean(index, axis=1)
        else:
            index_all[sid, cond_idx, :] = index

if fea == 'fEI':
    index_all = np.abs(1-index_all)
    fea = 'dis_cri'

#elc id
left_elc = [8,31,32,33,34]
right_elc = [17,40,41,42,43]

data_L = np.nanmean(index_all[:,:,left_elc],axis=2)
data_R = np.nanmean(index_all[:,:,right_elc],axis=2)
data_all = np.nanmean(index_all[:,:,left_elc+right_elc],axis=2)

side = ['R','R','L','R','L','L','R','R','L']

data_contra = np.stack([data_L[i] if s == 'R' else data_R[i] for i, s in enumerate(side)], axis=0)
data_ipsi = np.stack([data_L[i] if s == 'L' else data_R[i] for i, s in enumerate(side)], axis=0)
data = data_contra
# data = data_contra+data_ipsi
keep = np.ones(data.shape[0], dtype=bool)
keep[[6, 8]] = False  # remove 7th and 9th subject
if keep.shape[0] != data.shape[0]:
    raise ValueError(f'keep length {keep.shape[0]} does not match data rows {data.shape[0]}')
if fma_delta.shape[0] != data.shape[0]:
    raise ValueError(f'fma rows {fma_delta.shape[0]} does not match data rows {data.shape[0]}')
data_trim = data[keep]

stat, p = ttest_rel(data[:,0], data[:,1], alternative='less')
stat, p = ttest_rel(data_trim[:,0], data_trim[:,1], alternative='less')
p_trim = p  # save for ladder plot annotation
# coef[ele] = ranksums(np.squeeze(index_all[:, 0, ele]), np.squeeze(index_all[:, 1, ele]), alternative='greater')[0]
print(p)
print(data[:,0]-data[:,1])

coef, p = pearsonr(fma_pre, data[:,0])
print(coef, p)

coef, p = pearsonr(fma_post, data[:,1])
print(coef, p)

coef, p = pearsonr(fma_delta, data[:,1]-data[:,0])
print(coef, p)

# Ladder plot: data[:,0] vs data[:,1] paired comparison
fig, ax = plt.subplots(figsize=(4, 4))
x = np.array([0, 1])
# Plot excluded subjects (keep==False) in a distinct color
excl_idx = np.where(~keep)[0]
for i in excl_idx:
    ax.plot(x, [data[i, 0], data[i, 1]], '-o', color='grey', lw=1.2, alpha=0.5,
            markersize=6, markerfacecolor='lightcoral', markeredgecolor='k', markeredgewidth=0.5)
    ax.text(-0.02, data[i, 0], sub[i][0:2], fontsize=7, ha='right', va='center', color='lightcoral')
# Plot kept subjects (keep==True) in normal color
for i in range(data.shape[0]):
    if keep[i]:
        ax.plot(x, [data[i, 0], data[i, 1]], '-o', color='grey', lw=1.2, alpha=0.7,
                markersize=6, markerfacecolor='lightblue', markeredgecolor='k', markeredgewidth=0.5)
        ax.text(-0.02, data[i, 0], sub[i][0:2], fontsize=7, ha='right', va='center')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre', 'Post'])
ax.set_ylabel(fea)
ax.set_xlim(-0.3, 1.3)
ax.text(0.95, 0.05, f'Paired t-test p={p_trim:.4g}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(workpath, 'ladder_plot_'+'Dis_cri'+'.png'), dpi=150)
plt.show()

# topomap visualization
from mne.channels.layout import _find_topomap_coords

megpath = os.path.join(vns_root, 'RepairStudy', 'MEG_REST_MNE')
meg_info_file = [
    os.path.join(megpath, f'{subject_ids[0]}_pre', name)
    for name in os.listdir(os.path.join(megpath, f'{subject_ids[0]}_pre'))
    if name.endswith('_raw.fif')
][0]
raw = mne.io.read_raw_fif(meg_info_file, preload=False)
meg_picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, exclude=[])
meg_info = mne.pick_info(raw.info, meg_picks, copy=True)
result_ch_names = np.load(os.path.join(datapath, f'{subject_ids[0]}_pre', 'ch_names.npy'), allow_pickle=True).tolist()
meg_order = [meg_info['ch_names'].index(ch_name) for ch_name in result_ch_names]
meg_info = mne.pick_info(meg_info, meg_order, copy=True)
meg_pos = _find_topomap_coords(meg_info, np.arange(len(meg_info['ch_names'])))
meg_pos = np.column_stack([-meg_pos[:, 1], meg_pos[:, 0]])

topomap_pre = np.nanmean(index_all[:, 0, :], axis=0)
topomap_post = np.nanmean(index_all[:, 1, :], axis=0)
if topomap_pre.shape[0] != len(meg_info['ch_names']):
    raise ValueError(
        f'topomap data has {topomap_pre.shape[0]} channels, '
        f'but MEG info has {len(meg_info["ch_names"])} mag channels'
    )
vmin = np.nanmin([np.nanmin(topomap_pre), np.nanmin(topomap_post)])
vmax = np.nanmax([np.nanmax(topomap_pre), np.nanmax(topomap_post)])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
im, cn = mne.viz.plot_topomap(topomap_pre, meg_pos, axes=axes[0], show=False, vlim=[vmin, vmax])
axes[0].set_title(f'{fea} Pre mean', fontsize=14)
im, cn = mne.viz.plot_topomap(topomap_post, meg_pos, axes=axes[1], show=False, vlim=[vmin, vmax])
axes[1].set_title(f'{fea} Post mean', fontsize=14)
plt.colorbar(im, ax=axes, orientation='vertical', label=fea)
plt.tight_layout()
plt.show()

subject_vmin = np.nanmin(index_all)
subject_vmax = np.nanmax(index_all)
fig, axes = plt.subplots(2, len(subject_ids), figsize=(2.2 * len(subject_ids), 5))
for cond_idx, session in enumerate(['Pre', 'Post']):
    for sid, subject_id in enumerate(subject_ids):
        ax = axes[cond_idx, sid]
        subject_topomap = index_all[sid, cond_idx, :]
        valid_channels = ~np.isnan(subject_topomap)
        im, cn = mne.viz.plot_topomap(
            subject_topomap[valid_channels],
            meg_pos[valid_channels],
            axes=ax,
            show=False,
            vlim=[subject_vmin, subject_vmax]
        )
        ax.set_title(f'{session} {subject_id}', fontsize=10)
plt.colorbar(im, ax=axes, orientation='vertical', label=fea)
plt.tight_layout()
plt.show()


# Scatter plot: data[:,0] vs fma_u0 (excl subj 7, 9)
fig, ax = plt.subplots(figsize=(4, 4))
fma_trim = fma_delta[keep]
y = data[:, 0]
y_trim =y[keep]
ax.scatter(fma_delta, y, c='lightblue', edgecolors='k', s=50, zorder=3)
# linear fit
coef, p = pearsonr(fma_trim, y_trim)
slope, intercept = np.polyfit(fma_trim, y_trim, 1)
x_fit = np.linspace(fma_trim.min(), fma_trim.max(), 100)
ax.plot(x_fit, slope * x_fit + intercept, 'r--', lw=1.5)
ax.set_xlabel('Delta FMA-UE (Post-Pre Change)')
ax.set_ylabel('Contralateral Dis_cri (Post-Pre Change)')
ax.text(0.95, 0.05, f'r = {coef:.3f}\np = {p:.3g}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
# plt.savefig(os.path.join(workpath, 'scatter_'+fea+'_vs_fma.png'), dpi=150)
plt.show()
