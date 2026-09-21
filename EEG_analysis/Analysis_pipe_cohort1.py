import argparse
import os

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import pearsonr, ttest_rel


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze cohort 1 EEG feature changes.")
    parser.add_argument(
        "--fea",
        default="MSE",
        help="Feature name to load from data/cohort2/<subject>/<fea>.npy, e.g. MSE or fEI.",
    )
    parser.add_argument(
        "--hemi",
        choices=["ipsi", "contra"],
        default="contra",
        help="Hemisphere side relative to the affected side: ipsi for 同侧, contra for 对侧.",
    )
    return parser.parse_args()


def main(fea, hemi):
    workpath = "/mnt/pci-0000:00:17.0-ata-6/VNS/project/processed_eeg_ave_ar/"

    fma_u0 = [
        30, 38,
        27, 27,
        20, 23,
        39, 41,
        23, 24,
        39, 43,
        47, 50,
        27, 30,
        21, 22,
    ]
    fma_u0 = np.array(fma_u0).reshape(-1, 2)
    fma_pre = fma_u0[:, 0]
    fma_post = fma_u0[:, 1]
    fma_delta = fma_post - fma_pre

    side = ["R", "R", "L", "R", "L", "L", "R", "R", "L"]

    datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))
    sub = os.listdir(datapath)

    index_all = np.zeros([9, 2, 256])
    fea_label = fea
    for sid in range(len(sub)):
        index = np.load(os.path.join(datapath, sub[sid], fea + ".npy"))
        if fea == "MSE":
            index_all[sid, :, :] = np.mean(index, axis=2)
        else:
            index_all[sid, :, :] = index

    if fea == "fEI":
        index_all = np.abs(1 - index_all)
        fea_label = "dis_cri"

    left_elc = [49, 42, 24, 64, 59, 44, 76, 66, 79]
    right_elc = [207, 206, 213, 185, 183, 194, 143, 164, 172]

    data_L = np.mean(index_all[:, :, left_elc], axis=2)
    data_R = np.mean(index_all[:, :, right_elc], axis=2)

    keep = np.ones(len(fma_delta), dtype=bool)
    keep[[6, 8]] = False
    
    data_ipsi = np.stack([data_L[i] if s == "L" else data_R[i] for i, s in enumerate(side)], axis=0)
    data_contra = np.stack([data_L[i] if s == "R" else data_R[i] for i, s in enumerate(side)], axis=0)
    data_by_hemi = {
        "ipsi": data_ipsi,
        "contra": data_contra,
    }
    hemi_label = {
        "ipsi": "Ipsilateral",
        "contra": "Contralateral",
    }[hemi]
    data = data_by_hemi[hemi]
    data_trim = data[keep]

    stat, p = ttest_rel(data[:, 0], data[:, 1], alternative="greater")
    stat, p = ttest_rel(data_trim[:, 0], data_trim[:, 1], alternative="greater")
    p_trim = p
    print(p)
    print(data[:, 0] - data[:, 1])

    coef, p = pearsonr(fma_u0, data[:, 0])
    print(coef, p)

    coef, p = pearsonr(fma_u0, data[:, 1])
    print(coef, p)

    coef, p = pearsonr(fma_u0, data[:, 1] - data[:, 0])
    print(coef, p)

    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.array([0, 1])
    excl_idx = np.where(~keep)[0]
    for i in excl_idx:
        ax.plot(
            x,
            [data[i, 0], data[i, 1]],
            "-o",
            color="grey",
            lw=1.2,
            alpha=0.5,
            markersize=6,
            markerfacecolor="lightcoral",
            markeredgecolor="k",
            markeredgewidth=0.5,
        )
        ax.text(-0.02, data[i, 0], sub[i][0:2], fontsize=7, ha="right", va="center", color="lightcoral")
    for i in range(data.shape[0]):
        if keep[i]:
            ax.plot(
                x,
                [data[i, 0], data[i, 1]],
                "-o",
                color="grey",
                lw=1.2,
                alpha=0.7,
                markersize=6,
                markerfacecolor="lightblue",
                markeredgecolor="k",
                markeredgewidth=0.5,
            )
            ax.text(-0.02, data[i, 0], sub[i][0:2], fontsize=7, ha="right", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre", "Post"])
    ax.set_ylabel(f"{hemi_label} {fea_label}")
    ax.set_xlim(-0.3, 1.3)
    ax.text(
        0.95,
        0.05,
        f"Paired t-test p={p_trim:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 4))
    fma_trim = fma_delta[keep]
    y = data[:, 1] - data[:, 0]
    y_trim = y[keep]
    ax.scatter(fma_trim, y_trim, c="lightblue", edgecolors="k", s=50, zorder=3)
    coef, p = pearsonr(fma_trim, y_trim)
    slope, intercept = np.polyfit(fma_trim, y_trim, 1)
    x_fit = np.linspace(fma_trim.min(), fma_trim.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.5)
    ax.set_xlabel("Delta FMA-UE (Post-Pre Change)")
    ax.set_ylabel(f"{hemi_label} {fea_label} (Post-Pre Change)")
    ax.text(
        0.95,
        0.05,
        f"r = {coef:.3f}\np = {p:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    eeg_root = "/mnt/pci-0000:00:17.0-ata-6/VNS/VNS-EEG"
    eeg_sub = os.listdir(eeg_root)
    mfffile = os.listdir(os.path.join(eeg_root, eeg_sub[0]))
    raw = mne.io.read_raw_egi(os.path.join(eeg_root, eeg_sub[0], mfffile[0]), preload=True)
    raw.drop_channels(["VREF"])
    raw.info.get_montage()

    topo_data = np.squeeze(np.mean(index_all[keep, 0, :], axis=0))
    fig, ax = plt.subplots()
    im, cn = mne.viz.plot_topomap(topo_data, raw.info, axes=ax, show=False, vlim=[topo_data.min(), topo_data.max()])
    ax.set_title("EEG Topomap " + fea_label, fontsize=14)
    plt.colorbar(im, ax=ax, orientation="vertical", label=fea_label)
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.fea, args.hemi)
