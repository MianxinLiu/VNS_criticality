import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, ttest_rel, wilcoxon, zscore


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

RUN_MIXED_MODELS = True
RUN_FMA_SCATTER = True
RUN_SUBJECT_TRAJECTORY = True
RUN_TOPO_BY_WEEK = True
RUN_WEEK_TREND = True
RUN_PAIRED_TEST = True
RUN_DELTA_CORR = True


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze cohort 2 EEG feature changes.")
    parser.add_argument(
        "--fea",
        default="fEI",
        help="Feature name to load from data/cohort1/<subject>/<fea>.npy, e.g. fEI or MSE.",
    )
    parser.add_argument(
        "--hemi",
        choices=["ipsi", "contra"],
        default="ipsi",
        help="Hemisphere side relative to the affected side: ipsi for 同侧, contra for 对侧.",
    )
    parser.add_argument(
        "--only-models",
        action="store_true",
        help="Only run mixed linear model comparisons; skip plots and other analyses.",
    )
    return parser.parse_args()


def fit_mixed_model(name, formula, df, re_formula=None):
    try:
        model = smf.mixedlm(formula, data=df, groups=df["subid"], re_formula=re_formula)
        result = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    except Exception as exc:
        print(f"{name:<20} FAILED: {exc}")
        return None

    metric_terms = [term for term in result.params.index if "metric" in term]
    term_summary = ", ".join(
        f"{term}: beta={result.params[term]:.4f}, p={result.pvalues[term]:.4g}"
        for term in metric_terms
    )
    converged = getattr(result, "converged", "NA")
    print(
        f"{name:<20} AIC={result.aic:>9.3f}  BIC={result.bic:>9.3f}  "
        f"llf={result.llf:>9.3f}  converged={converged}  {term_summary}"
    )
    return result


def run_mixed_model_comparison(df):
    print("\n" + "=" * 80)
    print("Mixed linear model comparison")
    print("=" * 80)
    print("Lower AIC/BIC is better for model fit; metric-term p values test EEG-FMA association.")

    models = [
        ("base", "resp ~ metric + time", None),
        ("time_cat", "resp ~ metric + C(time_cat)", None),
        ("interaction", "resp ~ metric * time", None),
        ("interaction_time_z", "resp ~ metric * time_z", None),
        ("within_between", "resp ~ metric_within + metric_mean + time", None),
        ("baseline_resp", "resp ~ metric + time + baseline_resp", None),
        ("baseline_adjusted", "resp ~ metric + time + baseline_resp + baseline_metric", None),
        ("random_time", "resp ~ metric + time", "~time"),
    ]

    results = []
    for name, formula, re_formula in models:
        result = fit_mixed_model(name, formula, df, re_formula=re_formula)
        if result is not None:
            results.append((name, result))

    if not results:
        return

    best_aic = min(results, key=lambda item: item[1].aic)
    best_bic = min(results, key=lambda item: item[1].bic)
    print("\nBest by AIC:", best_aic[0], f"AIC={best_aic[1].aic:.3f}")
    print("Best by BIC:", best_bic[0], f"BIC={best_bic[1].bic:.3f}")

    base_result = dict(results).get("base")
    if base_result is not None:
        print("\nBase model summary:")
        print(base_result.summary())


def main(fea, hemi, only_models=False):
    workpath = "/mnt/pci-0000:00:17.0-ata-6/VNS/project/processed_eeg_ave_ar/"

    fma_u0 = [
        40, 48, 52,
        51, 51, 53, 61, 63,
        10, 15, 16, 24, 22, 26,
        49, 48, 54,
        33, 44, 55,
        35, 43, 46,
        13, 23, 31, 28, 24,
        39, 48, 48, 48, 47,
        47, 59, 60, 62,
        44, 47, 54, 56,
    ]

    subid = [
        10, 10, 10,
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2,
        3, 3, 3,
        4, 4, 4,
        5, 5, 5,
        6, 6, 6, 6, 6,
        7, 7, 7, 7, 7,
        8, 8, 8, 8,
        9, 9, 9, 9,
    ]

    datapath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort1"))
    sub = os.listdir(datapath)
    fea_label = fea

    if fea == "MSE":
        index_all = np.zeros([3, 256, 20])
    elif fea == "s_scale" or fea == "t_scale":
        index_all = np.zeros([3])
    else:
        index_all = np.zeros([3, 256])
    time = np.zeros([3])

    index = np.load(os.path.join(datapath, sub[0], fea + ".npy"))
    wk = np.load(os.path.join(datapath, sub[0], "Weeks.npy"))
    if fea == "MSE":
        index_all = np.mean(index[0:3, :, :], axis=2)
    elif fea == "s_scale" or fea == "t_scale":
        index_all = index[0:3]
    else:
        index_all = index[0:3, :]
    time = wk[0:3]

    for sid in range(1, len(sub)):
        index = np.load(os.path.join(datapath, sub[sid], fea + ".npy"))
        wk = np.load(os.path.join(datapath, sub[sid], "Weeks.npy"))
        if sid == 4:
            if fea == "MSE":
                index_all = np.append(index_all, np.mean(index[0:3, :, :], axis=2), axis=0)
            elif fea == "s_scale" or fea == "t_scale":
                index_all = np.append(index_all, index[0:3], axis=0)
            else:
                index_all = np.append(index_all, index[0:3, :], axis=0)
            time = np.append(time, wk[0:3], axis=0)
        elif sid == 9:
            if fea == "MSE":
                index_all = np.append(index_all, np.mean(index[0:4, :, :], axis=2), axis=0)
            elif fea == "s_scale" or fea == "t_scale":
                index_all = np.append(index_all, index[0:4], axis=0)
            else:
                index_all = np.append(index_all, index[0:4, :], axis=0)
            time = np.append(time, wk[0:4], axis=0)
        else:
            if fea == "MSE":
                index_all = np.append(index_all, np.mean(index[:, :, :], axis=2), axis=0)
            elif fea == "s_scale" or fea == "t_scale":
                index_all = np.append(index_all, index[:], axis=0)
            else:
                index_all = np.append(index_all, index[:, :], axis=0)
            time = np.append(time, wk, axis=0)

    if fea == "fEI":
        index_all = np.abs(1 - index_all)
        fea_label = "dis_cri"

    left_elc = [49, 42, 24, 64, 59, 44, 76, 66, 79]
    right_elc = [207, 206, 213, 185, 183, 194, 143, 164, 172]
    side = ["L", "L", "L", "L", "R", "L", "L", "L", "R", "R"]

    allside = []
    for i in range(len(subid)):
        allside.append(side[subid[i] - 1])
    allside = np.array(allside)

    data_L = np.mean(index_all[:, left_elc], axis=1)
    data_R = np.mean(index_all[:, right_elc], axis=1)
    data_ipsi = np.stack([data_L[i] if s == "L" else data_R[i] for i, s in enumerate(allside)], axis=0)
    data_contra = np.stack([data_L[i] if s == "R" else data_R[i] for i, s in enumerate(allside)], axis=0)
    data_by_hemi = {
        "ipsi": data_ipsi,
        "contra": data_contra,
    }
    hemi_label = {
        "ipsi": "Ipsilateral",
        "contra": "Contralateral",
    }[hemi]
    data = np.asarray(data_by_hemi[hemi])

    subid = np.asarray(subid)
    fma_u0 = np.asarray(fma_u0)

    raw_df = pd.DataFrame({"subid": subid, "time": time, "metric_raw": data, "resp_raw": fma_u0})
    raw_df["time_round"] = np.round(raw_df["time"]).astype(int)
    raw_df["time_cat"] = raw_df["time_round"].astype("category")
    raw_df["time_z"] = zscore(raw_df["time"])
    raw_df["metric"] = zscore(raw_df["metric_raw"])
    raw_df["resp"] = zscore(raw_df["resp_raw"])
    raw_df["metric_mean"] = raw_df.groupby("subid")["metric"].transform("mean")
    raw_df["metric_within"] = raw_df["metric"] - raw_df["metric_mean"]
    raw_df["baseline_metric"] = raw_df.groupby("subid")["metric"].transform("first")
    raw_df["baseline_resp"] = raw_df.groupby("subid")["resp"].transform("first")
    df = raw_df.copy()

    if RUN_MIXED_MODELS:
        run_mixed_model_comparison(df)

    if only_models:
        return

    d = {"subid": subid, "time": time, "metric": zscore(data), "resp": zscore(fma_u0)}
    df = pd.DataFrame(data=d)

    if RUN_FMA_SCATTER:
        fig, ax = plt.subplots(figsize=(4, 4))
        y = data
        ax.scatter(fma_u0, y, c="lightblue", edgecolors="k", s=50, zorder=3)
        coef, p = pearsonr(fma_u0, y)
        slope, intercept = np.polyfit(fma_u0, y, 1)
        x_fit = np.linspace(fma_u0.min(), fma_u0.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.5)
        ax.set_xlabel("FMA-UE")
        ax.set_ylabel(f"{hemi_label} {fea_label}")
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

    if RUN_SUBJECT_TRAJECTORY:
        fig, ax = plt.subplots(figsize=(8, 5))
        unique_subid = np.unique(subid)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subid)))

        for i, sid in enumerate(unique_subid):
            mask = subid == sid
            ax.plot(time[mask], data[mask], "-o", color=colors[i], lw=1.5, markersize=6, label=f"Subj {sid}", alpha=0.8)

        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel(f"{hemi_label} {fea_label}")
        ax.legend(fontsize=8, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show()

    time_round = np.round(time).astype(int)
    unique_weeks = np.sort(np.unique(time_round))
    n_plots = min(6, len(unique_weeks))

    if RUN_TOPO_BY_WEEK:
        eeg_root = "/mnt/pci-0000:00:17.0-ata-6/VNS/VNS-EEG"
        eeg_sub = os.listdir(eeg_root)
        mfffile = os.listdir(os.path.join(eeg_root, eeg_sub[0]))
        raw = mne.io.read_raw_egi(os.path.join(eeg_root, eeg_sub[0], mfffile[0]), preload=True)
        raw.drop_channels(["VREF"])
        raw.info.get_montage()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        im = None
        for i in range(n_plots):
            wk = unique_weeks[i]
            mask = time_round == wk
            visdata = np.asarray(index_all)[mask].mean(axis=0) - 0.3
            im, cn = mne.viz.plot_topomap(visdata, raw.info, axes=axes[i], show=False, extrapolate="local")
            axes[i].set_title(f"Week {wk} (n={mask.sum()})", fontsize=11)

        for i in range(n_plots, 6):
            axes[i].set_visible(False)

        fig.suptitle("EEG Topomap by Week (averaged across subjects)", fontsize=14)
        fig.colorbar(im, ax=axes, orientation="horizontal", label=f"{fea_label}-0.3", fraction=0.04, pad=0.08)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.show()

    week_mean = np.array([np.asarray(data)[np.asarray(time_round) == wk].mean() for wk in unique_weeks])
    week_std = np.array([np.asarray(data)[np.asarray(time_round) == wk].std() for wk in unique_weeks])
    week_n = np.array([(np.asarray(time_round) == wk).sum() for wk in unique_weeks])
    week_se = week_std / np.sqrt(week_n)

    if RUN_WEEK_TREND:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(time_round, data, alpha=0.3, color="gray", s=20, label="Individual data")
        ax.errorbar(
            unique_weeks,
            week_mean,
            yerr=week_se,
            fmt="-o",
            capsize=4,
            color="#2E86C1",
            ecolor="gray",
            elinewidth=1.5,
            markersize=6,
            label="Mean +/- SE",
        )
        r, p_val = pearsonr(time_round, data)
        slope, intercept = np.polyfit(time_round, data, 1)
        x_fit = np.linspace(time_round.min(), time_round.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.5, label=f"Linear fit (r={r:.3f}, p={p_val:.3g})")
        ax.set_xlabel("Week")
        ax.set_ylabel(f"{hemi_label} {fea_label}")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show()

    if RUN_PAIRED_TEST:
        print("\n" + "=" * 50)
        print(f"Paired test: {hemi_label} {fea_label} each week vs baseline (week 0)")
        print("=" * 50)
        print(
            f"[DEBUG] data type: {type(data).__name__}, subid type: {type(np.asarray(subid)).__name__}, "
            f"time_round type: {type(np.asarray(time_round)).__name__}"
        )

        data_arr = np.asarray(data)
        subid_arr = np.asarray(subid)
        time_round_arr = np.asarray(time_round)
        unique_weeks_arr = np.sort(np.unique(time_round_arr))

        baseline_data = data_arr[time_round_arr == 0]
        baseline_subid = subid_arr[time_round_arr == 0]
        bl_map = dict(zip(baseline_subid, baseline_data))

        for wk in unique_weeks_arr[unique_weeks_arr > 0]:
            mask = time_round_arr == wk
            wk_data = data_arr[mask]
            wk_subid = subid_arr[mask]
            pairs = [(bl_map[sid], wk_data[i]) for i, sid in enumerate(wk_subid) if sid in bl_map]
            if len(pairs) >= 3:
                bl_vals, wk_vals = zip(*pairs)
                bl_arr = np.array(bl_vals)
                wk_arr = np.array(wk_vals)
                stat_w, p_w = wilcoxon(bl_arr, wk_arr, alternative="greater")
                stat_t, p_t = ttest_rel(bl_arr, wk_arr, alternative="greater")
                decrease = (wk_arr < bl_arr).mean() * 100
                print(
                    f"Week {wk:>3d}: n={len(pairs)}, "
                    f"Wilcoxon W={stat_w:.1f}, p={p_w:.4f} | "
                    f"t-test t={stat_t:.3f}, p={p_t:.4f}, "
                    f"{decrease:.0f}% subjects increase"
                )

    if RUN_DELTA_CORR:
        print("\n" + "=" * 50)
        print(f"Correlation: delta {fea_label} vs delta FMA-UE (change from baseline)")
        print("=" * 50)
        data_arr = np.asarray(data)
        subid_arr = np.asarray(subid)
        time_round_arr = np.asarray(time_round)
        target_weeks = [2, 4, 6]

        results = {wk: {"subid": [], "bl_dis": [], "d_fma": []} for wk in target_weeks}
        for sid in np.unique(subid_arr):
            mask = subid_arr == sid
            bl_idx = np.where(mask & (time_round_arr == 0))[0]
            if len(bl_idx) == 0:
                continue
            bl_dis = data_arr[bl_idx[0]]
            bl_fma = np.asarray(fma_u0)[bl_idx[0]]
            for wk in target_weeks:
                wk_idx = np.where(mask & (time_round_arr == wk))[0]
                if len(wk_idx) > 0:
                    results[wk]["subid"].append(sid)
                    results[wk]["bl_dis"].append(bl_dis)
                    results[wk]["d_fma"].append(np.asarray(fma_u0)[wk_idx[0]] - bl_fma)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for i, wk in enumerate(target_weeks):
            ax = axes[i]
            bl_dis = np.array(results[wk]["bl_dis"])
            d_fma = np.array(results[wk]["d_fma"])
            n = len(bl_dis)
            r, p = pearsonr(bl_dis, d_fma)
            ax.scatter(bl_dis, d_fma, c="lightblue", edgecolors="k", s=50, zorder=3)
            if n >= 3:
                slope, intercept = np.polyfit(bl_dis, d_fma, 1)
                x_fit = np.linspace(bl_dis.min(), bl_dis.max(), 100)
                ax.plot(x_fit, slope * x_fit + intercept, "r--", lw=1.5)
            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.4)
            ax.set_xlabel(f"Baseline {fea_label}")
            ax.set_ylabel("Delta FMA-UE")
            ax.set_title(f"Week {wk} (n={n})")
            ax.text(
                0.95,
                0.05,
                f"r={r:.3f}\np={p:.3g}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
            )
            ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args.fea, args.hemi, args.only_models)
