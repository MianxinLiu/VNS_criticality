#!/usr/bin/env python3
"""Generate manuscript-style figures for paper_draft_0730.md.

The EEG figures are built from subject-level NPY outputs under data/cohort1
and data/cohort2. Report-level summary values are used only for the merged
paired statistics shown in aggregate panels.
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.stats import pearsonr, sem, ttest_rel


OUT_DIR = Path("figures/")

COLORS = {
    "baseline": "#445B6E",
    "post": "#D96545",
    "green": "#2E8B73",
    "gold": "#C4972F",
    "gray": "#8A96A3",
    "light": "#B9C2CC",
    "dark": "#263244",
    "grid": "#D8DEE6",
}

HEMI_SPECS = [
    ("ipsi", "Contralateral"),
    ("contra", "Ipsilateral"),
    ("bilateral", "Bilateral mean"),
]
HEMI_PAIR_KEYS = ["ipsi", "contra"]
HEMI_PAIR_TITLES = ["contralateral", "ipsilateral"]

LEFT_ELC = [49, 42, 24, 64, 59, 44, 76, 66, 79]
RIGHT_ELC = [207, 206, 213, 185, 183, 194, 143, 164, 172]

FMA_C1 = np.array(
    [
        30,
        38,
        27,
        27,
        20,
        23,
        39,
        41,
        23,
        24,
        39,
        43,
        47,
        50,
        27,
        30,
        21,
        22,
    ]
).reshape(-1, 2)

FMA_C2 = np.array(
    [
        40,
        48,
        52,
        51,
        51,
        53,
        61,
        63,
        10,
        15,
        16,
        24,
        22,
        26,
        49,
        48,
        54,
        33,
        44,
        55,
        35,
        43,
        46,
        13,
        23,
        31,
        28,
        24,
        39,
        48,
        48,
        48,
        47,
        47,
        59,
        60,
        62,
        44,
        47,
        54,
        56,
    ]
)


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def savefig(name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pad_inches = 0.0 if name == "figure6_rww_bw_stability_results" else 0.1
    for ext in ("png", "pdf"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", pad_inches=pad_inches)
    plt.close()


def add_trajectory_caption(fig, handles):
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )


def panel_label(ax, label):
    bbox = ax.get_position()
    left_label_x = ax.figure.subplotpars.left - 0.035
    label_x = left_label_x if bbox.x0 <= ax.figure.subplotpars.left + 0.08 else bbox.x0 - 0.035
    ax.figure.text(
        label_x,
        bbox.y1 + 0.012,
        label,
        transform=ax.figure.transFigure,
        fontsize=13,
        fontweight="bold",
        color=COLORS["dark"],
        ha="left",
        va="bottom",
    )


def load_metric(path, feature):
    arr = np.load(path / f"{feature}.npy", allow_pickle=True)
    if feature == "MSE":
        arr = np.nanmean(arr, axis=2)
    return np.asarray(arr, dtype=float)


def metric_transform(feature, arr):
    if feature == "fEI":
        return np.abs(1 - arr), "dis_cri"
    return arr, feature


def load_cohort1(feature, transform=True):
    """Pre/post dataset from data/cohort2, displayed as Cohort 2."""
    root = Path("data/cohort2")
    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    side = np.array(["R", "R", "L", "R", "L", "L", "R", "R", "L"])
    keep = np.ones(len(subjects), dtype=bool)
    keep[[6, 8]] = False

    all_ch = []
    for subject in subjects:
        arr = load_metric(subject, feature)[:2]
        if transform:
            arr, label = metric_transform(feature, arr)
        else:
            label = feature
        all_ch.append(arr)
    all_ch = np.stack(all_ch, axis=0)

    left = all_ch[:, :, LEFT_ELC].mean(axis=2)
    right = all_ch[:, :, RIGHT_ELC].mean(axis=2)
    ipsi = np.stack([left[i] if s == "L" else right[i] for i, s in enumerate(side)], axis=0)
    contra = np.stack([left[i] if s == "R" else right[i] for i, s in enumerate(side)], axis=0)
    bilateral = (ipsi + contra) / 2
    return {
        "label": label,
        "subjects": [p.name for p in subjects],
        "keep": keep,
        "channels": all_ch,
        "ipsi": ipsi,
        "contra": contra,
        "bilateral": bilateral,
    }


def load_cohort2(feature, transform=True):
    """Longitudinal first-wave dataset from data/cohort1, displayed as Cohort 1."""
    root = Path("data/cohort1")
    visit_counts = {10: 3, 1: 5, 2: 6, 3: 3, 4: 3, 5: 3, 6: 5, 7: 5, 8: 4, 9: 4}
    subjects = [root / f"sub{sid:02d}" for sid in visit_counts]
    side_by_subject_id = {1: "L", 2: "L", 3: "L", 4: "L", 5: "R", 6: "L", 7: "L", 8: "L", 9: "R", 10: "R"}

    rows = []
    all_week_channels = []
    fma_i = 0
    for subject in subjects:
        sid = int(subject.name.replace("sub", ""))
        n_visits = visit_counts[sid]
        weeks = np.load(subject / "Weeks.npy", allow_pickle=True)[:n_visits]
        arr = load_metric(subject, feature)[:n_visits]
        if transform:
            arr, label = metric_transform(feature, arr)
        else:
            label = feature
        left = arr[:, LEFT_ELC].mean(axis=1)
        right = arr[:, RIGHT_ELC].mean(axis=1)
        ipsi = left if side_by_subject_id[sid] == "L" else right
        contra = left if side_by_subject_id[sid] == "R" else right
        bilateral = (ipsi + contra) / 2
        for i, week in enumerate(weeks):
            rows.append(
                {
                    "sid": sid,
                    "week": int(round(float(week))),
                    "ipsi": float(ipsi[i]),
                    "contra": float(contra[i]),
                    "bilateral": float(bilateral[i]),
                    "fma": float(FMA_C2[fma_i]),
                }
            )
            all_week_channels.append((int(round(float(week))), arr[i]))
            fma_i += 1
    return {"label": label, "rows": rows, "channels": all_week_channels}


def load_cohort1_composite():
    se = load_cohort1("SE")
    dfa = load_cohort1("DFA")
    mse = load_cohort1("MSE")
    out = {"label": "raw composite", "subjects": se["subjects"], "keep": se["keep"]}
    for key in ("channels", "ipsi", "contra", "bilateral"):
        out[key] = se[key] + mse[key] - dfa[key]
    return out


def load_cohort2_composite():
    se = load_cohort2("SE")
    dfa = load_cohort2("DFA")
    mse = load_cohort2("MSE")
    rows = []
    for r_se, r_dfa, r_mse in zip(se["rows"], dfa["rows"], mse["rows"]):
        rows.append(
            {
                "sid": r_se["sid"],
                "week": r_se["week"],
                "ipsi": r_se["ipsi"] + r_mse["ipsi"] - r_dfa["ipsi"],
                "contra": r_se["contra"] + r_mse["contra"] - r_dfa["contra"],
                "bilateral": r_se["bilateral"] + r_mse["bilateral"] - r_dfa["bilateral"],
                "fma": r_se["fma"],
            }
        )
    channels = [(wk, se_ch + mse_ch - dfa_ch) for (wk, se_ch), (_, dfa_ch), (_, mse_ch) in zip(se["channels"], dfa["channels"], mse["channels"])]
    return {"label": "raw composite", "rows": rows, "channels": channels}


def get_info():
    montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
    info = mne.create_info(montage.ch_names, sfreq=1000.0, ch_types="eeg")
    info.set_montage(montage)
    return info


def plot_cohort1_pair(ax, data, ylabel, title, alt="greater"):
    keep = data["keep"]
    x = np.array([0, 1])
    vals = data["bilateral"][keep]
    for row in vals:
        ax.plot(x, row, color=COLORS["light"], lw=1.3, marker="o", ms=4, alpha=0.75)
    mean = vals.mean(axis=0)
    se = sem(vals, axis=0, nan_policy="omit")
    ax.errorbar(x, mean, yerr=se, color=COLORS["dark"], lw=2.4, marker="o", ms=6, capsize=4)
    _, p = ttest_rel(vals[:, 0], vals[:, 1], alternative=alt)
    ax.text(0.5, 0.96, f"n={vals.shape[0]}, paired p={p:.3g}", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Post"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, lw=0.8)


def plot_cohort1_ladder(ax, data, key, ylabel, title, alt="greater", ylim_from_kept=False):
    keep = data["keep"]
    x = np.array([0, 1])
    vals = data[key]
    kept_vals = vals[keep]
    skipped_vals = vals[~keep]
    skipped_color = "#7B61A8"

    for row in kept_vals:
        ax.plot(x, row, color=COLORS["light"], lw=1.25, marker="o", ms=3.8, alpha=0.72)
    for row in skipped_vals:
        ax.plot(x, row, color=skipped_color, lw=1.65, marker="D", ms=4.6, alpha=0.95)

    mean = np.nanmean(kept_vals, axis=0)
    se = sem(kept_vals, axis=0, nan_policy="omit")
    ax.errorbar(x, mean, yerr=se, color=COLORS["dark"], lw=2.25, marker="o", ms=5.5, capsize=3.5)
    _, p = ttest_rel(kept_vals[:, 0], kept_vals[:, 1], alternative=alt)
    ax.text(0.5, 0.96, f"kept n={kept_vals.shape[0]}, p={p:.3g}", ha="center", va="top", transform=ax.transAxes, fontsize=8.3)
    if ylim_from_kept:
        finite = kept_vals[np.isfinite(kept_vals)]
        if finite.size:
            y_min = float(np.nanmin(finite))
            y_max = float(np.nanmax(finite))
            pad = max((y_max - y_min) * 0.12, 0.05)
            ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "Post"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, lw=0.8)


def plot_topomap_set(fig, axes, info, arrays, titles, cbar_label, cmap, symmetric=False, split_last=False):
    if split_last and len(arrays) >= 3:
        main_vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays[:-1]])
        main_vlim = (np.nanpercentile(main_vals, 5), np.nanpercentile(main_vals, 95))
        diff_vals = arrays[-1][np.isfinite(arrays[-1])]
        diff_lim = np.nanpercentile(np.abs(diff_vals), 95)
        if not np.isfinite(diff_lim) or diff_lim == 0:
            diff_lim = np.nanmax(np.abs(diff_vals))
        diff_vlim = (-diff_lim, diff_lim)

        main_im = None
        for ax, arr, title in zip(axes[:-1], arrays[:-1], titles[:-1]):
            main_im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap=cmap, vlim=main_vlim, contours=0, sensors=False)
            ax.set_title(title, fontsize=9)
        diff_im, _ = mne.viz.plot_topomap(
            arrays[-1], info, axes=axes[-1], show=False, cmap=cmap, vlim=diff_vlim, contours=0, sensors=False
        )
        axes[-1].set_title(titles[-1], fontsize=9)
        fig.colorbar(main_im, ax=list(axes[:-1]), orientation="horizontal", fraction=0.046, pad=0.08, label=cbar_label)
        fig.colorbar(diff_im, ax=[axes[-1]], orientation="horizontal", fraction=0.046, pad=0.08, label=f"{cbar_label} change")
        return

    vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays])
    if symmetric:
        lim = np.nanmax(np.abs(vals))
        vlim = (-lim, lim)
    else:
        vlim = (np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
    im = None
    for ax, arr, title in zip(axes, arrays, titles):
        im, _ = mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=0, sensors=False)
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.046, pad=0.08, label=cbar_label)


def cohort2_matrix(cohort2, key="bilateral"):
    rows = cohort2["rows"]
    weeks = np.array(sorted(set(r["week"] for r in rows)))
    sids = np.array(sorted(set(r["sid"] for r in rows)))
    mat = np.full((len(sids), len(weeks)), np.nan)
    sid_index = {sid: i for i, sid in enumerate(sids)}
    week_index = {week: i for i, week in enumerate(weeks)}
    for row in rows:
        mat[sid_index[row["sid"]], week_index[row["week"]]] = row[key]
    return sids, weeks, mat


def plot_cohort2_trajectory(ax, cohort2, ylabel, title, stats):
    _, weeks, mat = cohort2_matrix(cohort2)
    for row in mat:
        ax.plot(weeks, row, color=COLORS["light"], lw=1.2, marker="o", ms=3.5, alpha=0.45)
    mean = np.nanmean(mat, axis=0)
    se = sem(mat, axis=0, nan_policy="omit")
    n = np.sum(np.isfinite(mat), axis=0)
    ax.errorbar(weeks, mean, yerr=se, color=COLORS["green"], lw=2.6, marker="o", ms=6, capsize=4)
    for week, nn in zip(weeks, n):
        ax.text(week, ax.get_ylim()[0], f"n={int(nn)}", ha="center", va="bottom", fontsize=7.5, color=COLORS["gray"])
    y0, y1 = ax.get_ylim()
    y = y1 - 0.08 * (y1 - y0)
    for i, (week, p) in enumerate(stats):
        y_stat = y - (i % 2) * 0.07 * (y1 - y0)
        ax.text(
            week,
            y_stat,
            f"p={p:.3g}",
            ha="center",
            va="top",
            fontsize=8.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0),
        )
    ax.set_xticks(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, lw=0.8)


def plot_cohort2_key_trajectory(ax, cohort2, key, ylabel, title, stats=None):
    _, weeks, mat = cohort2_matrix(cohort2, key=key)
    for row in mat:
        ax.plot(weeks, row, color=COLORS["light"], lw=1.1, marker="o", ms=3.2, alpha=0.42)
    mean = np.nanmean(mat, axis=0)
    se = sem(mat, axis=0, nan_policy="omit")
    n = np.sum(np.isfinite(mat), axis=0)
    ax.errorbar(weeks, mean, yerr=se, color=COLORS["green"], lw=2.35, marker="o", ms=5.3, capsize=3.5)
    y0, y1 = ax.get_ylim()
    for week, nn in zip(weeks, n):
        ax.text(week, y0, f"n={int(nn)}", ha="center", va="bottom", fontsize=7.0, color=COLORS["gray"])
    if stats:
        y = y1 - 0.08 * (y1 - y0)
        for i, (week, p) in enumerate(stats):
            y_stat = y - (i % 2) * 0.07 * (y1 - y0)
            ax.text(
                week,
                y_stat,
                f"p={p:.3g}",
                ha="center",
                va="top",
                fontsize=7.6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
            )
    ax.set_xticks(weeks)
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, lw=0.8)


def cohort2_paired_week_stats(cohort2, key, target_weeks, alt):
    _, weeks, mat = cohort2_matrix(cohort2, key=key)
    if 0 not in weeks:
        raise ValueError("Longitudinal paired stats require week 0 baseline data.")
    pre_idx = int(np.where(weeks == 0)[0][0])
    stats = []
    for week in target_weeks:
        if week not in weeks:
            continue
        week_idx = int(np.where(weeks == week)[0][0])
        vals = mat[:, [pre_idx, week_idx]]
        vals = vals[np.isfinite(vals).all(axis=1)]
        if vals.shape[0] < 3:
            continue
        _, p = ttest_rel(vals[:, 0], vals[:, 1], alternative=alt)
        stats.append((week, p))
    return stats


def merged_pre_post_from_week6(c1, c2, key):
    c1_vals = c1[key][c1["keep"]]
    sids, weeks, mat = cohort2_matrix(c2, key=key)
    if 0 not in weeks or 6 not in weeks:
        raise ValueError("Cohort 2 merge requires paired week 0 and week 6 data.")
    pre_idx = int(np.where(weeks == 0)[0][0])
    post_idx = int(np.where(weeks == 6)[0][0])
    c2_vals = mat[:, [pre_idx, post_idx]]
    c2_vals = c2_vals[np.isfinite(c2_vals).all(axis=1)]
    vals = np.vstack([c1_vals, c2_vals])
    groups = np.array(["C1"] * c1_vals.shape[0] + ["C2"] * c2_vals.shape[0])
    return vals, groups


def plot_merge_ladder(ax, c1, c2, key, ylabel, title, alt="greater"):
    vals, groups = merged_pre_post_from_week6(c1, c2, key)
    x = np.array([0, 1])
    for row, group in zip(vals, groups):
        color = COLORS["light"] if group == "C1" else "#A7BFA4"
        ax.plot(x, row, color=color, lw=1.25, marker="o", ms=3.7, alpha=0.75)
    mean = np.nanmean(vals, axis=0)
    se = sem(vals, axis=0, nan_policy="omit")
    ax.errorbar(x, mean, yerr=se, color=COLORS["dark"], lw=2.35, marker="o", ms=5.6, capsize=3.5)
    _, p = ttest_rel(vals[:, 0], vals[:, 1], alternative=alt)
    ax.text(0.5, 0.96, f"n={vals.shape[0]}, p={p:.3g}", ha="center", va="top", transform=ax.transAxes, fontsize=8.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "Post / wk6"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, lw=0.8)


def fisher_ci(r, n):
    if n <= 3 or not np.isfinite(r):
        return -1.0, 1.0
    r_clip = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r_clip)
    half_width = 1.96 / np.sqrt(n - 3)
    return float(np.tanh(z - half_width)), float(np.tanh(z + half_width))


def collect_change_behavior_pairs(c1, c2, key, return_groups=False, c2_weeks=None, c2_mode="baseline", include_c1_skipped=False):
    c1_keep = np.ones_like(c1["keep"], dtype=bool) if include_c1_skipped else c1["keep"]
    delta_metric = list(c1[key][c1_keep, 1] - c1[key][c1_keep, 0])
    delta_fma = list(FMA_C1[c1_keep, 1] - FMA_C1[c1_keep, 0])
    groups = ["sham group" if not keep else "active group" for keep in c1["keep"][c1_keep]]

    rows = c2["rows"]
    by_sid = {}
    for row in rows:
        by_sid.setdefault(row["sid"], {})[row["week"]] = row
    for visits in by_sid.values():
        if 0 not in visits:
            continue
        sorted_weeks = sorted(visits)
        if c2_mode == "adjacent":
            for prev_week, week in zip(sorted_weeks[:-1], sorted_weeks[1:]):
                if c2_weeks is not None and week not in c2_weeks:
                    continue
                previous = visits[prev_week]
                followup = visits[week]
                delta_metric.append(followup[key] - previous[key])
                delta_fma.append(followup["fma"] - previous["fma"])
                groups.append("Cohort 1")
        else:
            baseline = visits[0]
            for week in sorted_weeks:
                if week == 0:
                    continue
                if c2_weeks is not None and week not in c2_weeks:
                    continue
                followup = visits[week]
                delta_metric.append(followup[key] - baseline[key])
                delta_fma.append(followup["fma"] - baseline["fma"])
                groups.append("Cohort 1")

    if return_groups:
        return np.asarray(delta_fma, dtype=float), np.asarray(delta_metric, dtype=float), np.asarray(groups)
    return np.asarray(delta_fma, dtype=float), np.asarray(delta_metric, dtype=float)


def behavior_change_correlation_rows():
    loaders = [
        ("dis_cri", lambda: (load_cohort1("fEI"), load_cohort2("fEI"))),
        ("fEI", lambda: (load_cohort1("fEI", transform=False), load_cohort2("fEI", transform=False))),
        ("SE", lambda: (load_cohort1("SE"), load_cohort2("SE"))),
        ("DFA", lambda: (load_cohort1("DFA"), load_cohort2("DFA"))),
        ("MSE", lambda: (load_cohort1("MSE"), load_cohort2("MSE"))),
        ("Composite", lambda: (load_cohort1_composite(), load_cohort2_composite())),
    ]
    rows = []
    for metric, loader in loaders:
        c1, c2 = loader()
        for hemi in ("contra", "ipsi", "bilateral"):
            x, y = collect_change_behavior_pairs(c1, c2, hemi, c2_weeks=set(), include_c1_skipped=True)
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            r, p = pearsonr(x, y)
            lo, hi = fisher_ci(r, len(x))
            rows.append({"metric": metric, "hemi": hemi, "n": len(x), "r": r, "p": p, "lo": lo, "hi": hi})
    return rows


def zscore_safe(values):
    arr = np.asarray(values, dtype=float)
    sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return arr * np.nan
    return (arr - np.nanmean(arr)) / sd


def merged_lmm_rows():
    loaders = [
        ("dis_cri", lambda: (load_cohort1("fEI"), load_cohort2("fEI"))),
        ("fEI", lambda: (load_cohort1("fEI", transform=False), load_cohort2("fEI", transform=False))),
        ("SE", lambda: (load_cohort1("SE"), load_cohort2("SE"))),
        ("DFA", lambda: (load_cohort1("DFA"), load_cohort2("DFA"))),
        ("MSE", lambda: (load_cohort1("MSE"), load_cohort2("MSE"))),
        ("Composite", lambda: (load_cohort1_composite(), load_cohort2_composite())),
    ]
    rows = []
    for metric_name, loader in loaders:
        c1, c2 = loader()
        for hemi in ("contra", "ipsi", "bilateral"):
            recs = []
            for i, keep in enumerate(c1["keep"]):
                if not keep:
                    continue
                for visit_i, week in enumerate([0, 6]):
                    recs.append(
                        {
                            "subject": f"C1_{i + 1}",
                            "cohort": "C1",
                            "time": float(week),
                            "metric_raw": float(c1[hemi][i, visit_i]),
                            "resp_raw": float(FMA_C1[i, visit_i]),
                        }
                    )
            for row in c2["rows"]:
                recs.append(
                    {
                        "subject": f"C2_{row['sid']}",
                        "cohort": "C2",
                        "time": float(row["week"]),
                        "metric_raw": float(row[hemi]),
                        "resp_raw": float(row["fma"]),
                    }
                )
            df = pd.DataFrame(recs)
            df["metric"] = zscore_safe(df["metric_raw"])
            df["resp"] = zscore_safe(df["resp_raw"])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = smf.mixedlm("resp ~ metric + time + C(cohort)", data=df, groups=df["subject"]).fit(
                        reml=False, method="lbfgs", disp=False
                    )
                beta = float(result.params["metric"])
                p = float(result.pvalues["metric"])
            except Exception:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = smf.ols("resp ~ metric + time + C(cohort)", data=df).fit(
                        cov_type="cluster", cov_kwds={"groups": df["subject"]}
                    )
                beta = float(result.params["metric"])
                p = float(result.pvalues["metric"])
            rows.append({"hemi": hemi, "metric": metric_name, "beta": beta, "p": p})
    return rows


def plot_merge_aggregate(ax, rows, ylabel, title):
    x = np.arange(len(rows))
    colors = [COLORS["post"] if r["delta"] < 0 else COLORS["green"] for r in rows]
    ax.axhline(0, color=COLORS["dark"], lw=0.9)
    ax.bar(x, [r["delta"] for r in rows], color=colors, width=0.62)
    for i, r in enumerate(rows):
        va = "top" if r["delta"] < 0 else "bottom"
        y = r["delta"] - 0.012 if r["delta"] < 0 else r["delta"] + 0.012
        ax.text(i, y, f"p={r['p']:.3g}", ha="center", va=va, fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace(" ", "\n") for r in rows])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, lw=0.8)


def model_motor_discri_metrics(sig):
    if {"left_motor_dis_cri", "right_motor_dis_cri"}.issubset(sig.columns):
        left = [sig.loc["baseline", "left_motor_dis_cri"], sig.loc["followup6w", "left_motor_dis_cri"]]
        right = [sig.loc["baseline", "right_motor_dis_cri"], sig.loc["followup6w", "right_motor_dis_cri"]]
    else:
        # Existing CSVs stored only side-weighted contra/ipsi values.
        # The model summary used 4 right-side and 3 left-side subjects.
        contra = [sig.loc["baseline", "contra_dis_cri"], sig.loc["followup6w", "contra_dis_cri"]]
        ipsi = [sig.loc["baseline", "ipsi_dis_cri"], sig.loc["followup6w", "ipsi_dis_cri"]]
        left = [4 * c - 3 * i for c, i in zip(contra, ipsi)]
        right = [-3 * c + 4 * i for c, i in zip(contra, ipsi)]
    average = [0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1])]
    return [
        ("Left motor", left[0], left[1]),
        ("Right motor", right[0], right[1]),
        ("Average", average[0], average[1]),
    ]


def plot_figure2_criticality():
    info = get_info()
    c1 = load_cohort1("fEI")
    c2 = load_cohort2("fEI")

    fig = plt.figure(figsize=(13.2, 13.8))
    fig.subplots_adjust(top=0.985, bottom=0.075)
    gs = fig.add_gridspec(5, 6, height_ratios=[1.05, 1.0, 0.18, 1.0, 1.0], hspace=0.30, wspace=0.58)
    c2_axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    topo_axes = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    c1_axes = [fig.add_subplot(gs[3, i * 2 : (i + 1) * 2]) for i in range(3)]
    merge_axes = [fig.add_subplot(gs[4, i * 2 : (i + 1) * 2]) for i in range(3)]

    c2_stats = {
        "contra": [(2, 0.232), (4, 0.107), (6, 0.224)],
        "ipsi": [(2, 0.449), (4, 0.381), (6, 0.265)],
    }
    for ax, key, title in zip(c2_axes, HEMI_PAIR_KEYS, HEMI_PAIR_TITLES):
        plot_cohort2_key_trajectory(ax, c2, key, "dis_cri", f"Cohort 1 {title}", stats=c2_stats[key])
    panel_label(c2_axes[0], "A")

    keep = c1["keep"]
    pre_topo = np.nanmean(c1["channels"][keep, 0, :], axis=0)
    post_topo = np.nanmean(c1["channels"][keep, 1, :], axis=0)
    delta_topo = post_topo - pre_topo
    plot_topomap_set(
        fig,
        topo_axes,
        info,
        [pre_topo, post_topo, delta_topo],
        ["Baseline topomap", "Post topomap", "Post - baseline"],
        "Cohort 2 channel-level dis_cri",
        "RdBu_r",
        symmetric=False,
        split_last=True,
    )
    panel_label(topo_axes[0], "B")

    for ax, (key, title) in zip(c1_axes, HEMI_SPECS):
        plot_cohort1_ladder(ax, c1, key, "dis_cri", f"Cohort 2 {title}", alt="greater")
    panel_label(c1_axes[0], "C")

    for ax, (key, title) in zip(merge_axes, HEMI_SPECS):
        plot_merge_ladder(ax, c1, c2, key, "dis_cri", f"Merged Cohort {title}", alt="greater")
    panel_label(merge_axes[0], "D")

    handles = [
        plt.Line2D([0], [0], color="#A7BFA4", marker="o", lw=1.4, ms=4, label="Cohort 1 week 6"),
        plt.Line2D([0], [0], color=COLORS["light"], marker="o", lw=1.4, ms=4, label="Cohort 2 active group"),
        plt.Line2D([0], [0], color="#7B61A8", marker="D", lw=1.4, ms=4, label="Cohort 2 sham group"),
        plt.Line2D([0], [0], color=COLORS["dark"], marker="o", lw=2.2, ms=5, label="Mean +/- SEM"),
    ]
    add_trajectory_caption(fig, handles)
    savefig("figure2_criticality_results")


def plot_figure3_fei():
    info = get_info()
    c1 = load_cohort1("fEI", transform=False)
    c2 = load_cohort2("fEI", transform=False)

    fig = plt.figure(figsize=(13.2, 13.8))
    fig.subplots_adjust(top=0.985, bottom=0.075)
    gs = fig.add_gridspec(5, 6, height_ratios=[1.05, 1.0, 0.18, 1.0, 1.0], hspace=0.30, wspace=0.58)
    c2_axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    topo_axes = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    c1_axes = [fig.add_subplot(gs[3, i * 2 : (i + 1) * 2]) for i in range(3)]
    merge_axes = [fig.add_subplot(gs[4, i * 2 : (i + 1) * 2]) for i in range(3)]

    c2_stats = {
        "contra": cohort2_paired_week_stats(c2, "contra", [2, 4, 6], alt="less"),
        "ipsi": cohort2_paired_week_stats(c2, "ipsi", [2, 4, 6], alt="less"),
    }
    for ax, key, title in zip(c2_axes, HEMI_PAIR_KEYS, HEMI_PAIR_TITLES):
        plot_cohort2_key_trajectory(ax, c2, key, "fEI", f"Cohort 1 {title}", stats=c2_stats[key])
    panel_label(c2_axes[0], "A")

    keep = c1["keep"]
    pre_topo = np.nanmean(c1["channels"][keep, 0, :], axis=0)
    post_topo = np.nanmean(c1["channels"][keep, 1, :], axis=0)
    delta_topo = post_topo - pre_topo
    plot_topomap_set(
        fig,
        topo_axes,
        info,
        [pre_topo, post_topo, delta_topo],
        ["Baseline topomap", "Post topomap", "Post - baseline"],
        "Cohort 2 channel-level fEI",
        "RdBu_r",
        symmetric=False,
        split_last=True,
    )
    panel_label(topo_axes[0], "B")

    for ax, (key, title) in zip(c1_axes, HEMI_SPECS):
        plot_cohort1_ladder(ax, c1, key, "fEI", f"Cohort 2 {title}", alt="less")
    panel_label(c1_axes[0], "C")

    for ax, (key, title) in zip(merge_axes, HEMI_SPECS):
        plot_merge_ladder(ax, c1, c2, key, "fEI", f"Merged Cohort {title}", alt="less")
    panel_label(merge_axes[0], "D")

    handles = [
        plt.Line2D([0], [0], color="#A7BFA4", marker="o", lw=1.4, ms=4, label="Cohort 1 week 6"),
        plt.Line2D([0], [0], color=COLORS["light"], marker="o", lw=1.4, ms=4, label="Cohort 2 active group"),
        plt.Line2D([0], [0], color="#7B61A8", marker="D", lw=1.4, ms=4, label="Cohort 2 sham group"),
        plt.Line2D([0], [0], color=COLORS["dark"], marker="o", lw=2.2, ms=5, label="Mean +/- SEM"),
    ]
    add_trajectory_caption(fig, handles)
    savefig("figure3_fei_results")


def plot_figure4_complexity():
    info = get_info()
    c1 = load_cohort1_composite()
    c2 = load_cohort2_composite()

    fig = plt.figure(figsize=(13.2, 13.8))
    fig.subplots_adjust(top=0.985, bottom=0.075)
    gs = fig.add_gridspec(5, 6, height_ratios=[1.05, 1.0, 0.18, 1.0, 1.0], hspace=0.30, wspace=0.58)
    c2_axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    topo_axes = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    c1_axes = [fig.add_subplot(gs[3, i * 2 : (i + 1) * 2]) for i in range(3)]
    merge_axes = [fig.add_subplot(gs[4, i * 2 : (i + 1) * 2]) for i in range(3)]

    c2_stats = {
        "contra": cohort2_paired_week_stats(c2, "contra", [2, 4, 6], alt="less"),
        "ipsi": cohort2_paired_week_stats(c2, "ipsi", [2, 4, 6], alt="less"),
    }
    for ax, key, title in zip(c2_axes, HEMI_PAIR_KEYS, HEMI_PAIR_TITLES):
        plot_cohort2_key_trajectory(ax, c2, key, "Raw composite", f"Cohort 1 {title}", stats=c2_stats[key])
    panel_label(c2_axes[0], "A")

    keep = c1["keep"]
    pre_topo = np.nanmean(c1["channels"][keep, 0, :], axis=0)
    post_topo = np.nanmean(c1["channels"][keep, 1, :], axis=0)
    delta_topo = post_topo - pre_topo
    plot_topomap_set(
        fig,
        topo_axes,
        info,
        [pre_topo, post_topo, delta_topo],
        ["Baseline topomap", "Post topomap", "Post - baseline"],
        "Cohort 2 channel-level raw composite",
        "RdBu_r",
        symmetric=False,
        split_last=True,
    )
    panel_label(topo_axes[0], "B")

    for ax, (key, title) in zip(c1_axes, HEMI_SPECS):
        plot_cohort1_ladder(ax, c1, key, "Raw composite", f"Cohort 2 {title}", alt="less", ylim_from_kept=True)
    panel_label(c1_axes[0], "C")

    for ax, (key, title) in zip(merge_axes, HEMI_SPECS):
        plot_merge_ladder(ax, c1, c2, key, "Raw composite", f"Merged Cohort {title}", alt="less")
    panel_label(merge_axes[0], "D")

    handles = [
        plt.Line2D([0], [0], color="#A7BFA4", marker="o", lw=1.4, ms=4, label="Cohort 1 week 6"),
        plt.Line2D([0], [0], color=COLORS["light"], marker="o", lw=1.4, ms=4, label="Cohort 2 active group"),
        plt.Line2D([0], [0], color="#7B61A8", marker="D", lw=1.4, ms=4, label="Cohort 2 sham group"),
        plt.Line2D([0], [0], color=COLORS["dark"], marker="o", lw=2.2, ms=5, label="Mean +/- SEM"),
    ]
    add_trajectory_caption(fig, handles)
    savefig("figure4_complexity_results_with_cohort1")


def plot_figure5_behavior():
    fig = plt.figure(figsize=(13.2, 11.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.28], hspace=0.52, wspace=0.95)
    scatter_axes = [fig.add_subplot(gs[0, i * 2 : (i + 1) * 2]) for i in range(3)]
    ax_a = fig.add_subplot(gs[1, :3])
    ax_b = fig.add_subplot(gs[1, 3:])

    metrics = ["dis_cri", "fEI", "SE", "DFA", "MSE", "Composite"]
    hemi_order = ["ipsi", "contra", "bilateral"]
    hemi_labels = {"ipsi": "Contra", "contra": "Ipsi", "bilateral": "Bilateral"}
    hemi_colors = {"ipsi": COLORS["post"], "contra": COLORS["green"], "bilateral": COLORS["gold"]}
    offsets = {"ipsi": 0.22, "contra": 0.0, "bilateral": -0.22}

    corr_rows = behavior_change_correlation_rows()
    corr_lookup = {(r["metric"], r["hemi"]): r for r in corr_rows}
    y = np.arange(len(metrics))[::-1]
    ax_a.axvline(0, color=COLORS["dark"], lw=0.9)
    for yi, metric in zip(y, metrics):
        for hemi in hemi_order:
            row = corr_lookup[(metric, hemi)]
            ypos = yi + offsets[hemi]
            color = hemi_colors[hemi]
            ax_a.plot([row["lo"], row["hi"]], [ypos, ypos], color=color, lw=2.2, alpha=0.78)
            ax_a.scatter(row["r"], ypos, s=42, color=color, zorder=3)
            star = "*" if row["p"] < 0.05 else ""
            ax_a.text(1.03, ypos, f"{row['r']:+.2f}{star}", ha="left", va="center", fontsize=7.4, color=COLORS["dark"])
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(metrics)
    ax_a.set_xlabel("Pearson r: delta metric vs delta FMA-UE")
    ax_a.set_title("Cohort 2 change-score correlations")
    ax_a.set_xlim(-1.0, 1.18)
    ax_a.grid(axis="x", color=COLORS["grid"], alpha=0.75, lw=0.8)
    ax_a.legend(
        handles=[
            plt.Line2D([0], [0], color=hemi_colors[h], marker="o", lw=2.0, ms=5, label=hemi_labels[h])
            for h in hemi_order
        ],
        frameon=False,
        loc="lower left",
        fontsize=8.2,
    )
    panel_label(ax_a, "B")

    lmm = [
        ("contra", "dis_cri", 0.058, 0.345),
        ("contra", "fEI", -0.065, 0.291),
        ("contra", "SE", 0.171, 0.048),
        ("contra", "DFA", -0.175, 0.012),
        ("contra", "MSE", 0.104, 0.048),
        ("contra", "Composite", 0.160, 0.021),
        ("ipsi", "dis_cri", 0.107, 0.091),
        ("ipsi", "fEI", -0.108, 0.093),
        ("ipsi", "SE", 0.246, 0.009),
        ("ipsi", "DFA", -0.210, 0.000803),
        ("ipsi", "MSE", 0.092, 0.078),
        ("ipsi", "Composite", 0.192, 0.004),
        ("bilateral", "dis_cri", 0.086, 0.170),
        ("bilateral", "fEI", -0.089, 0.155),
        ("bilateral", "SE", 0.219, 0.015),
        ("bilateral", "DFA", -0.197, 0.003),
        ("bilateral", "MSE", 0.099, 0.057),
        ("bilateral", "Composite", 0.179, 0.008),
    ]
    beta_lookup = {(hemi, metric): (beta, p) for hemi, metric, beta, p in lmm}
    yb = np.arange(len(metrics))[::-1]
    ax_b.axvline(0, color=COLORS["dark"], lw=0.9)
    bar_h = 0.18
    for yi, metric in zip(yb, metrics):
        for hemi in hemi_order:
            beta, p = beta_lookup[(hemi, metric)]
            ypos = yi + offsets[hemi]
            alpha = 0.92 if p < 0.05 else 0.42
            ax_b.barh(ypos, beta, color=hemi_colors[hemi], height=bar_h, alpha=alpha)
            x = beta + (0.011 if beta >= 0 else -0.011)
            ha = "left" if beta >= 0 else "right"
            star = "*" if p < 0.05 else ""
            ax_b.text(x, ypos, f"{beta:+.2f}{star}", va="center", ha=ha, fontsize=7.4)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(metrics)
    ax_b.tick_params(axis="y", pad=2, labelsize=9)
    ax_b.set_xlabel("Standardized metric beta")
    ax_b.set_title("Cohort 1 longitudinal FMA-UE mixed model")
    ax_b.set_xlim(-0.30, 0.34)
    ax_b.grid(axis="x", color=COLORS["grid"], alpha=0.75, lw=0.8)
    ax_b.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=hemi_colors[h], label=hemi_labels[h])
            for h in hemi_order
        ],
        frameon=False,
        loc="lower left",
        fontsize=8.2,
    )
    panel_label(ax_b, "C")

    c1_dis = load_cohort1("fEI")
    c2_dis = load_cohort2("fEI")
    for ax, hemi in zip(scatter_axes, hemi_order):
        delta_fma, delta_metric, groups = collect_change_behavior_pairs(
            c1_dis, c2_dis, hemi, return_groups=True, c2_weeks=set(), include_c1_skipped=True
        )
        finite = np.isfinite(delta_fma) & np.isfinite(delta_metric)
        delta_fma = delta_fma[finite]
        delta_metric = delta_metric[finite]
        groups = groups[finite]
        r, p = pearsonr(delta_metric, delta_fma)
        for group, color, marker, label in [
            ("active group", COLORS["light"], "o", "Cohort 2 active group"),
            ("sham group", "#7B61A8", "D", "Cohort 2 sham group"),
            ("Cohort 1", "#A7BFA4", "o", "Cohort 1"),
        ]:
            mask = groups == group
            if not np.any(mask):
                continue
            ax.scatter(
                delta_metric[mask],
                delta_fma[mask],
                s=38,
                color=color,
                marker=marker,
                edgecolor=COLORS["dark"],
                linewidth=0.45,
                alpha=0.82,
                label=label,
                zorder=3,
            )
        slope, intercept = np.polyfit(delta_metric, delta_fma, 1)
        x_fit = np.linspace(delta_metric.min(), delta_metric.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, color=hemi_colors[hemi], lw=1.8)
        ax.axhline(0, color=COLORS["grid"], lw=0.9)
        ax.axvline(0, color=COLORS["grid"], lw=0.9)
        ax.text(
            0.96,
            0.06,
            f"r={r:+.2f}\np={p:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
        )
        ax.set_title(f"Cohort 2 {hemi_labels[hemi]} dis_cri")
        ax.set_xlabel("Delta dis_cri")
        ax.grid(color=COLORS["grid"], alpha=0.55, lw=0.75)
    scatter_axes[0].set_ylabel("Delta FMA-UE")
    scatter_axes[0].legend(frameon=False, loc="upper left", fontsize=8.0)
    panel_label(scatter_axes[0], "A")

    savefig("figure5_behavior_results")


def plot_figure6_rww():
    fig = plt.figure(figsize=(11.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    sessions = ["baseline", "Post"]
    fc = [0.141, 0.214]
    x = np.arange(2)
    ax_a.bar(x, fc, color=[COLORS["baseline"], COLORS["post"]], width=0.58)
    ax_a.plot(x, fc, color=COLORS["gray"], lw=1.7)
    ax_a.text(0.5, max(fc) + 0.018, "Δ=+0.072", ha="center", fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sessions)
    ax_a.set_ylabel("Fitted FC similarity")
    ax_a.set_title("rWW-only FC fit")
    ax_a.set_ylim(0, 0.27)
    panel_label(ax_a, "A")

    metrics = [
        ("Left motor", 0.3121305361298167, 0.3077067707036425),
        ("Right motor", 0.38760310009512405, 0.3375260832458095),
        ("Bilateral motor", 0.3552577155385638, 0.32474637787059507),
    ]
    centers = np.arange(len(metrics))
    width = 0.34
    ax_b.bar(centers - width / 2, [m[1] for m in metrics], width, color=COLORS["baseline"], label="baseline")
    ax_b.bar(centers + width / 2, [m[2] for m in metrics], width, color=COLORS["post"], label="Post")
    for i, m in enumerate(metrics):
        ax_b.plot([i - width / 2, i + width / 2], [m[1], m[2]], color=COLORS["gray"], lw=1.6)
        ax_b.text(i, max(m[1], m[2]) + 0.006, f"Δ={m[2] - m[1]:+.3f}", ha="center", fontsize=9)
    ax_b.set_xticks(centers)
    ax_b.set_xticklabels([m[0] for m in metrics])
    ax_b.set_ylabel("Simulated motor dis_cri")
    ax_b.set_title("Model signal motor dis_cri decreased")
    ax_b.set_ylim(0.285, 0.405)
    ax_b.legend(frameon=False, loc="upper right")
    panel_label(ax_b, "B")

    internal = [
        ("I_E", [7.445645771965139, 6.725165107309035, 7.034302268329967]),
        ("r_E", [44.06424441920917, 38.22905876893777, 40.79839810051969]),
        ("S_E/S_I", [188.551320338698, 191.70437699381256, 190.36015834906942]),
        ("w_IE", [24.74229068570199, 24.74229068570199, 24.74229068570199]),
    ]
    labels = [m[0] for m in internal]
    region_labels = ["Left", "Right", "Bilateral"]
    region_colors = ["#6C5B7B", COLORS["gold"], COLORS["green"]]
    x_c = np.arange(len(internal))
    width_c = 0.22
    ax_c.axhline(0, color=COLORS["dark"], lw=0.9)
    for j, (region, color) in enumerate(zip(region_labels, region_colors)):
        values = [row[1][j] for row in internal]
        xpos = x_c + (j - 1) * width_c
        ax_c.bar(xpos, values, color=color, width=width_c, label=region)
        for x_pos, value in zip(xpos, values):
            ax_c.text(x_pos, value + 2.2, f"{value:+.1f}%", ha="center", va="bottom", rotation=90, fontsize=7.2)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel("Post change from baseline (%)")
    ax_c.set_title("rWW-only motor-region internal variables", pad=18)
    ax_c.set_ylim(-8, 252)
    ax_c.legend(frameon=False, loc="center right")
    panel_label(ax_c, "C")

    savefig("figure6_rww_results")


def plot_figure6_rww_bw():
    fit_path = Path("data/model/dmf_rww_ei_jax_gpu_bw_fc_only_20260824_bw_forest90_fixed_sigma_i0.csv")
    sig_path = Path("data/model/model_signal_fei_compare_bw_rww_20260825_bw_forest90_fixed_sigma_i0_balanced.csv")
    fit = pd.read_csv(fit_path).set_index("session")
    sig = pd.read_csv(sig_path).set_index("session")

    fig = plt.figure(figsize=(12.2, 8.1))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.12], hspace=0.7, wspace=0.5)
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    c_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    sessions = ["baseline", "Post"]
    idx = ["baseline", "followup6w"]
    x = np.arange(2)

    fc = [fit.loc[k, "fc_r"] for k in idx]
    ax_a.bar(x, fc, color=[COLORS["baseline"], COLORS["post"]], width=0.58)
    ax_a.plot(x, fc, color=COLORS["gray"], lw=1.7)
    ax_a.text(0.5, max(fc) + 0.014, f"Δ={fc[1] - fc[0]:+.3f}", ha="center", fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sessions)
    ax_a.set_ylabel("Fitted FC similarity")
    ax_a.set_title("BW-rWW FC fit")
    ax_a.set_ylim(0, max(fc) + 0.06)
    panel_label(ax_a, "A")

    metrics = model_motor_discri_metrics(sig)
    centers = np.arange(len(metrics))
    width = 0.34
    ax_b.bar(centers - width / 2, [m[1] for m in metrics], width, color=COLORS["baseline"], label="baseline")
    ax_b.bar(centers + width / 2, [m[2] for m in metrics], width, color=COLORS["post"], label="Post")
    for i, m in enumerate(metrics):
        ax_b.plot([i - width / 2, i + width / 2], [m[1], m[2]], color=COLORS["gray"], lw=1.6)
        ax_b.text(i, max(m[1], m[2]) + 0.018, f"Δ={m[2] - m[1]:+.3f}", ha="center", fontsize=9)
    ax_b.set_xticks(centers)
    ax_b.set_xticklabels([m[0] for m in metrics])
    ax_b.set_ylabel("Simulated motor dis_cri")
    ax_b.set_title("BW-rWW model signal moved closer to criticality")
    ax_b.set_ylim(0, max(m[1] for m in metrics) + 0.075)
    ax_b.legend(frameon=False, loc="lower left", fontsize=8.6)
    panel_label(ax_b, "B")

    internal = [
        ("I_E", "motor_I_E", "Input current"),
        ("r_E", "motor_r_E", "Firing rate"),
        ("S_E/S_I", "motor_S_E_over_S_I", "Gating ratio"),
        ("w_IE", "w_IE", "I-to-E feedback"),
    ]
    for ax, (label, col, ylabel) in zip(c_axes, internal):
        vals = [fit.loc[k, col] for k in idx]
        ax.bar(x, vals, color=[COLORS["baseline"], COLORS["post"]], width=0.58)
        ax.plot(x, vals, color=COLORS["gray"], lw=1.4)
        ax.axhline(0, color=COLORS["dark"], lw=0.8)
        val_min = min(vals)
        val_max = max(vals)
        y_span = val_max - val_min if val_max != val_min else max(abs(v) for v in vals)
        ax.set_ylim(val_min - 0.18 * y_span, val_max + 0.26 * y_span)
        ax.text(
            0.5,
            0.94,
            f"Δ={vals[1] - vals[0]:+.3g}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["baseline", "Post"])
        ax.set_title(label, pad=10)
        ax.set_ylabel(ylabel)
    panel_label(c_axes[0], "C")

    savefig("figure6_rww_bw_results")


def plot_figure6_rww_bw_stability():
    fit_path = Path("data/model/dmf_rww_ei_jax_gpu_bw_fc_only_20260824_bw_forest90_fixed_sigma_i0.csv")
    sig_path = Path("data/model/model_signal_fei_compare_bw_rww_20260825_bw_forest90_fixed_sigma_i0_balanced.csv")
    fixed_path = Path("data/model/stability/20260825_bw_fixedpoint_v4_fixed_points.csv")
    scan_path = Path("data/model/stability/20260825_bw_fixedpoint_v4_continuation.csv")
    current_path = Path("data/model/bw_rww_motor_current_terms_20260904.csv")
    fit = pd.read_csv(fit_path).set_index("session")
    sig = pd.read_csv(sig_path).set_index("session")
    fixed = pd.read_csv(fixed_path).set_index("session")
    scan = pd.read_csv(scan_path)
    current = pd.read_csv(current_path).set_index("session")

    sessions = ["baseline", "Post"]
    idx = ["baseline", "followup6w"]
    x = np.arange(2)
    colors = [COLORS["baseline"], COLORS["post"]]

    fig = plt.figure(figsize=(13.8, 11.2))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.045)
    outer = fig.add_gridspec(4, 1, height_ratios=[0.9, 1.05, 1.0, 1.0], hspace=0.78)
    top = outer[0].subgridspec(1, 2, wspace=0.38)
    middle = outer[1].subgridspec(1, 4, wspace=0.62)
    continuation = outer[2].subgridspec(1, 4, wspace=0.55)
    bottom = outer[3].subgridspec(1, 4, wspace=0.62)
    ax_a = fig.add_subplot(top[0, 0])
    ax_param = fig.add_subplot(top[0, 1])
    ax_b = fig.add_subplot(middle[0, :2])
    d_axes = [fig.add_subplot(middle[0, 2]), fig.add_subplot(middle[0, 3])]
    f_axes = [fig.add_subplot(continuation[0, i]) for i in range(4)]
    c_axes = [fig.add_subplot(bottom[0, 0]), fig.add_subplot(bottom[0, 1])]
    ax_g = fig.add_subplot(bottom[0, 2:])

    def panel_label_inner(ax, label):
        panel_label(ax, label)

    fc = [fit.loc[k, "fc_r"] for k in idx]
    ax_a.bar(x, fc, color=colors, width=0.58)
    for x_pos, value in zip(x, fc):
        ax_a.text(x_pos, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(sessions)
    ax_a.set_ylabel("Fitted FC similarity")
    ax_a.set_ylim(0, max(fc) + 0.06)
    panel_label_inner(ax_a, "A")

    fitted_params = [("G", "G"), ("w_EE", "w_EE"), ("w_EI", "w_EI"), ("w_IE", "w_IE")]
    param_delta = np.array([fit.loc["followup6w", col] - fit.loc["baseline", col] for col, _ in fitted_params])
    param_labels = [label for _, label in fitted_params]
    y_param = np.arange(len(fitted_params))[::-1]
    ax_param.axvline(0, color=COLORS["dark"], lw=0.9)
    param_change_colors = ["#4E79A7", "#59A14F", "#8F6BB1", "#6B7280"]
    ax_param.barh(y_param, param_delta, color=param_change_colors, height=0.6)
    for y_pos, delta in zip(y_param, param_delta):
        x_text = delta + (0.06 if delta >= 0 else -0.06)
        ax_param.text(x_text, y_pos, f"{delta:+.3g}", va="center", ha="left" if delta >= 0 else "right", fontsize=8.2)
    ax_param.set_yticks(y_param)
    ax_param.set_yticklabels(param_labels)
    ax_param.set_xlabel("Post - baseline fitted parameter")
    ax_param.set_title("Fitted parameter changes")
    ax_param.set_xlim(min(param_delta.min() - 0.35, -0.15), max(param_delta.max() + 0.35, 0.15))
    panel_label_inner(ax_param, "B")

    metrics = model_motor_discri_metrics(sig)
    centers = np.arange(len(metrics))
    width = 0.34
    ax_b.bar(centers - width / 2, [m[1] for m in metrics], width, color=COLORS["baseline"], label="baseline")
    ax_b.bar(centers + width / 2, [m[2] for m in metrics], width, color=COLORS["post"], label="Post")
    for i, m in enumerate(metrics):
        ax_b.plot([i - width / 2, i + width / 2], [m[1], m[2]], color=COLORS["gray"], lw=1.6)
        ax_b.text(i, max(m[1], m[2]) + 0.018, f"Δ={m[2] - m[1]:+.3f}", ha="center", fontsize=9)
    ax_b.set_xticks(centers)
    ax_b.set_xticklabels([m[0] for m in metrics])
    ax_b.set_ylabel("Simulated motor dis_cri")
    ax_b.set_ylim(0, max(m[1] for m in metrics) + 0.075)
    ax_b.legend(frameon=False, loc="lower left", fontsize=8.6)
    panel_label_inner(ax_b, "C")

    internal = [
        ("I_E", "motor_I_E", "Input current"),
        ("S_E/S_I", "motor_S_E_over_S_I", "Gating ratio"),
    ]
    for ax, (label, col, ylabel) in zip(c_axes, internal):
        vals = [fit.loc[k, col] for k in idx]
        ax.bar(x, vals, color=colors, width=0.58)
        ax.plot(x, vals, color=COLORS["gray"], lw=1.4)
        ax.axhline(0, color=COLORS["dark"], lw=0.8)
        val_min = min(vals)
        val_max = max(vals)
        y_span = val_max - val_min if val_max != val_min else max(abs(v) for v in vals)
        ax.set_ylim(val_min - 0.18 * y_span, val_max + 0.26 * y_span)
        ax.text(0.5, 0.94, f"Δ={vals[1] - vals[0]:+.3g}", transform=ax.transAxes, ha="center", va="top", fontsize=8.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["baseline", "Post"])
        ax.set_title(f"Motor {ylabel}", pad=10)
        ax.set_ylabel(label)
    panel_label_inner(c_axes[0], "G")

    d_specs = [
        (d_axes[0], "max_real_lambda", "Cortical stability margin", "-max Re(lambda)"),
        (d_axes[1], "local_motor_block_max_real_lambda", "Motor stability margin", "-max Re(lambda)"),
    ]
    for ax, col, title, ylabel in d_specs:
        vals = [-fixed.loc[k, col] for k in idx]
        ax.bar(x, vals, color=colors, width=0.58)
        ax.plot(x, vals, color=COLORS["gray"], lw=1.4)
        ax.axhline(0, color=COLORS["dark"], lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(sessions)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    panel_label_inner(d_axes[0], "D")
    panel_label_inner(d_axes[1], "E")

    param_titles = {"G": "G", "w_EE": "w_EE", "w_EI": "w_EI", "w_IE": "w_IE"}
    for ax, param in zip(f_axes, ["G", "w_EE", "w_EI", "w_IE"]):
        for session, color, label in zip(idx, colors, sessions):
            sub = scan[(scan["session"] == session) & (scan["param"] == param)].sort_values("ratio")
            ax.plot(sub["ratio"], -sub["max_real_lambda"], color=color, lw=1.5, label=label)
        ax.axhline(0, color=COLORS["dark"], lw=0.75)
        ax.axvline(1, color=COLORS["gray"], lw=0.75, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.23, 4.35)
        ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1.0, 2.0, 4.0]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "4"]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(param_titles[param], pad=5, fontsize=8.5)
        ax.set_xlabel("p / p_fit", fontsize=7.5)
        ax.set_ylabel("-max Re(lambda)", fontsize=7.5)
        ax.tick_params(axis="both", labelsize=7.0)
    f_axes[0].legend(frameon=False, fontsize=7.0, loc="lower left")
    panel_label_inner(f_axes[0], "F")

    term_specs = [
        ("background", "I0"),
        ("local_E", "w_EE*S_E"),
        ("network_E", "G*SC*S_E"),
        ("local_inhibition", "w_IE*S_I"),
    ]
    term_delta = np.array([current.loc["followup6w", col] - current.loc["baseline", col] for col, _ in term_specs])
    # Display inhibitory feedback as the positive quantity w_IE*S_I; its
    # post-minus-baseline change is therefore the sign-reversed source term.
    term_delta[-1] *= -1
    term_labels = [label for _, label in term_specs]
    term_colors = ["#9AA4B2", "#2F6F9F", "#4B9B72", "#8F6BB1"]
    y_terms = np.arange(len(term_specs))[::-1]
    ax_g.axvline(0, color=COLORS["dark"], lw=0.9)
    ax_g.barh(y_terms, term_delta, color=term_colors, height=0.62)
    for y_pos, delta in zip(y_terms, term_delta):
        x_text = delta + (0.025 if delta >= 0 else -0.025)
        ax_g.text(x_text, y_pos, f"{delta:+.3f}", va="center", ha="left" if delta >= 0 else "right", fontsize=8.0)
    ax_g.set_yticks(y_terms)
    ax_g.set_yticklabels(term_labels)
    ax_g.set_xlabel("Post - baseline current")
    ax_g.set_title("Motor I_E current-term change")
    ax_g.set_xlim(min(term_delta.min() - 0.13, -0.08), max(term_delta.max() + 0.13, 0.08))
    panel_label_inner(ax_g, "H")

    savefig("figure6_rww_bw_stability_results")


def plot_supp_complexity_metric(feature, ylabel, alt, title, out_name):
    info = get_info()
    c1 = load_cohort1(feature)
    c2 = load_cohort2(feature)

    fig = plt.figure(figsize=(13.2, 13.8))
    fig.subplots_adjust(top=0.985, bottom=0.075)
    gs = fig.add_gridspec(5, 6, height_ratios=[1.05, 1.0, 0.18, 1.0, 1.0], hspace=0.30, wspace=0.58)
    c2_axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:])]
    topo_axes = [fig.add_subplot(gs[1, i * 2 : (i + 1) * 2]) for i in range(3)]
    c1_axes = [fig.add_subplot(gs[3, i * 2 : (i + 1) * 2]) for i in range(3)]
    merge_axes = [fig.add_subplot(gs[4, i * 2 : (i + 1) * 2]) for i in range(3)]

    c2_stats = {
        "contra": cohort2_paired_week_stats(c2, "contra", [2, 4, 6], alt=alt),
        "ipsi": cohort2_paired_week_stats(c2, "ipsi", [2, 4, 6], alt=alt),
    }
    for ax, key, hemi_title in zip(c2_axes, HEMI_PAIR_KEYS, HEMI_PAIR_TITLES):
        plot_cohort2_key_trajectory(ax, c2, key, ylabel, f"Cohort 1 {hemi_title}", stats=c2_stats[key])
    panel_label(c2_axes[0], "A")

    keep = c1["keep"]
    pre_topo = np.nanmean(c1["channels"][keep, 0, :], axis=0)
    post_topo = np.nanmean(c1["channels"][keep, 1, :], axis=0)
    delta_topo = post_topo - pre_topo
    plot_topomap_set(
        fig,
        topo_axes,
        info,
        [pre_topo, post_topo, delta_topo],
        ["Baseline topomap", "Post topomap", "Post - baseline"],
        f"Cohort 2 channel-level {ylabel}",
        "RdBu_r",
        symmetric=False,
        split_last=True,
    )
    panel_label(topo_axes[0], "B")

    for ax, (key, hemi_title) in zip(c1_axes, HEMI_SPECS):
        plot_cohort1_ladder(ax, c1, key, ylabel, f"Cohort 2 {hemi_title}", alt=alt, ylim_from_kept=True)
    panel_label(c1_axes[0], "C")

    for ax, (key, hemi_title) in zip(merge_axes, HEMI_SPECS):
        plot_merge_ladder(ax, c1, c2, key, ylabel, f"Merged Cohort {hemi_title}", alt=alt)
    panel_label(merge_axes[0], "D")

    handles = [
        plt.Line2D([0], [0], color="#A7BFA4", marker="o", lw=1.4, ms=4, label="Cohort 1 week 6"),
        plt.Line2D([0], [0], color=COLORS["light"], marker="o", lw=1.4, ms=4, label="Cohort 2 active group"),
        plt.Line2D([0], [0], color="#7B61A8", marker="D", lw=1.4, ms=4, label="Cohort 2 sham group"),
        plt.Line2D([0], [0], color=COLORS["dark"], marker="o", lw=2.2, ms=5, label="Mean +/- SEM"),
    ]
    add_trajectory_caption(fig, handles)
    savefig(out_name)


def plot_supp_complexity_metrics():
    plot_supp_complexity_metric(
        "SE",
        "SE",
        "less",
        "Motor EEG SE increased after VNS",
        "supplementary_complexity_se_results",
    )
    plot_supp_complexity_metric(
        "DFA",
        "DFA",
        "greater",
        "Motor EEG DFA decreased after VNS",
        "supplementary_complexity_dfa_results",
    )
    plot_supp_complexity_metric(
        "MSE",
        "MSE",
        "less",
        "Motor EEG MSE increased after VNS",
        "supplementary_complexity_mse_results",
    )


def plot_supp_lc_stimulation(
    path=Path("modelling/deco_lc_stimulation_robustness_20260731_lcstim_robust.csv"),
    out_name="supplementary_lc_stimulation_model",
):
    path = Path(path)
    if not path.exists():
        path = Path("modelling/history/no_bw_20260825") / path.name
    df = pd.read_csv(path)
    roi_order = ["left_central_DK24", "right_central_DK58"]
    roi_labels = {"left_central_DK24": "Left central gyrus (DK24)", "right_central_DK58": "Right central gyrus (DK58)"}
    roi_colors = {"left_central_DK24": "#6C5B7B", "right_central_DK58": COLORS["gold"]}
    doses = np.array(sorted(df["dose"].unique()))

    fig = plt.figure(figsize=(13.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25], wspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    lc = df.groupby("dose")["net_lc_delta_r_E"].agg(["mean", "sem"]).reindex(doses)
    ax_a.plot(doses, lc["mean"], color=COLORS["green"], marker="o", lw=2.0)
    ax_a.fill_between(doses, lc["mean"] - lc["sem"], lc["mean"] + lc["sem"], color=COLORS["green"], alpha=0.16, lw=0)
    ax_a.set_xscale("log")
    ax_a.set_xlabel("LC external input")
    ax_a.set_ylabel("Net delta LC r_E")
    ax_a.set_title("LC target engagement", fontsize=10.5)
    ax_a.axhline(0, color=COLORS["dark"], lw=0.8)
    panel_label(ax_a, "A")

    offsets = {"left_central_DK24": -0.06, "right_central_DK58": 0.06}
    for roi in roi_order:
        sub = df[df["roi"] == roi]
        stats = sub.groupby("dose")["net_delta_r_E"].agg(["mean", "sem"]).reindex(doses)
        xpos = doses * (1 + offsets[roi])
        ax_b.errorbar(
            xpos,
            stats["mean"],
            yerr=stats["sem"],
            color=roi_colors[roi],
            marker="o",
            lw=2.0,
            capsize=3,
            label=roi_labels[roi],
        )
    ax_b.set_xscale("log")
    ax_b.set_xlabel("LC external input")
    ax_b.set_ylabel("Net delta central-gyrus r_E")
    ax_b.set_title("Central-gyrus excitability", fontsize=10.5)
    ax_b.axhline(0, color=COLORS["dark"], lw=0.8)
    ax_b.legend(frameon=False, loc="upper left", fontsize=8.2)
    panel_label(ax_b, "B")

    rng = np.random.default_rng(31)
    x_positions = np.arange(len(doses))
    for i, dose in enumerate(doses):
        for j, roi in enumerate(roi_order):
            vals = df[(df["dose"] == dose) & (df["roi"] == roi)]["net_delta_dis_cri"].to_numpy()
            x = np.full_like(vals, x_positions[i] + (-0.16 if roi == roi_order[0] else 0.16), dtype=float)
            x = x + rng.uniform(-0.035, 0.035, size=len(vals))
            ax_c.scatter(x, vals, s=14, color=roi_colors[roi], alpha=0.46, edgecolor="none")
            mean = vals.mean()
            sem_val = vals.std(ddof=1) / np.sqrt(len(vals))
            ax_c.errorbar(
                x_positions[i] + (-0.16 if roi == roi_order[0] else 0.16),
                mean,
                yerr=sem_val,
                fmt="o",
                ms=4.5,
                color=roi_colors[roi],
                capsize=3,
                zorder=4,
            )
    ax_c.axhline(0, color=COLORS["dark"], lw=0.8)
    ax_c.set_xticks(x_positions)
    ax_c.set_xticklabels([f"{d:g}" for d in doses])
    ax_c.set_xlabel("LC external input")
    ax_c.set_ylabel("Net delta dis_cri")
    ax_c.set_title("Criticality was seed-unstable", fontsize=10.5)
    panel_label(ax_c, "C")

    savefig(out_name)


def plot_supp_lc_stimulation_rerun():
    plot_supp_lc_stimulation(
        Path("modelling/deco_lc_stimulation_robustness_20260824_lcstim_rerun_robust.csv"),
        "supplementary_lc_stimulation_model_20260824_rerun",
    )


def main():
    setup_style()
    plot_figure2_criticality()
    plot_figure3_fei()
    plot_figure4_complexity()
    plot_figure5_behavior()
    plot_figure6_rww()
    plot_figure6_rww_bw()
    plot_figure6_rww_bw_stability()
    plot_supp_complexity_metrics()
    plot_supp_lc_stimulation_rerun()


if __name__ == "__main__":
    main()
