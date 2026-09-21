#!/usr/bin/env python3
"""Signal-level fEI/dis_cri for BW-BOLD rWW-only fits."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
os.environ.setdefault("BRAINEVENT_CACHE_DIR", str(ROOT / ".cache" / "brainevent"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "project"))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import hilbert

import dmf_rww_ei_jax_gpu_fc_only as base
from dmf_rww_ei_jax_gpu_bw_fc_only import simulate_dmf_bw_jax
from functionEI import calculate_fei


DATA_DEFAULT = ROOT / "VNS_active_only_exclude_07_09.npz"
FIT_JSON_DEFAULT = ROOT / "dmf_rww_ei_jax_gpu_bw_fc_only_20260824_bw_forest90_fixed_sigma_i0.json"
SESSIONS = ("baseline", "followup6w")


def no_overwrite_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return path


def weighted_side_values(values, left_rois, right_rois, subject_sides):
    values = np.asarray(values, dtype=float)
    left = float(np.nanmean(values[left_rois]))
    right = float(np.nanmean(values[right_rois]))
    n = len(subject_sides)
    n_right = sum(s == "R" for s in subject_sides)
    n_left = sum(s == "L" for s in subject_sides)
    contra = (n_right * left + n_left * right) / n
    ipsi = (n_left * left + n_right * right) / n
    return left, right, contra, ipsi


def balanced_bilateral(values, left_rois, right_rois):
    values = np.asarray(values, dtype=float)
    left = float(np.nanmean(values[left_rois]))
    right = float(np.nanmean(values[right_rois]))
    return 0.5 * (left + right)


def model_fei_from_ts(ts_roi_time, window_size, overlap):
    ts = np.asarray(ts_roi_time, dtype=float)
    ts = ts - np.nanmean(ts, axis=1, keepdims=True)
    sd = np.nanstd(ts, axis=1, keepdims=True)
    ts = ts / (sd + 1e-12)
    envelope = np.abs(hilbert(ts, axis=1)).T
    fei, _, _ = calculate_fei(envelope, window_size, overlap)
    return fei, np.abs(1.0 - fei)


def simulate_bw_rww(sc, params, args, seed):
    steps_per_tr = max(1, int(round(args.tr_s / args.rww_dt_s)))
    burn_steps = int(round(args.burn_s / args.rww_dt_s))
    summary_start_steps = burn_steps
    total_steps = burn_steps + args.n_tr * steps_per_tr
    sc_j = jnp.asarray(base.prepare_sc(sc), dtype=jnp.float32)
    params_j = jnp.asarray(params, dtype=jnp.float32)
    bold_input = 0 if args.bold_input == "S_E" else 1
    bold, summary = simulate_dmf_bw_jax(
        sc_j,
        params_j,
        jax.random.PRNGKey(seed),
        args.n_tr,
        steps_per_tr,
        burn_steps,
        summary_start_steps,
        total_steps,
        args.rww_dt_s,
        args.fixed_sigma,
        args.fixed_i0,
        bold_input,
    )
    return np.asarray(jax.device_get(bold)), np.asarray(jax.device_get(summary))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="20260824_bw_rww")
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--fit-json", type=Path, default=FIT_JSON_DEFAULT)
    p.add_argument("--n-tr", type=int, default=360)
    p.add_argument("--tr-s", type=float, default=0.72)
    p.add_argument("--rww-dt-s", type=float, default=0.01)
    p.add_argument("--burn-s", type=float, default=30.0)
    p.add_argument("--window-size", type=int, default=60)
    p.add_argument("--window-overlap", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-sigma", type=float, default=None)
    p.add_argument("--fixed-i0", type=float, default=None)
    p.add_argument("--bold-input", choices=("S_E", "r_E"), default=None)
    p.add_argument("--subject-sides", nargs="+", default=["R", "R", "L", "R", "L", "L", "R"])
    return p.parse_args()


def main():
    args = parse_args()
    prefix = ROOT / f"model_signal_fei_compare_bw_rww_{args.tag}"
    out_csv = no_overwrite_path(prefix.with_suffix(".csv"))
    out_json = no_overwrite_path(prefix.with_suffix(".json"))
    out_md = no_overwrite_path(prefix.with_suffix(".md"))

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, got {jax.default_backend()}")

    data = np.load(args.data, allow_pickle=True)
    labels = np.asarray(data["roi_labels"], dtype=object)
    left_motor, right_motor = base.default_motor_rois(labels)
    motor = left_motor + right_motor
    sc = np.asarray(data["SC"], dtype=float)
    fc_emp = np.asarray(data["FC"], dtype=float)
    fit = json.loads(args.fit_json.read_text())
    fixed_sigma = float(fit["fixed"]["sigma"] if args.fixed_sigma is None else args.fixed_sigma)
    fixed_i0 = float(fit["fixed"]["I0"] if args.fixed_i0 is None else args.fixed_i0)
    bold_input = str(fit["bold_input"] if args.bold_input is None else args.bold_input)
    args.fixed_sigma = fixed_sigma
    args.fixed_i0 = fixed_i0
    args.bold_input = bold_input

    rows = []
    summary_names = ["S_E", "S_I", "I_E", "I_I", "r_E", "r_I", "S_E_over_S_I", "I_E_over_I_I", "r_E_over_r_I"]
    for sess_i, session in enumerate(SESSIONS):
        sc_group = base.sanitize_matrix(base.finite_nanmean(sc[:, sess_i], axis=0))
        fc_group = base.sanitize_matrix(base.finite_nanmean(fc_emp[:, sess_i], axis=0))
        params = fit["results"][session]["params"]
        ts, summary = simulate_bw_rww(sc_group, params, args, args.seed + sess_i)
        summary_by_name = {name: summary[i] for i, name in enumerate(summary_names)}
        fc_r = base.corr_upper(fc_group, base.fc_from_ts(ts))
        fei, dis = model_fei_from_ts(ts, args.window_size, args.window_overlap)
        left_fei, right_fei, contra_fei, ipsi_fei = weighted_side_values(fei, left_motor, right_motor, args.subject_sides)
        left_dis, right_dis, contra_dis, ipsi_dis = weighted_side_values(dis, left_motor, right_motor, args.subject_sides)
        rows.append(
            {
                "model": "rWW-only BW",
                "session": session,
                "fc_r": float(fc_r),
                "global_fEI": float(np.nanmean(fei)),
                "motor_fEI": balanced_bilateral(fei, left_motor, right_motor),
                "left_motor_fEI": left_fei,
                "right_motor_fEI": right_fei,
                "contra_fEI": contra_fei,
                "ipsi_fEI": ipsi_fei,
                "global_dis_cri": float(np.nanmean(dis)),
                "motor_dis_cri": balanced_bilateral(dis, left_motor, right_motor),
                "left_motor_dis_cri": left_dis,
                "right_motor_dis_cri": right_dis,
                "contra_dis_cri": contra_dis,
                "ipsi_dis_cri": ipsi_dis,
                "motor_internal_I_E": float(np.nanmean(summary_by_name["I_E"][motor])),
                "motor_internal_r_E": float(np.nanmean(summary_by_name["r_E"][motor])),
                "motor_internal_S_E_over_S_I": float(np.nanmean(summary_by_name["S_E_over_S_I"][motor])),
            }
        )

    with out_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "fit_json": str(args.fit_json),
        "data": str(args.data),
        "n_tr": args.n_tr,
        "window_size": args.window_size,
        "window_overlap": args.window_overlap,
        "fixed_sigma": fixed_sigma,
        "fixed_i0": fixed_i0,
        "bold_input": bold_input,
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by = {row["session"]: row for row in rows}
    b = by["baseline"]
    f = by["followup6w"]
    lines = [
        "# BW-rWW model-simulated signal fEI",
        "",
        "Signal-level fEI was computed from Hilbert envelopes of regenerated BW-BOLD model time series using `project/functionEI.py`.",
        f"Fit source: `{args.fit_json.name}`; n_tr={args.n_tr}; window_size={args.window_size}; overlap={args.window_overlap}.",
        "",
        "| session | FC r | motor fEI | motor dis_cri | contra dis_cri | ipsi dis_cri | motor I_E | motor r_E | motor S_E/S_I |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {session} | {fc_r:.3f} | {motor_fEI:.3f} | {motor_dis_cri:.3f} | {contra_dis_cri:.3f} | {ipsi_dis_cri:.3f} | "
            "{motor_internal_I_E:.4f} | {motor_internal_r_E:.3f} | {motor_internal_S_E_over_S_I:.3f} |".format(**row)
        )
    lines += [
        "",
        "## Pre-post change",
        "",
        "| metric | delta |",
        "|---|---:|",
        f"| FC r | {f['fc_r'] - b['fc_r']:+.3f} |",
        f"| motor fEI | {f['motor_fEI'] - b['motor_fEI']:+.3f} |",
        f"| motor dis_cri | {f['motor_dis_cri'] - b['motor_dis_cri']:+.3f} |",
        f"| contra dis_cri | {f['contra_dis_cri'] - b['contra_dis_cri']:+.3f} |",
        f"| ipsi dis_cri | {f['ipsi_dis_cri'] - b['ipsi_dis_cri']:+.3f} |",
        "",
        f"- `{out_csv.name}`",
        f"- `{out_json.name}`",
        f"- `{out_md.name}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
