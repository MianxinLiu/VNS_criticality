#!/usr/bin/env python3
"""Fixed-point and stability-boundary analysis for BW-rWW fits."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("BRAINEVENT_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "brainevent"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import dmf_rww_ei_jax_gpu_fc_only as base

jax.config.update("jax_enable_x64", True)

DATA_DEFAULT = ROOT / "VNS_active_only_exclude_07_09.npz"
FIT_JSON_DEFAULT = ROOT / "dmf_rww_ei_jax_gpu_bw_fc_only_20260824_bw_forest90_fixed_sigma_i0.json"
OUT_DEFAULT = ROOT / "stability"
SESSIONS = ("baseline", "followup6w")
PARAM_NAMES = ("G", "w_EE", "w_EI", "w_IE")
EPS = 1e-12


@jax.jit
def h_transfer(x, a, b, d):
    y = a * x - b
    den = 1.0 - jnp.exp(-d * y)
    out = jnp.where(jnp.abs(den) < 1e-8, 1.0 / d, y / den)
    return jnp.clip(out, 0.0, 500.0)


def no_overwrite_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return path


def neural_rhs_jax(x, sc, params, fixed_i0):
    n_roi = sc.shape[0]
    s_e = x[:n_roi]
    s_i = x[n_roi:]
    g, w_ee, w_ei, w_ie = params

    j_nmda = 0.15
    a_e, b_e, d_e = 310.0, 125.0, 0.16
    a_i, b_i, d_i = 615.0, 177.0, 0.087
    gamma_e, gamma_i = 0.641, 1.0
    tau_e, tau_i = 0.1, 0.01
    w_e, w_i, w_ii = 1.0, 0.7, 1.0

    coupling = sc @ s_e
    i_e = w_e * fixed_i0 + w_ee * j_nmda * s_e + g * j_nmda * coupling - w_ie * s_i
    i_i = w_i * fixed_i0 + w_ei * j_nmda * s_e - w_ii * s_i
    r_e = h_transfer(i_e, a_e, b_e, d_e)
    r_i = h_transfer(i_i, a_i, b_i, d_i)
    ds_e = -s_e / tau_e + (1.0 - s_e) * gamma_e * r_e
    ds_i = -s_i / tau_i + gamma_i * r_i
    return jnp.concatenate([ds_e, ds_i])


def neural_summary(x, sc, params, fixed_i0):
    n_roi = sc.shape[0]
    s_e = x[:n_roi]
    s_i = x[n_roi:]
    g, w_ee, w_ei, w_ie = params
    j_nmda = 0.15
    a_e, b_e, d_e = 310.0, 125.0, 0.16
    a_i, b_i, d_i = 615.0, 177.0, 0.087
    i_e = fixed_i0 + w_ee * j_nmda * s_e + g * j_nmda * (sc @ s_e) - w_ie * s_i
    i_i = 0.7 * fixed_i0 + w_ei * j_nmda * s_e - s_i
    r_e = np.asarray(h_transfer(jnp.asarray(i_e), a_e, b_e, d_e))
    r_i = np.asarray(h_transfer(jnp.asarray(i_i), a_i, b_i, d_i))
    return {
        "S_E": np.asarray(s_e),
        "S_I": np.asarray(s_i),
        "I_E": np.asarray(i_e),
        "I_I": np.asarray(i_i),
        "r_E": r_e,
        "r_I": r_i,
        "S_E_over_S_I": np.asarray(s_e) / (np.asarray(s_i) + EPS),
    }


def rhs_np(x, sc, params, fixed_i0):
    out = neural_rhs_jax(jnp.asarray(x), jnp.asarray(sc), jnp.asarray(params), fixed_i0)
    return np.asarray(jax.device_get(out), dtype=float)


def solve_fixed_point(sc, params, fixed_i0, x0=None):
    n_roi = sc.shape[0]
    if x0 is None:
        x0 = np.concatenate([np.full(n_roi, 0.05), np.full(n_roi, 0.05)])
    lower = np.zeros(2 * n_roi)
    upper = np.concatenate([np.ones(n_roi), np.full(n_roi, 5.0)])
    res = least_squares(
        lambda z: rhs_np(z, sc, params, fixed_i0),
        np.clip(x0, lower + 1e-9, upper - 1e-9),
        jac=lambda z: jacobian_np(z, sc, params, fixed_i0),
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=400,
    )
    residual = rhs_np(res.x, sc, params, fixed_i0)
    return res.x, float(np.linalg.norm(residual) / np.sqrt(residual.size)), bool(res.success), res.message


def jacobian_np(x, sc, params, fixed_i0):
    jac_fn = jax.jacfwd(lambda z: neural_rhs_jax(z, jnp.asarray(sc), jnp.asarray(params), fixed_i0))
    return np.asarray(jax.device_get(jac_fn(jnp.asarray(x))), dtype=float)


def side_weighted_roi_energy(roi_energy, left_motor, right_motor, subject_sides):
    left = float(np.nanmean(roi_energy[left_motor]))
    right = float(np.nanmean(roi_energy[right_motor]))
    n = len(subject_sides)
    n_right = sum(s == "R" for s in subject_sides)
    n_left = sum(s == "L" for s in subject_sides)
    contra = (n_right * left + n_left * right) / n
    ipsi = (n_left * left + n_right * right) / n
    bilateral = 0.5 * (left + right)
    return left, right, contra, ipsi, bilateral


def stability_metrics(jac, left_motor, right_motor, subject_sides):
    n_roi = jac.shape[0] // 2
    vals, vecs = np.linalg.eig(jac)
    lead_i = int(np.argmax(vals.real))
    lead_val = vals[lead_i]
    vec = vecs[:, lead_i]
    energy = np.abs(vec) ** 2
    total_energy = float(np.sum(energy))
    motor = sorted(set(left_motor + right_motor))
    motor_idx = np.asarray(motor, dtype=int)
    motor_state_idx = np.concatenate([motor_idx, n_roi + motor_idx])
    motor_loading = float(np.sum(energy[motor_state_idx]) / total_energy)
    motor_size_fraction = len(motor) / n_roi
    roi_energy = energy[:n_roi] + energy[n_roi:]
    roi_energy = roi_energy / (np.sum(roi_energy) + EPS)
    left, right, contra, ipsi, bilateral = side_weighted_roi_energy(roi_energy, left_motor, right_motor, subject_sides)

    e_total = float(np.sum(energy[:n_roi]))
    i_total = float(np.sum(energy[n_roi:]))
    motor_e_loading = float(np.sum(energy[motor_idx]) / (e_total + EPS))
    motor_i_loading = float(np.sum(energy[n_roi + motor_idx]) / (i_total + EPS))
    local_idx = motor_state_idx
    local_jac = jac[np.ix_(local_idx, local_idx)]
    local_vals = np.linalg.eigvals(local_jac)
    return {
        "max_real_lambda": float(lead_val.real),
        "leading_lambda_imag": float(lead_val.imag),
        "stability_margin": float(-lead_val.real),
        "motor_loading_fraction": motor_loading,
        "motor_loading_enrichment": float(motor_loading / motor_size_fraction),
        "motor_E_loading_fraction": motor_e_loading,
        "motor_I_loading_fraction": motor_i_loading,
        "contra_motor_loading_density": contra,
        "ipsi_motor_loading_density": ipsi,
        "bilateral_motor_loading_density": bilateral,
        "local_motor_block_max_real_lambda": float(np.max(local_vals.real)),
    }


def crossing_distance(scan_rows, param_name):
    rows = sorted(scan_rows, key=lambda r: r["ratio"])
    fitted = min(rows, key=lambda r: abs(r["ratio"] - 1.0))
    y0 = fitted["max_real_lambda"]
    out = {
        "fit_max_real_lambda": y0,
        "critical_ratio": np.nan,
        "distance_log_ratio": np.nan,
        "crossing_side": "none",
    }
    best = None
    for a, b in zip(rows[:-1], rows[1:]):
        ya = a["max_real_lambda"]
        yb = b["max_real_lambda"]
        if not np.isfinite(ya) or not np.isfinite(yb):
            continue
        if ya == 0:
            candidate = (abs(np.log(a["ratio"])), a["ratio"])
        elif ya * yb <= 0:
            t = -ya / (yb - ya)
            ratio = a["ratio"] + t * (b["ratio"] - a["ratio"])
            candidate = (abs(np.log(ratio)), ratio)
        else:
            continue
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is not None:
        _, ratio = best
        out["critical_ratio"] = float(ratio)
        out["distance_log_ratio"] = float(abs(np.log(ratio)))
        out["crossing_side"] = "above" if ratio > 1 else "below"
    return out


def plot_results(fixed_rows, scan_rows, out_png, out_pdf):
    colors = {"baseline": "#466176", "followup6w": "#D96545"}
    labels = {"baseline": "Baseline", "followup6w": "Follow-up 6w"}
    fig = plt.figure(figsize=(12.8, 8.4))
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    scan_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_g = fig.add_subplot(gs[1, 2])
    ax_h = fig.add_subplot(gs[1, 3])

    x = np.arange(2)
    fixed_by = {r["session"]: r for r in fixed_rows}
    ax_a.bar(x, [fixed_by[s]["max_real_lambda"] for s in SESSIONS], color=[colors[s] for s in SESSIONS], width=0.58)
    ax_a.axhline(0, color="#2F3A45", lw=0.9)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([labels[s] for s in SESSIONS], rotation=15)
    ax_a.set_ylabel("max Re(lambda)")
    ax_a.set_title("Fitted-point stability")

    ax_b.bar(x, [fixed_by[s]["motor_loading_enrichment"] for s in SESSIONS], color=[colors[s] for s in SESSIONS], width=0.58)
    ax_b.axhline(1, color="#2F3A45", lw=0.9, ls="--")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([labels[s] for s in SESSIONS], rotation=15)
    ax_b.set_ylabel("Motor enrichment")
    ax_b.set_title("Leading-mode motor loading")

    for ax, pname in zip(scan_axes, PARAM_NAMES):
        for session in SESSIONS:
            sub = [r for r in scan_rows if r["session"] == session and r["param"] == pname]
            sub = sorted(sub, key=lambda r: r["ratio"])
            ax.plot([r["ratio"] for r in sub], [r["max_real_lambda"] for r in sub], color=colors[session], lw=1.8, label=labels[session])
        ax.axhline(0, color="#2F3A45", lw=0.8)
        ax.axvline(1, color="#8B96A3", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(0.23, 4.35)
        ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1.0, 2.0, 4.0]))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.25", "0.5", "1", "2", "4"]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(pname)
        ax.set_xlabel("p / p_fit")
        ax.set_ylabel("max Re(lambda)")
    scan_axes[0].legend(frameon=False, fontsize=8.5)

    width = 0.34
    for i, session in enumerate(SESSIONS):
        vals = [fixed_by[session]["motor_E_loading_fraction"], fixed_by[session]["motor_I_loading_fraction"]]
        ax_g.bar(np.arange(2) + (i - 0.5) * width, vals, width=width, color=colors[session], label=labels[session])
    ax_g.set_xticks(np.arange(2))
    ax_g.set_xticklabels(["E state", "I state"])
    ax_g.set_ylabel("Motor loading fraction")
    ax_g.set_title("E/I components")
    ax_g.legend(frameon=False, fontsize=8.5)

    ax_h.bar(x, [fixed_by[s]["local_motor_block_max_real_lambda"] for s in SESSIONS], color=[colors[s] for s in SESSIONS], width=0.58)
    ax_h.axhline(0, color="#2F3A45", lw=0.9)
    ax_h.set_xticks(x)
    ax_h.set_xticklabels([labels[s] for s in SESSIONS], rotation=15)
    ax_h.set_ylabel("max Re(lambda)")
    ax_h.set_title("Local motor Jacobian block")

    fig.suptitle("BW-rWW fixed-point stability and parameter continuation", fontsize=14, fontweight="bold", y=0.99)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--fit-json", type=Path, default=FIT_JSON_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--tag", default="20260825_bw_fixedpoint")
    p.add_argument("--n-ratios", type=int, default=25)
    p.add_argument("--ratio-min", type=float, default=0.25)
    p.add_argument("--ratio-max", type=float, default=4.0)
    p.add_argument("--subject-sides", nargs="+", default=["R", "R", "L", "R", "L", "L", "R"])
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_dir / args.tag
    fixed_csv = no_overwrite_path(prefix.with_name(prefix.name + "_fixed_points.csv"))
    scan_csv = no_overwrite_path(prefix.with_name(prefix.name + "_continuation.csv"))
    dist_csv = no_overwrite_path(prefix.with_name(prefix.name + "_distances.csv"))
    out_json = no_overwrite_path(prefix.with_suffix(".json"))
    out_md = no_overwrite_path(prefix.with_suffix(".md"))
    out_png = no_overwrite_path(prefix.with_suffix(".png"))
    out_pdf = no_overwrite_path(prefix.with_suffix(".pdf"))

    data = np.load(args.data, allow_pickle=True)
    sc_all = np.asarray(data["SC"], dtype=float)
    labels = np.asarray(data["roi_labels"], dtype=object)
    left_motor, right_motor = base.default_motor_rois(labels)
    motor = sorted(set(left_motor + right_motor))
    fit = json.loads(args.fit_json.read_text())
    fixed_i0 = float(fit["fixed"]["I0"])

    fixed_rows = []
    scan_rows = []
    distance_rows = []
    fixed_points = {}
    for sess_i, session in enumerate(SESSIONS):
        sc = base.prepare_sc(base.sanitize_matrix(base.finite_nanmean(sc_all[:, sess_i], axis=0)))
        params = np.asarray(fit["results"][session]["params"], dtype=float)
        x_star, residual, success, message = solve_fixed_point(sc, params, fixed_i0)
        jac = jacobian_np(x_star, sc, params, fixed_i0)
        metrics = stability_metrics(jac, left_motor, right_motor, args.subject_sides)
        summary = neural_summary(x_star, sc, params, fixed_i0)
        row = {
            "session": session,
            "fixed_point_success": success,
            "fixed_point_residual_rms": residual,
            "fixed_point_message": message,
            **{name: float(value) for name, value in zip(PARAM_NAMES, params)},
            "global_S_E": float(np.nanmean(summary["S_E"])),
            "global_S_I": float(np.nanmean(summary["S_I"])),
            "global_I_E": float(np.nanmean(summary["I_E"])),
            "global_r_E": float(np.nanmean(summary["r_E"])),
            "motor_S_E": float(np.nanmean(summary["S_E"][motor])),
            "motor_S_I": float(np.nanmean(summary["S_I"][motor])),
            "motor_I_E": float(np.nanmean(summary["I_E"][motor])),
            "motor_r_E": float(np.nanmean(summary["r_E"][motor])),
            "motor_S_E_over_S_I": float(np.nanmean(summary["S_E_over_S_I"][motor])),
            **metrics,
        }
        fixed_rows.append(row)
        fixed_points[session] = {"sc": sc, "params": params, "x_star": x_star}

        ratios = np.exp(np.linspace(np.log(args.ratio_min), np.log(args.ratio_max), args.n_ratios))
        for p_i, pname in enumerate(PARAM_NAMES):
            prev = x_star
            param_scan_rows = []
            for ratio in ratios:
                p_scan = params.copy()
                p_scan[p_i] = params[p_i] * ratio
                x_scan, res_scan, ok_scan, msg_scan = solve_fixed_point(sc, p_scan, fixed_i0, x0=prev)
                prev = x_scan
                jac_scan = jacobian_np(x_scan, sc, p_scan, fixed_i0)
                scan_metrics = stability_metrics(jac_scan, left_motor, right_motor, args.subject_sides)
                rec = {
                    "session": session,
                    "param": pname,
                    "ratio": float(ratio),
                    "param_value": float(p_scan[p_i]),
                    "fixed_point_success": ok_scan,
                    "fixed_point_residual_rms": res_scan,
                    **scan_metrics,
                }
                scan_rows.append(rec)
                param_scan_rows.append(rec)
            distance_rows.append({"session": session, "param": pname, **crossing_distance(param_scan_rows, pname)})

    with fixed_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fixed_rows)
    with scan_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scan_rows)
    with dist_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(distance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(distance_rows)

    plot_results(fixed_rows, scan_rows, out_png, out_pdf)
    payload = {
        "data": str(args.data),
        "fit_json": str(args.fit_json),
        "fixed_i0": fixed_i0,
        "param_names": PARAM_NAMES,
        "left_motor_rois": left_motor,
        "right_motor_rois": right_motor,
        "outputs": {
            "fixed_csv": fixed_csv.name,
            "scan_csv": scan_csv.name,
            "dist_csv": dist_csv.name,
            "figure_png": out_png.name,
            "figure_pdf": out_pdf.name,
        },
        "fixed_rows": fixed_rows,
        "distance_rows": distance_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    by = {r["session"]: r for r in fixed_rows}
    lines = [
        "# BW-rWW Fixed-Point Stability Analysis",
        "",
        f"Fit source: `{args.fit_json.name}`.",
        "The deterministic neural rWW subsystem was analyzed with `sigma=0`; the Balloon-Windkessel state was treated as the observation model and was not included in the neural Jacobian.",
        "",
        "## Fitted-Point Stability",
        "",
        "| session | max Re(lambda) | stability margin | motor loading | motor enrichment | local motor block max Re(lambda) | residual RMS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for session in SESSIONS:
        row = by[session]
        lines.append(
            "| {session} | {max_real_lambda:.6g} | {stability_margin:.6g} | {motor_loading_fraction:.3f} | "
            "{motor_loading_enrichment:.3f} | {local_motor_block_max_real_lambda:.6g} | {fixed_point_residual_rms:.3g} |".format(**row)
        )
    lines += [
        "",
        "## Continuation Distance",
        "",
        "| session | parameter | nearest critical ratio | log-distance | side |",
        "|---|---|---:|---:|---|",
    ]
    for row in distance_rows:
        pname = row["param"]
        lines.append(
            f"| {row['session']} | {pname} | {row['critical_ratio']:.6g} | "
            f"{row['distance_log_ratio']:.6g} | {row['crossing_side']} |"
        )
    lines += [
        "",
        f"![BW-rWW stability analysis]({out_png.name})",
        "",
        "## Files",
        "",
        f"- `{fixed_csv.name}`",
        f"- `{scan_csv.name}`",
        f"- `{dist_csv.name}`",
        f"- `{out_json.name}`",
        f"- `{out_png.name}`",
        f"- `{out_pdf.name}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
