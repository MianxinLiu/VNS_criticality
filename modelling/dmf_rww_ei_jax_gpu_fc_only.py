#!/usr/bin/env python3
"""JAX/GPU two-population DMF/rWW FC-only pilot.

This is a GPU-oriented companion to dmf_rww_ei_bayes_fc_only.py. It keeps the
same E/I outputs but uses JAX scan/vmap for the simulation and FIC grid.
Outputs are no-overwrite.
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("BRAINEVENT_CACHE_DIR", str(Path(__file__).resolve().parent / ".cache" / "brainevent"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import jax
import jax.numpy as jnp
import numpy as np
from skopt import forest_minimize, gp_minimize
from skopt.space import Real


ROOT = Path(__file__).resolve().parent
DATA_DEFAULT = ROOT / "VNS_active_only_exclude_07_09.npz"
SESSIONS = ("baseline", "followup6w")
EPS = 1e-12


def no_overwrite_path(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    return path


def sanitize_matrix(mat):
    mat = np.asarray(mat, dtype=float).copy()
    mat[~np.isfinite(mat)] = 0.0
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat


def prepare_sc(sc):
    sc = sanitize_matrix(sc)
    mx = np.nanmax(sc)
    if mx > 0:
        sc = sc / mx
    return sc


def finite_nanmean(arr, axis):
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=axis)


def corr_upper(a, b):
    iu = np.triu_indices_from(a, 1)
    av = np.asarray(a, dtype=float)[iu]
    bv = np.asarray(b, dtype=float)[iu]
    ok = np.isfinite(av) & np.isfinite(bv)
    if ok.sum() < 5:
        return np.nan
    av = av[ok]
    bv = bv[ok]
    if np.std(av) < EPS or np.std(bv) < EPS:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def fc_from_ts(ts):
    ts = np.asarray(ts, dtype=float)
    ts = ts[:, np.all(np.isfinite(ts), axis=0)]
    if ts.shape[1] < 5:
        return np.full((ts.shape[0], ts.shape[0]), np.nan)
    return np.corrcoef(ts)


def default_motor_rois(labels):
    left = [i for i, lab in enumerate(labels) if str(lab).startswith("LH_SomMot_")]
    right = [i for i, lab in enumerate(labels) if str(lab).startswith("RH_SomMot_")]
    return left, right


def weighted_side_values(values, left_rois, right_rois, subject_sides):
    left = float(np.nanmean(values[left_rois]))
    right = float(np.nanmean(values[right_rois]))
    n = len(subject_sides)
    n_right = sum(s == "R" for s in subject_sides)
    n_left = sum(s == "L" for s in subject_sides)
    contra = (n_right * left + n_left * right) / n
    ipsi = (n_left * left + n_right * right) / n
    return left, right, contra, ipsi


@jax.jit
def h_transfer(x, a, b, d):
    y = a * x - b
    den = 1.0 - jnp.exp(-d * y)
    out = jnp.where(jnp.abs(den) < 1e-8, 1.0 / d, y / den)
    return jnp.clip(out, 0.0, 500.0)


@jax.jit(static_argnames=("n_tr", "steps_per_tr", "burn_steps", "summary_start_steps", "total_steps"))
def simulate_dmf_jax(sc, params, w_ie, key, n_tr, steps_per_tr, burn_steps, summary_start_steps, total_steps, dt):
    G, w_EE, w_EI, sigma, I0 = params
    n_roi = sc.shape[0]

    J_NMDA = 0.15
    a_E, b_E, d_E = 310.0, 125.0, 0.16
    a_I, b_I, d_I = 615.0, 177.0, 0.087
    gamma_E, gamma_I = 0.641, 1.0
    tau_E, tau_I = 0.1, 0.01
    w_E, w_I, w_II = 1.0, 0.7, 1.0
    tau_bold_proxy = 1.5

    k1, k2 = jax.random.split(key)
    noise_e = jax.random.normal(k1, (total_steps, n_roi), dtype=jnp.float32)
    noise_i = jax.random.normal(k2, (total_steps, n_roi), dtype=jnp.float32)

    def step(carry, xs):
        S_E, S_I, B = carry
        ne, ni = xs
        coupling = sc @ S_E
        I_E = w_E * I0 + w_EE * J_NMDA * S_E + G * J_NMDA * coupling - w_ie * S_I
        I_I = w_I * I0 + w_EI * J_NMDA * S_E - w_II * S_I
        r_E = h_transfer(I_E, a_E, b_E, d_E)
        r_I = h_transfer(I_I, a_I, b_I, d_I)
        S_E = S_E + (-S_E / tau_E + (1.0 - S_E) * gamma_E * r_E) * dt + sigma * jnp.sqrt(dt) * ne
        S_I = S_I + (-S_I / tau_I + gamma_I * r_I) * dt + sigma * jnp.sqrt(dt) * ni
        S_E = jnp.clip(S_E, 0.0, 1.0)
        S_I = jnp.clip(S_I, 0.0, 1.0)
        B = B + dt * (S_E - B) / tau_bold_proxy
        out = (B, S_E, S_I, I_E, I_I, r_E, r_I)
        return (S_E, S_I, B), out

    init = (
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
    )
    _, hist = jax.lax.scan(step, init, (noise_e, noise_i), length=total_steps)
    B_hist, S_E_hist, S_I_hist, I_E_hist, I_I_hist, r_E_hist, r_I_hist = hist
    bold = B_hist[burn_steps::steps_per_tr][:n_tr].T
    sl = slice(summary_start_steps, total_steps)
    S_E = jnp.mean(S_E_hist[sl], axis=0)
    S_I = jnp.mean(S_I_hist[sl], axis=0)
    I_E = jnp.mean(I_E_hist[sl], axis=0)
    I_I = jnp.mean(I_I_hist[sl], axis=0)
    r_E = jnp.mean(r_E_hist[sl], axis=0)
    r_I = jnp.mean(r_I_hist[sl], axis=0)
    summary = jnp.stack([S_E, S_I, I_E, I_I, r_E, r_I, S_E / (S_I + EPS), I_E / (I_I + EPS), r_E / (r_I + EPS)])
    return bold, summary


@jax.jit(static_argnames=("n_tr", "steps_per_tr", "burn_steps", "summary_start_steps", "total_steps", "target_index"))
def fic_grid_jax(sc, params, w_values, key, n_tr, steps_per_tr, burn_steps, summary_start_steps, total_steps, dt, target_value, target_index):
    n_roi = sc.shape[0]

    def eval_w(w):
        w_vec = jnp.full((n_roi,), w, dtype=jnp.float32)
        _, summary = simulate_dmf_jax(sc, params, w_vec, key, n_tr, steps_per_tr, burn_steps, summary_start_steps, total_steps, dt)
        return summary[target_index], summary[4]

    metrics, rates = jax.vmap(eval_w)(w_values)
    err = jnp.abs(metrics - target_value)
    idx = jnp.argmin(err, axis=0)
    final_w = w_values[idx]
    _, final_summary = simulate_dmf_jax(sc, params, final_w, key, n_tr, steps_per_tr, burn_steps, summary_start_steps, total_steps, dt)
    return final_w, final_summary, jnp.mean(err[idx, jnp.arange(n_roi)])


def simulate_eval(sc, fc_emp, params, mode, args, seed):
    steps_per_tr = max(1, int(round(args.tr / args.dt)))
    burn_steps = int(round(args.burn_s / args.dt))
    summary_start_steps = int(round(args.summary_start_s / args.dt))
    total_steps = burn_steps + args.n_tr * steps_per_tr
    sc_j = jnp.asarray(prepare_sc(sc), dtype=jnp.float32)
    params_j = jnp.asarray(params, dtype=jnp.float32)
    key = jax.random.PRNGKey(seed)
    target_index = 2 if args.fic_target == "I_E" else 4
    target_value = args.target_current if args.fic_target == "I_E" else args.target_rate

    if mode == "dmf_fic":
        w_values = jnp.linspace(args.w_ie_bounds[0], args.w_ie_bounds[1], args.fic_grid_size, dtype=jnp.float32)
        w_ie, _, fic_mae = fic_grid_jax(
            sc_j, params_j, w_values, key, args.fic_trs, steps_per_tr, burn_steps, summary_start_steps,
            burn_steps + args.fic_trs * steps_per_tr, args.dt, target_value, target_index
        )
    else:
        w_ie = jnp.full((sc.shape[0],), params[-3], dtype=jnp.float32)
        params_j = jnp.asarray([params[0], params[1], params[2], params[4], params[5]], dtype=jnp.float32)
        fic_mae = jnp.asarray(np.nan, dtype=jnp.float32)

    bold, summary = simulate_dmf_jax(sc_j, params_j, w_ie, key, args.n_tr, steps_per_tr, burn_steps, summary_start_steps, total_steps, args.dt)
    bold_np = np.asarray(jax.device_get(bold))
    summary_np = np.asarray(jax.device_get(summary))
    w_np = np.asarray(jax.device_get(w_ie))
    fc_r = corr_upper(fc_emp, fc_from_ts(bold_np))
    return fc_r, summary_np, w_np, float(jax.device_get(fic_mae))


def sample_params(rng, mode, n, args):
    if mode == "dmf_only":
        cols = [
            10 ** rng.uniform(np.log10(args.g_bounds[0]), np.log10(args.g_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ee_bounds[0]), np.log10(args.w_ee_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ei_bounds[0]), np.log10(args.w_ei_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ie_bounds[0]), np.log10(args.w_ie_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.sigma_bounds[0]), np.log10(args.sigma_bounds[1]), n),
            rng.uniform(args.i0_bounds[0], args.i0_bounds[1], n),
        ]
    else:
        cols = [
            10 ** rng.uniform(np.log10(args.g_bounds[0]), np.log10(args.g_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ee_bounds[0]), np.log10(args.w_ee_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ei_bounds[0]), np.log10(args.w_ei_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.sigma_bounds[0]), np.log10(args.sigma_bounds[1]), n),
            rng.uniform(args.i0_bounds[0], args.i0_bounds[1], n),
        ]
    return np.column_stack(cols)


def fit_mode_session(sc, fc_emp, mode, session, args, seed_offset):
    names, space = param_space(mode, args)
    eval_count = {"n": 0}

    def evaluate(params):
        i = eval_count["n"]
        eval_count["n"] += 1
        fc_r, summary, w_ie, fic_mae = simulate_eval(sc, fc_emp, params, mode, args, args.seed + seed_offset + i)
        loss = 1.0 - fc_r if np.isfinite(fc_r) else args.bad_loss
        print(f"{mode} {session} eval={i} loss={loss:.6f} fc_r={fc_r:.6f} params={dict(zip(names, params))}", flush=True)
        return loss, {"loss": loss, "fc_r": fc_r, "params": np.asarray(params, dtype=float), "summary": summary, "w_ie": w_ie, "fic_mae": fic_mae}

    if args.optimizer == "random":
        rng = np.random.default_rng(args.random_state + seed_offset)
        best = None
        for params in sample_params(rng, mode, args.n_candidates, args):
            _, rec = evaluate(params)
            if best is None or rec["loss"] < best["loss"]:
                best = rec
        return best

    best_seen = {"rec": None}

    def loss_fn(x):
        loss, rec = evaluate(x)
        if best_seen["rec"] is None or loss < best_seen["rec"]["loss"]:
            best_seen["rec"] = rec
        return loss

    optimizer = gp_minimize if args.optimizer == "gp" else forest_minimize
    res = optimizer(
        loss_fn,
        space,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        random_state=args.random_state + seed_offset,
    )
    fc_r, summary, w_ie, fic_mae = simulate_eval(sc, fc_emp, res.x, mode, args, args.seed + seed_offset + 100000)
    final_loss = 1.0 - fc_r if np.isfinite(fc_r) else args.bad_loss
    best = {
        "loss": float(final_loss),
        "fc_r": float(fc_r),
        "params": np.asarray(res.x, dtype=float),
        "summary": summary,
        "w_ie": w_ie,
        "fic_mae": fic_mae,
        "optimizer_fun": float(res.fun),
    }
    if best_seen["rec"] is not None and best_seen["rec"]["loss"] < best["loss"]:
        best["best_seen_loss"] = float(best_seen["rec"]["loss"])
    return best


def param_space(mode, args):
    if mode == "dmf_only":
        names = ("G", "w_EE", "w_EI", "w_IE", "sigma", "I0")
        space = [
            Real(args.g_bounds[0], args.g_bounds[1], prior="log-uniform", name="G"),
            Real(args.w_ee_bounds[0], args.w_ee_bounds[1], prior="log-uniform", name="w_EE"),
            Real(args.w_ei_bounds[0], args.w_ei_bounds[1], prior="log-uniform", name="w_EI"),
            Real(args.w_ie_bounds[0], args.w_ie_bounds[1], prior="log-uniform", name="w_IE"),
            Real(args.sigma_bounds[0], args.sigma_bounds[1], prior="log-uniform", name="sigma"),
            Real(args.i0_bounds[0], args.i0_bounds[1], name="I0"),
        ]
    else:
        names = ("G", "w_EE", "w_EI", "sigma", "I0")
        space = [
            Real(args.g_bounds[0], args.g_bounds[1], prior="log-uniform", name="G"),
            Real(args.w_ee_bounds[0], args.w_ee_bounds[1], prior="log-uniform", name="w_EE"),
            Real(args.w_ei_bounds[0], args.w_ei_bounds[1], prior="log-uniform", name="w_EI"),
            Real(args.sigma_bounds[0], args.sigma_bounds[1], prior="log-uniform", name="sigma"),
            Real(args.i0_bounds[0], args.i0_bounds[1], name="I0"),
        ]
    return names, space


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--tag", default="20260701_jax_gpu")
    p.add_argument("--optimizer", default="forest", choices=("random", "forest", "gp"))
    p.add_argument("--n-calls", type=int, default=90)
    p.add_argument("--n-initial-points", type=int, default=14)
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--n-tr", type=int, default=80)
    p.add_argument("--fic-trs", type=int, default=40)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tr", type=float, default=0.72)
    p.add_argument("--burn-s", type=float, default=10.0)
    p.add_argument("--summary-start-s", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--fic-target", choices=("I_E", "r_E"), default="I_E")
    p.add_argument("--target-current", type=float, default=0.3773805650)
    p.add_argument("--target-rate", type=float, default=3.0)
    p.add_argument("--fic-grid-size", type=int, default=31)
    p.add_argument("--g-bounds", type=float, nargs=2, default=[0.05, 5.0])
    p.add_argument("--w-ee-bounds", type=float, nargs=2, default=[0.5, 2.5])
    p.add_argument("--w-ei-bounds", type=float, nargs=2, default=[0.5, 2.5])
    p.add_argument("--w-ie-bounds", type=float, nargs=2, default=[0.2, 10.0])
    p.add_argument("--sigma-bounds", type=float, nargs=2, default=[0.0001, 0.01])
    p.add_argument("--i0-bounds", type=float, nargs=2, default=[0.32, 0.44])
    p.add_argument("--bad-loss", type=float, default=3.0)
    p.add_argument("--subject-sides", nargs="+", default=["R", "R", "L", "R", "L", "L", "R"])
    return p.parse_args()


def main():
    args = parse_args()
    prefix = ROOT / f"dmf_rww_ei_jax_gpu_fc_only_{args.tag}"
    out_csv = no_overwrite_path(prefix.with_suffix(".csv"))
    out_json = no_overwrite_path(prefix.with_suffix(".json"))
    out_md = no_overwrite_path(prefix.with_suffix(".md"))
    print(f"JAX devices: {jax.devices()}", flush=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, got {jax.default_backend()}")

    data = np.load(args.data, allow_pickle=True)
    sc = np.asarray(data["SC"], dtype=float)
    fc = np.asarray(data["FC"], dtype=float)
    labels = np.asarray(data["roi_labels"], dtype=object)
    left_motor, right_motor = default_motor_rois(labels)
    motor = sorted(set(left_motor + right_motor))

    rows = []
    args_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    payload = {"data": str(args.data), "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()], "args": args_payload, "results": {}}
    names = ["S_E", "S_I", "I_E", "I_I", "r_E", "r_I", "S_E_over_S_I", "I_E_over_I_I", "r_E_over_r_I"]
    for mode in ("dmf_only", "dmf_fic"):
        payload["results"][mode] = {}
        for sess_i, session in enumerate(SESSIONS):
            sc_group = sanitize_matrix(finite_nanmean(sc[:, sess_i], axis=0))
            fc_emp = sanitize_matrix(finite_nanmean(fc[:, sess_i], axis=0))
            best = fit_mode_session(sc_group, fc_emp, mode, session, args, 100 * sess_i + (1000 if mode == "dmf_fic" else 0))
            summary = {name: best["summary"][i] for i, name in enumerate(names)}
            row = {
                "mode": mode,
                "session": session,
                "loss": best["loss"],
                "fc_r": best["fc_r"],
                "global_I_E": float(np.nanmean(summary["I_E"])),
                "motor_I_E": float(np.nanmean(summary["I_E"][motor])),
                "global_r_E": float(np.nanmean(summary["r_E"])),
                "motor_r_E": float(np.nanmean(summary["r_E"][motor])),
                "global_S_E_over_S_I": float(np.nanmean(summary["S_E_over_S_I"])),
                "motor_S_E_over_S_I": float(np.nanmean(summary["S_E_over_S_I"][motor])),
                "global_I_E_over_I_I": float(np.nanmean(summary["I_E_over_I_I"])),
                "motor_I_E_over_I_I": float(np.nanmean(summary["I_E_over_I_I"][motor])),
                "mean_w_IE": float(np.nanmean(best["w_ie"])),
                "motor_w_IE": float(np.nanmean(best["w_ie"][motor])),
                "fic_mae": best["fic_mae"],
            }
            for key in ("I_E", "r_E", "S_E_over_S_I", "I_E_over_I_I"):
                left, right, contra, ipsi = weighted_side_values(summary[key], left_motor, right_motor, args.subject_sides)
                row[f"contra_motor_{key}"] = contra
                row[f"ipsi_motor_{key}"] = ipsi
            rows.append(row)
            payload["results"][mode][session] = {"row": row, "params": best["params"].tolist()}

    with out_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# JAX/GPU two-population DMF/rWW FC-only pilot",
        "",
        f"Backend: `{jax.default_backend()}`; devices: `{', '.join(map(str, jax.devices()))}`",
        f"Optimizer: `{args.optimizer}`; calls={args.n_calls if args.optimizer != 'random' else args.n_candidates}; initial={args.n_initial_points if args.optimizer != 'random' else 'NA'}; n_tr={args.n_tr}; dt={args.dt}; burn={args.burn_s}s.",
        "",
        "| mode | session | FC r | motor I_E | motor r_E | motor S_E/S_I | motor I_E/I_I | motor w_IE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {session} | {fc_r:.3f} | {motor_I_E:.4f} | {motor_r_E:.3f} | "
            "{motor_S_E_over_S_I:.3f} | {motor_I_E_over_I_I:.3f} | {motor_w_IE:.3f} |".format(**row)
        )
    lines += ["", f"- `{out_csv.name}`", f"- `{out_json.name}`", f"- `{out_md.name}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
