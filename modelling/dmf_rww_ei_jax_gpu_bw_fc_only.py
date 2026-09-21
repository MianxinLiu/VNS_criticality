#!/usr/bin/env python3
"""JAX/GPU rWW-only FC fit with Balloon-Windkessel BOLD.

This is a stricter rWW-only rerun of dmf_rww_ei_jax_gpu_fc_only.py:

- observable: Balloon-Windkessel BOLD, not the previous low-pass S_E proxy
- fitted parameters: G, w_EE, w_EI, w_IE
- fixed parameters: sigma, I0
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

import dmf_rww_ei_jax_gpu_fc_only as base


ROOT = Path(__file__).resolve().parent
DATA_DEFAULT = ROOT / "VNS_active_only_exclude_07_09.npz"
SESSIONS = ("baseline", "followup6w")
EPS = 1e-12


@jax.jit
def h_transfer(x, a, b, d):
    y = a * x - b
    den = 1.0 - jnp.exp(-d * y)
    out = jnp.where(jnp.abs(den) < 1e-8, 1.0 / d, y / den)
    return jnp.clip(out, 0.0, 500.0)


@jax.jit(static_argnames=("n_tr", "steps_per_tr", "burn_steps", "summary_start_steps", "total_steps", "bold_input"))
def simulate_dmf_bw_jax(
    sc,
    params,
    key,
    n_tr,
    steps_per_tr,
    burn_steps,
    summary_start_steps,
    total_steps,
    dt,
    fixed_sigma,
    fixed_i0,
    bold_input,
):
    G, w_EE, w_EI, w_IE = params
    n_roi = sc.shape[0]

    J_NMDA = 0.15
    a_E, b_E, d_E = 310.0, 125.0, 0.16
    a_I, b_I, d_I = 615.0, 177.0, 0.087
    gamma_E, gamma_I = 0.641, 1.0
    tau_E, tau_I = 0.1, 0.01
    w_E, w_I, w_II = 1.0, 0.7, 1.0

    # Same Balloon-Windkessel constants as bold_simu.py.
    kappa, gamma, tau = 0.65, 0.41, 0.98
    alpha, E0, V0 = 0.32, 0.34, 0.02
    k1, k2, k3 = 7.0, 2.0, 2.0

    k_e, k_i = jax.random.split(key)
    noise_e = jax.random.normal(k_e, (total_steps, n_roi), dtype=jnp.float32)
    noise_i = jax.random.normal(k_i, (total_steps, n_roi), dtype=jnp.float32)

    def step(carry, xs):
        S_E, S_I, s, f, v, q = carry
        ne, ni = xs
        coupling = sc @ S_E
        I_E = w_E * fixed_i0 + w_EE * J_NMDA * S_E + G * J_NMDA * coupling - w_IE * S_I
        I_I = w_I * fixed_i0 + w_EI * J_NMDA * S_E - w_II * S_I
        r_E = h_transfer(I_E, a_E, b_E, d_E)
        r_I = h_transfer(I_I, a_I, b_I, d_I)

        S_E = S_E + (-S_E / tau_E + (1.0 - S_E) * gamma_E * r_E) * dt + fixed_sigma * jnp.sqrt(dt) * ne
        S_I = S_I + (-S_I / tau_I + gamma_I * r_I) * dt + fixed_sigma * jnp.sqrt(dt) * ni
        S_E = jnp.clip(S_E, 0.0, 1.0)
        S_I = jnp.clip(S_I, 0.0, 1.0)

        neural = jnp.where(bold_input == 0, S_E, r_E / 100.0)
        f_safe = jnp.maximum(f, 1e-6)
        v_safe = jnp.maximum(v, 1e-6)
        extraction = f_safe * (1.0 - (1.0 - E0) ** (1.0 / f_safe)) / E0
        ds = neural - kappa * s - gamma * (f_safe - 1.0)
        df = s
        dv = (f_safe - v_safe ** (1.0 / alpha)) / tau
        dq = (extraction - q / v_safe ** (1.0 - 1.0 / alpha)) / tau
        s = s + ds * dt
        f = jnp.maximum(f + df * dt, 1e-6)
        v = jnp.maximum(v + dv * dt, 1e-6)
        q = jnp.maximum(q + dq * dt, 1e-6)
        bold = V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))

        out = (bold, S_E, S_I, I_E, I_I, r_E, r_I)
        return (S_E, S_I, s, f, v, q), out

    init = (
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
        jnp.zeros((n_roi,), dtype=jnp.float32),
        jnp.ones((n_roi,), dtype=jnp.float32),
        jnp.ones((n_roi,), dtype=jnp.float32),
        jnp.ones((n_roi,), dtype=jnp.float32),
    )
    _, hist = jax.lax.scan(step, init, (noise_e, noise_i), length=total_steps)
    bold_hist, S_E_hist, S_I_hist, I_E_hist, I_I_hist, r_E_hist, r_I_hist = hist
    bold = bold_hist[burn_steps::steps_per_tr][:n_tr].T
    sl = slice(summary_start_steps, total_steps)
    S_E = jnp.mean(S_E_hist[sl], axis=0)
    S_I = jnp.mean(S_I_hist[sl], axis=0)
    I_E = jnp.mean(I_E_hist[sl], axis=0)
    I_I = jnp.mean(I_I_hist[sl], axis=0)
    r_E = jnp.mean(r_E_hist[sl], axis=0)
    r_I = jnp.mean(r_I_hist[sl], axis=0)
    summary = jnp.stack([S_E, S_I, I_E, I_I, r_E, r_I, S_E / (S_I + EPS), I_E / (I_I + EPS), r_E / (r_I + EPS)])
    return bold, summary


def simulate_eval(sc, fc_emp, params, args, seed):
    steps_per_tr = max(1, int(round(args.tr / args.dt)))
    burn_steps = int(round(args.burn_s / args.dt))
    summary_start_steps = int(round(args.summary_start_s / args.dt))
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
        args.dt,
        args.fixed_sigma,
        args.fixed_i0,
        bold_input,
    )
    bold_np = np.asarray(jax.device_get(bold))
    summary_np = np.asarray(jax.device_get(summary))
    fc_r = base.corr_upper(fc_emp, base.fc_from_ts(bold_np))
    return fc_r, summary_np


def sample_params(rng, n, args):
    return np.column_stack(
        [
            10 ** rng.uniform(np.log10(args.g_bounds[0]), np.log10(args.g_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ee_bounds[0]), np.log10(args.w_ee_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ei_bounds[0]), np.log10(args.w_ei_bounds[1]), n),
            10 ** rng.uniform(np.log10(args.w_ie_bounds[0]), np.log10(args.w_ie_bounds[1]), n),
        ]
    )


def param_space(args):
    names = ("G", "w_EE", "w_EI", "w_IE")
    space = [
        Real(args.g_bounds[0], args.g_bounds[1], prior="log-uniform", name="G"),
        Real(args.w_ee_bounds[0], args.w_ee_bounds[1], prior="log-uniform", name="w_EE"),
        Real(args.w_ei_bounds[0], args.w_ei_bounds[1], prior="log-uniform", name="w_EI"),
        Real(args.w_ie_bounds[0], args.w_ie_bounds[1], prior="log-uniform", name="w_IE"),
    ]
    return names, space


def fit_session(sc, fc_emp, session, args, seed_offset):
    names, space = param_space(args)
    eval_count = {"n": 0}

    def evaluate(params):
        i = eval_count["n"]
        eval_count["n"] += 1
        fc_r, summary = simulate_eval(sc, fc_emp, params, args, args.seed + seed_offset + i)
        loss = 1.0 - fc_r if np.isfinite(fc_r) else args.bad_loss
        print(f"rww_bw {session} eval={i} loss={loss:.6f} fc_r={fc_r:.6f} params={dict(zip(names, params))}", flush=True)
        return loss, {"loss": loss, "fc_r": fc_r, "params": np.asarray(params, dtype=float), "summary": summary}

    if args.optimizer == "random":
        rng = np.random.default_rng(args.random_state + seed_offset)
        best = None
        for params in sample_params(rng, args.n_candidates, args):
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
    fc_r, summary = simulate_eval(sc, fc_emp, res.x, args, args.seed + seed_offset + 100000)
    final_loss = 1.0 - fc_r if np.isfinite(fc_r) else args.bad_loss
    best = {
        "loss": float(final_loss),
        "fc_r": float(fc_r),
        "params": np.asarray(res.x, dtype=float),
        "summary": summary,
        "optimizer_fun": float(res.fun),
    }
    if best_seen["rec"] is not None and best_seen["rec"]["loss"] < best["loss"]:
        best["best_seen_loss"] = float(best_seen["rec"]["loss"])
        best["best_seen_fc_r"] = float(best_seen["rec"]["fc_r"])
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_DEFAULT)
    p.add_argument("--tag", default="20260824_bw")
    p.add_argument("--optimizer", default="forest", choices=("random", "forest", "gp"))
    p.add_argument("--n-calls", type=int, default=90)
    p.add_argument("--n-initial-points", type=int, default=14)
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--n-tr", type=int, default=160)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tr", type=float, default=0.72)
    p.add_argument("--burn-s", type=float, default=30.0)
    p.add_argument("--summary-start-s", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--fixed-sigma", type=float, default=0.001)
    p.add_argument("--fixed-i0", type=float, default=0.3773805650)
    p.add_argument("--bold-input", choices=("S_E", "r_E"), default="S_E")
    p.add_argument("--g-bounds", type=float, nargs=2, default=[0.05, 5.0])
    p.add_argument("--w-ee-bounds", type=float, nargs=2, default=[0.5, 2.5])
    p.add_argument("--w-ei-bounds", type=float, nargs=2, default=[0.5, 2.5])
    p.add_argument("--w-ie-bounds", type=float, nargs=2, default=[0.2, 10.0])
    p.add_argument("--bad-loss", type=float, default=3.0)
    p.add_argument("--subject-sides", nargs="+", default=["R", "R", "L", "R", "L", "L", "R"])
    return p.parse_args()


def main():
    args = parse_args()
    prefix = ROOT / f"dmf_rww_ei_jax_gpu_bw_fc_only_{args.tag}"
    out_csv = base.no_overwrite_path(prefix.with_suffix(".csv"))
    out_json = base.no_overwrite_path(prefix.with_suffix(".json"))
    out_md = base.no_overwrite_path(prefix.with_suffix(".md"))
    print(f"JAX devices: {jax.devices()}", flush=True)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected JAX GPU backend, got {jax.default_backend()}")

    data = np.load(args.data, allow_pickle=True)
    sc = np.asarray(data["SC"], dtype=float)
    fc = np.asarray(data["FC"], dtype=float)
    labels = np.asarray(data["roi_labels"], dtype=object)
    left_motor, right_motor = base.default_motor_rois(labels)
    motor = sorted(set(left_motor + right_motor))

    rows = []
    args_payload = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    payload = {
        "data": str(args.data),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "param_names": ["G", "w_EE", "w_EI", "w_IE"],
        "fixed": {"sigma": args.fixed_sigma, "I0": args.fixed_i0},
        "observable": "Balloon-Windkessel BOLD",
        "bold_input": args.bold_input,
        "args": args_payload,
        "results": {},
    }
    names = ["S_E", "S_I", "I_E", "I_I", "r_E", "r_I", "S_E_over_S_I", "I_E_over_I_I", "r_E_over_r_I"]
    for sess_i, session in enumerate(SESSIONS):
        sc_group = base.sanitize_matrix(base.finite_nanmean(sc[:, sess_i], axis=0))
        fc_emp = base.sanitize_matrix(base.finite_nanmean(fc[:, sess_i], axis=0))
        best = fit_session(sc_group, fc_emp, session, args, 100 * sess_i)
        summary = {name: best["summary"][i] for i, name in enumerate(names)}
        row = {
            "mode": "rww_bw_only",
            "session": session,
            "loss": best["loss"],
            "fc_r": best["fc_r"],
            "G": float(best["params"][0]),
            "w_EE": float(best["params"][1]),
            "w_EI": float(best["params"][2]),
            "w_IE": float(best["params"][3]),
            "fixed_sigma": args.fixed_sigma,
            "fixed_I0": args.fixed_i0,
            "global_I_E": float(np.nanmean(summary["I_E"])),
            "motor_I_E": float(np.nanmean(summary["I_E"][motor])),
            "global_r_E": float(np.nanmean(summary["r_E"])),
            "motor_r_E": float(np.nanmean(summary["r_E"][motor])),
            "global_S_E_over_S_I": float(np.nanmean(summary["S_E_over_S_I"])),
            "motor_S_E_over_S_I": float(np.nanmean(summary["S_E_over_S_I"][motor])),
            "global_I_E_over_I_I": float(np.nanmean(summary["I_E_over_I_I"])),
            "motor_I_E_over_I_I": float(np.nanmean(summary["I_E_over_I_I"][motor])),
        }
        for key in ("I_E", "r_E", "S_E_over_S_I", "I_E_over_I_I"):
            left, right, contra, ipsi = base.weighted_side_values(summary[key], left_motor, right_motor, args.subject_sides)
            row[f"contra_motor_{key}"] = contra
            row[f"ipsi_motor_{key}"] = ipsi
        rows.append(row)
        payload["results"][session] = {"row": row, "params": best["params"].tolist()}

    with out_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# JAX/GPU rWW-only FC fit with Balloon-Windkessel BOLD",
        "",
        f"Data: `{args.data}`",
        f"Optimizer: `{args.optimizer}`; calls={args.n_calls if args.optimizer != 'random' else args.n_candidates}; initial={args.n_initial_points if args.optimizer != 'random' else 'NA'}; n_tr={args.n_tr}; dt={args.dt}; burn={args.burn_s}s.",
        f"Free parameters: `G`, `w_EE`, `w_EI`, `w_IE`; fixed sigma={args.fixed_sigma}, I0={args.fixed_i0}.",
        f"Observable: Balloon-Windkessel BOLD driven by `{args.bold_input}`.",
        "",
        "| session | FC r | motor I_E | motor r_E | motor S_E/S_I | motor I_E/I_I | G | w_EE | w_EI | w_IE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {session} | {fc_r:.3f} | {motor_I_E:.4f} | {motor_r_E:.3f} | "
            "{motor_S_E_over_S_I:.3f} | {motor_I_E_over_I_I:.3f} | {G:.3f} | {w_EE:.3f} | {w_EI:.3f} | {w_IE:.3f} |".format(**row)
        )
    lines += ["", f"- `{out_csv.name}`", f"- `{out_json.name}`", f"- `{out_md.name}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
