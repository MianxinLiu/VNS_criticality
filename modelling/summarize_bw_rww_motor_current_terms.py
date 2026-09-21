#!/usr/bin/env python3
"""Decompose BW-rWW motor E-population input current terms."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "project"))

import dmf_rww_ei_jax_gpu_fc_only as base
import model_signal_fei_compare_bw_rww as signal_model
from dmf_rww_ei_jax_gpu_bw_fc_only import h_transfer


EPS = 1e-12


@jax.jit(static_argnames=("summary_start_steps", "total_steps"))
def simulate_current_terms_jax(sc, params, key, summary_start_steps, total_steps, dt, fixed_sigma, fixed_i0):
    """Return means using the same pre-update state that defines I_E."""
    G, w_EE, w_EI, w_IE = params
    n_roi = sc.shape[0]

    j_nmda = 0.15
    a_e, b_e, d_e = 310.0, 125.0, 0.16
    a_i, b_i, d_i = 615.0, 177.0, 0.087
    gamma_e, gamma_i = 0.641, 1.0
    tau_e, tau_i = 0.1, 0.01
    w_e, w_i, w_ii = 1.0, 0.7, 1.0

    k_e, k_i = jax.random.split(key)
    noise_e = jax.random.normal(k_e, (total_steps, n_roi), dtype=jnp.float32)
    noise_i = jax.random.normal(k_i, (total_steps, n_roi), dtype=jnp.float32)

    def step(carry, xs):
        s_e, s_i = carry
        ne, ni = xs
        coupling = sc @ s_e
        background = jnp.full_like(s_e, w_e * fixed_i0)
        local_e = w_EE * j_nmda * s_e
        network_e = G * j_nmda * coupling
        local_inhibition = -w_IE * s_i
        i_e = background + local_e + network_e + local_inhibition
        i_i = w_i * fixed_i0 + w_EI * j_nmda * s_e - w_ii * s_i
        r_e = h_transfer(i_e, a_e, b_e, d_e)
        r_i = h_transfer(i_i, a_i, b_i, d_i)

        s_e_next = s_e + (-s_e / tau_e + (1.0 - s_e) * gamma_e * r_e) * dt + fixed_sigma * jnp.sqrt(dt) * ne
        s_i_next = s_i + (-s_i / tau_i + gamma_i * r_i) * dt + fixed_sigma * jnp.sqrt(dt) * ni
        s_e_next = jnp.clip(s_e_next, 0.0, 1.0)
        s_i_next = jnp.clip(s_i_next, 0.0, 1.0)

        out = (s_e_next, s_i_next, s_e, s_i, i_e, background, local_e, network_e, local_inhibition)
        return (s_e_next, s_i_next), out

    init = (
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
        jnp.full((n_roi,), 0.001, dtype=jnp.float32),
    )
    _, hist = jax.lax.scan(step, init, (noise_e, noise_i), length=total_steps)
    s_e_hist, s_i_hist, s_e_pre_hist, s_i_pre_hist, i_e_hist, background_hist, local_e_hist, network_e_hist, local_inhibition_hist = hist
    sl = slice(summary_start_steps, total_steps)
    s_e_mean = jnp.mean(s_e_hist[sl], axis=0)
    s_i_mean = jnp.mean(s_i_hist[sl], axis=0)
    state_summary = jnp.stack(
        [
            s_e_mean,
            s_i_mean,
            jnp.mean(i_e_hist[sl], axis=0),
            s_e_mean / (s_i_mean + EPS),
            jnp.mean(s_e_pre_hist[sl], axis=0),
            jnp.mean(s_i_pre_hist[sl], axis=0),
        ]
    )
    term_summary = jnp.stack(
        [
            jnp.mean(background_hist[sl], axis=0),
            jnp.mean(local_e_hist[sl], axis=0),
            jnp.mean(network_e_hist[sl], axis=0),
            jnp.mean(local_inhibition_hist[sl], axis=0),
        ]
    )
    return state_summary, term_summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "VNS_active_only_exclude_07_09.npz")
    p.add_argument("--fit-json", type=Path, default=ROOT / "dmf_rww_ei_jax_gpu_bw_fc_only_20260824_bw_forest90_fixed_sigma_i0.json")
    p.add_argument("--output", type=Path, default=ROOT / "bw_rww_motor_current_terms_20260904.csv")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    fit = json.loads(args.fit_json.read_text())
    data = np.load(args.data, allow_pickle=True)
    labels = np.asarray(data["roi_labels"], dtype=object)
    left_motor, right_motor = base.default_motor_rois(labels)
    motor = np.asarray(left_motor + right_motor, dtype=int)
    motor_set = set(motor.tolist())
    non_motor = np.asarray([i for i in range(len(labels)) if i not in motor_set], dtype=int)

    sim_args = signal_model.parse_args()
    sim_args.data = args.data
    sim_args.fit_json = args.fit_json
    sim_args.fixed_sigma = float(fit["fixed"]["sigma"])
    sim_args.fixed_i0 = float(fit["fixed"]["I0"])
    sim_args.bold_input = str(fit["bold_input"])
    sim_args.seed = args.seed

    sc_all = np.asarray(data["SC"], dtype=float)
    rows = []
    for sess_i, session in enumerate(signal_model.SESSIONS):
        sc = base.sanitize_matrix(base.finite_nanmean(sc_all[:, sess_i], axis=0))
        params = np.asarray(fit["results"][session]["params"], dtype=float)
        g, w_ee, w_ei, w_ie = params
        steps_per_tr = max(1, int(round(sim_args.tr_s / sim_args.rww_dt_s)))
        burn_steps = int(round(sim_args.burn_s / sim_args.rww_dt_s))
        total_steps = burn_steps + sim_args.n_tr * steps_per_tr
        sc_prepared = base.prepare_sc(sc)
        state_summary, term_summary = simulate_current_terms_jax(
            jnp.asarray(sc_prepared, dtype=jnp.float32),
            jnp.asarray(params, dtype=jnp.float32),
            jax.random.PRNGKey(args.seed + sess_i),
            burn_steps,
            total_steps,
            sim_args.rww_dt_s,
            sim_args.fixed_sigma,
            sim_args.fixed_i0,
        )
        state_summary = np.asarray(jax.device_get(state_summary), dtype=float)
        term_summary = np.asarray(jax.device_get(term_summary), dtype=float)
        state_by_name = {
            "S_E": state_summary[0],
            "S_I": state_summary[1],
            "I_E": state_summary[2],
            "S_E_over_S_I": state_summary[3],
            "S_E_pre": state_summary[4],
            "S_I_pre": state_summary[5],
        }

        current_terms = {
            "background": term_summary[0],
            "local_E": term_summary[1],
            "network_E": term_summary[2],
            "network_E_from_motor": g * 0.15 * (sc_prepared[:, motor] @ state_by_name["S_E_pre"][motor]),
            "network_E_from_nonmotor": g * 0.15 * (sc_prepared[:, non_motor] @ state_by_name["S_E_pre"][non_motor]),
            "local_inhibition": term_summary[3],
        }
        row = {
            "session": session,
            "G": g,
            "w_EE": w_ee,
            "w_EI": w_ei,
            "w_IE": w_ie,
            "motor_S_E": float(np.nanmean(state_by_name["S_E"][motor])),
            "motor_S_I": float(np.nanmean(state_by_name["S_I"][motor])),
            "motor_E_to_I_drive": float(np.nanmean(w_ei * 0.15 * state_by_name["S_E"][motor])),
            "motor_I_E": float(np.nanmean(state_by_name["I_E"][motor])),
            "motor_S_E_over_S_I": float(np.nanmean(state_by_name["S_E_over_S_I"][motor])),
        }
        for name, values in current_terms.items():
            row[name] = float(np.nanmean(values[motor]))
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
