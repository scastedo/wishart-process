from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import optim
from tqdm import tqdm
from scipy.io import loadmat

import inference
import models
import utils  # <- do NOT modify this; we only use it

# -----------------------------------------------------------------------------
# A. Data reshape (same logic as your working script)
# -----------------------------------------------------------------------------

def resort_preprocessing(datum, angle_arr, sf_arr, animal):
    data = np.copy(datum[animal, :])
    neurons = data[0].shape[0]
    reshape_data = np.full((60, neurons, data[0].shape[1]), np.nan)
    for i in range(60):
        reshape_data[i, :, :] = data[i]

    reshape_data = reshape_data.reshape(60, neurons, 12, 120)
    reshape_data = np.transpose(reshape_data, (1, 2, 0, 3))
    # Remove first two neurons
    reshape_data = reshape_data[2:, :, :, :]

    # Remove None trials
    max_trial = np.argmax(np.isnan(reshape_data[0, 1, :, 0]))
    reshape_data = reshape_data[:, :, :max_trial, :]

    # Reorder angles
    angles = np.copy(angle_arr[animal])
    for itrials in range(angles.shape[1]):
        order = angles[:, itrials] - 1
        reshape_data[:, :, itrials, :] = reshape_data[:, order, itrials, :]

    # Reorder SFs → stack by SF=1..5, padding with NaNs to equalize trial counts
    reshaped_data = []
    sfs = np.copy(sf_arr[animal])
    for experiment in range(1, 6):
        mask = sfs == experiment
        reshaped_data.append(reshape_data[:, :, mask, :])

    max_trials = max([exp.shape[2] for exp in reshaped_data])
    for i in range(len(reshaped_data)):
        if reshaped_data[i].shape[2] < max_trials:
            padding = max_trials - reshaped_data[i].shape[2]
            reshaped_data[i] = np.pad(
                reshaped_data[i],
                ((0, 0), (0, 0), (0, padding), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

    reshaped_data = np.stack(reshaped_data, axis=2)  # (N, Ca, Cs=5, K, T)
    return reshaped_data


# -----------------------------------------------------------------------------
# B. Small helpers (robust y access; no change to split_data)
# -----------------------------------------------------------------------------

def _as_array(x):
    try:
        return jax.device_get(x)
    except Exception:
        return x

def get_design(obj):
    if isinstance(obj, dict) and "x" in obj:
        return obj["x"]
    return obj

def get_observed_y(y_obj):
    """
    Accept either:
      - dict with key 'x_test' (your original working code), or
      - dict with key 'y'
      - or a bare array
    """
    if isinstance(y_obj, dict):
        if "x_test" in y_obj:
            return y_obj["x_test"]
        if "y" in y_obj:
            return y_obj["y"]
        # Fallthrough: if other keys are present, try the first ndarray
        for v in y_obj.values():
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                return v
        raise KeyError("Could not find observed test responses in y_te.")
    return y_obj


# -----------------------------------------------------------------------------
# C. Hyperparam sampling (log-uniform like your working version + 2D extras)
# -----------------------------------------------------------------------------

def sample_hyperparams(key: jax.Array, two_d: bool) -> Dict[str, float]:
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    def _logu(k, low, high):
        return jnp.exp(jax.random.uniform(k, (), minval=jnp.log(low), maxval=jnp.log(high)))

    hp = {
        # GP over angles
        "l_gp_a": _logu(k1, 0.2, 3.0),
        "g_gp_a": _logu(k2, 1e-5, 1e-3),
        "b_gp_a": _logu(k3, 0.05, 2.0),
        # WP over angles
        "l_wp_a": _logu(k4, 0.2, 3.0),
        "g_wp_a": _logu(k5, 1e-5, 1e-3),
        "b_wp_a": _logu(k6, 0.05, 2.0),
        # Wishart rank control
        "p": int(jax.random.randint(k7, (), 0, 6)),
    }
    if two_d:
        k8, k9, k10, k11, k12, k13 = jax.random.split(k6, 6)
        hp.update(
            {
                # GP over SF
                "l_gp_s": _logu(k8, 0.1, 5.0),
                "g_gp_s": _logu(k9, 1e-6, 1e-3),
                "b_gp_s": _logu(k10, 0.05, 2.0),
                # WP over SF
                "l_wp_s": _logu(k11, 0.2, 5.0),
                "g_wp_s": _logu(k12, 1e-6, 1e-3),
                "b_wp_s": _logu(k13, 0.05, 4.0),
            }
        )
    return {k: float(v) for k, v in hp.items()}


# -----------------------------------------------------------------------------
# D. Evaluator (keeps your model + scoring; safer numerics; V_scale·I)
# -----------------------------------------------------------------------------

def make_evaluator(
    N: int,
    period: int,
    two_d: bool,
    x_tr,
    y_tr,
    x_te,
    y_te,
    *,
    n_vi_steps: int = 1500,
    n_mc: int = 8,
    V_scale: float = 1e-2,
):

    # Normalize shapes to JAX arrays
    x_tr = jnp.asarray(get_design(x_tr))
    y_tr = jnp.asarray(_as_array(y_tr))
    x_te = jnp.asarray(get_design(x_te))
    y_te = y_te  # kept as is (dict or array) because models.lik expects same structure later

    # Likelihood
    lik = models.NormalConditionalLikelihood(N)

    # Kernel builders (matching your working definition)
    def build_kernels(hyper: Dict[str, float]):
        if two_d:
            per_gp = lambda a, b: hyper["g_gp_a"] * (a == b) + hyper["b_gp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_gp_a"]
            )
            per_wp = lambda a, b: hyper["g_wp_a"] * (a == b) + hyper["b_wp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_wp_a"]
            )
            sq_gp = lambda a, b: hyper["g_gp_s"] * (a == b) + hyper["b_gp_s"] * jnp.exp(
                -(a - b) ** 2 / hyper["l_gp_s"]
            )
            sq_wp = lambda a, b: hyper["g_wp_s"] * (a == b) + hyper["b_wp_s"] * jnp.exp(
                -(a - b) ** 2 / hyper["l_wp_s"]
            )
            K_gp = lambda x, y: per_gp(x[0], y[0]) * sq_gp(x[1], y[1])
            K_wp = lambda x, y: per_wp(x[0], y[0]) * sq_wp(x[1], y[1])
        else:
            K_gp = lambda a, b: hyper["g_gp_a"] * (a == b) + hyper["b_gp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_gp_a"]
            )
            K_wp = lambda a, b: hyper["g_wp_a"] * (a == b) + hyper["b_wp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_wp_a"]
            )
        return K_gp, K_wp

    def _evaluate(key, hyper: Dict[str, float]) -> float:
        # Model (same as your working code)
        K_gp, K_wp = build_kernels(hyper)
        gp = models.GaussianProcess(kernel=K_gp, N=N)
        P = max(int(hyper["p"]), 0)
        wp = models.WishartLRDProcess(
            kernel=K_wp,
            P=P,
            V=V_scale * jnp.eye(N),
            optimize_L=False,  # keep behaviour identical to your working script
        )
        joint = models.JointGaussianWishartProcess(gp, wp, lik)

        guide = inference.VariationalNormal(joint.model)
        optimizer = optim.Adam(1e-1)
        svi_key = jax.random.split(key, 1)[0]

        try:
            guide.infer(optimizer, x_tr, y_tr, n_vi_steps, key=svi_key)
            joint.update_params(guide.posterior)
            posterior = models.NormalGaussianWishartPosterior(joint, guide, x_tr)
        except ValueError:
            return -jnp.inf  # cholesky/scale_tril failure → reject

        # Monte-Carlo score on test
        y_obs = get_observed_y(y_te)  # supports dict["x_test"] or ["y"] or ndarray

        def _one_draw(rng):
            with numpyro.handlers.seed(rng_seed=rng):
                mu, Sigma, _ = posterior.sample(x_te)
            lp = lik.log_prob(y_obs, mu, Sigma).mean()
            return jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

        mc_keys = jax.random.split(key, max(1, n_mc))
        score = jax.vmap(_one_draw)(mc_keys).mean()
        return float(score)

    return _evaluate


# -----------------------------------------------------------------------------
# E. Random search driver (unchanged spirit, cleaner prints)
# -----------------------------------------------------------------------------

def random_search(
    eval_fn,
    two_d: bool,
    *,
    n_draws: int = 200,
    seed: int = 0,
):
    best_hp, best_score = None, -jnp.inf
    scores: List[float] = []
    combos: List[Dict[str, float]] = []

    key = jax.random.PRNGKey(seed)
    for i in tqdm(range(n_draws), desc="random search"):
        key, sub = jax.random.split(key)
        hp = sample_hyperparams(sub, two_d)
        score = float(eval_fn(sub, hp))
        scores.append(score)
        combos.append(hp)
        if score > best_score:
            best_score, best_hp = score, hp
            tqdm.write(f"▲ {i:3d}: {best_score:.3f}  {best_hp}")

    # Fallback if everything went -inf
    if best_hp is None:
        j = int(np.nanargmax(np.asarray(scores)))
        best_hp, best_score = combos[j], scores[j]

    return best_hp, best_score, np.asarray(scores), combos


# -----------------------------------------------------------------------------
# F. Build (x, y) per animal — mirrors your working shapes precisely
# -----------------------------------------------------------------------------

def build_xy_for_animal(DATA, ANG, SF, animal: int, two_d: bool, t0=40, t1=80):
    """
    Returns:
      x_full: (C,1) for 1D (flattened later), or (C,2) for 2D
      y_full: (K, C, N)   (K=trials, C=conditions, N=neurons)
      N:      int
      period: int  (angle period = number of angles)
    """
    TEST_DATA = resort_preprocessing(DATA, ANG, SF, animal)[:, :, :, :, t0:t1]
    RESP = jnp.nanmean(TEST_DATA, axis=-1)  # (N, Ca, Cs, K)
    N, Ca, Cs, K = RESP.shape

    if two_d:
        good_trials = ~jnp.isnan(RESP).all(axis=(0, 1, 2))  # (K,)
        RESP = RESP[..., good_trials]
        N, Ca, Cs, K = RESP.shape
        x_full = jnp.stack(
            jnp.meshgrid(jnp.arange(Ca), jnp.arange(Cs), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)  # (Ca*Cs, 2)
        y_full = jnp.transpose(RESP.reshape(N, Ca * Cs, K), (2, 1, 0))  # (K, C, N)
        period = int(Ca)
    else:
        # Angle-only: select SF=1 (exactly as in your working main())
        TEST_DATA_1D = resort_preprocessing(DATA, ANG, SF, animal)[:, :, 1, :, t0:t1]
        RESP_1D = jnp.nanmean(TEST_DATA_1D, axis=-1)  # (N, C, K)
        good_trials = ~jnp.isnan(RESP_1D).all(axis=(0, 1))  # (K,)
        RESP_1D = RESP_1D[..., good_trials]  # (N, C, K')
        y_full = jnp.transpose(RESP_1D, (2, 1, 0))  # (K', C, N)
        C = y_full.shape[1]
        # Keep x as 2D column so split_data sees it like before; we’ll flatten after the split
        x_full = jnp.arange(C)[:, None]  # (C,1)
        period = int(C)

    return x_full, y_full, int(N), int(period)


# -----------------------------------------------------------------------------
# G. One animal run wrapper (keeps split_data untouched)
# -----------------------------------------------------------------------------

def run_one_animal(x_full, y_full, N, period, two_d, *, steps, n_draws, seed, out_prefix):
    # Do NOT touch split_data; consume its outputs like your working script.
    x_tr, y_tr, _, _, x_te, y_te, *_ = utils.split_data(
        x=x_full,
        y=y_full,
        train_trial_prop=0.8,
        train_condition_prop=0.8,
        seed=seed,
    )
    # 1D path: the working script flattens x_tr/x_te to 1D index arrays
    if not two_d:
        x_tr = jnp.asarray(x_tr).reshape(-1)
        x_te = jnp.asarray(x_te).reshape(-1)

    eval_fn = make_evaluator(
        N=N,
        period=period,
        two_d=two_d,
        x_tr=x_tr,
        y_tr=y_tr,
        x_te=x_te,
        y_te=y_te,
        n_vi_steps=steps,
        n_mc=8,
        V_scale=1e-2,
    )
    best_hp, best_ll, scores, combos = random_search(eval_fn, two_d, n_draws=n_draws, seed=seed)

    # Save per-animal artifacts
    np.savez(
        f"{out_prefix}.npz",
        best_hp=best_hp,
        best_ll=best_ll,
        scores=scores,
        combos=np.array(combos, dtype=object),
    )
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump({"best_hp": best_hp, "best_ll": float(best_ll), "steps": steps, "n_draws": n_draws}, f, indent=2)

    return best_hp, float(best_ll)


# -----------------------------------------------------------------------------
# H. Main: iterate FR + CTR animals, write CSV summary
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two_d", type=int, default=1)  # 1 = angle×SF, 0 = angle only
    parser.add_argument("--n_draws", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    two_d = bool(args.two_d)

    # ----- Load data (adjust paths if needed)
    # (Use whichever set you want; below is HUNGRY for parity with your working file’s top block)
    DATA = np.load("../Data/predictions_fullTrace_hungry.npy", allow_pickle=True)
    ANG = loadmat("../Data/metadata_deconv/stimAngle_hungry.mat", simplify_cells=True)["order_of_stim_arossAnimals"]
    SF  = loadmat("../Data/metadata_deconv/stimSpatFreq_hungry.mat", simplify_cells=True)["stimSpatFreq_arossAnimals"]

    FOOD_RESTRICTED = [1, 2, 3, 6, 7, 9, 11, 12]
    CONTROL         = [0, 4, 5, 8, 10, 13]
    all_animals: List[Tuple[str, int]] = [("FR", a) for a in FOOD_RESTRICTED] + [("CTR", a) for a in CONTROL]

    summary_rows: List[List[Any]] = []

    for grp, animal in all_animals:
        print(f"\n=== {grp} animal {animal} ===")
        x_full, y_full, N, period = build_xy_for_animal(DATA, ANG, SF, animal, two_d=two_d)

        out_prefix = f"randsearch_{grp}_{animal}"
        best_hp, best_ll = run_one_animal(
            x_full,
            y_full,
            N,
            period,
            two_d=two_d,
            steps=args.steps,
            n_draws=args.n_draws,
            seed=args.seed,
            out_prefix=out_prefix,
        )

        P_val = int(best_hp.get("p", 0)) if isinstance(best_hp, dict) else 0
        lwp   = float(best_hp.get("l_wp_a", np.nan)) if isinstance(best_hp, dict) else np.nan
        bwp   = float(best_hp.get("b_wp_a", np.nan)) if isinstance(best_hp, dict) else np.nan
        summary_rows.append([grp, animal, int(N), int(period), float(best_ll), P_val, lwp, bwp])

    # CSV summary
    with open("random_search_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "animal", "N", "period", "best_ll", "p", "l_wp_a", "b_wp_a"])
        w.writerows(summary_rows)

    print("\nDone. Wrote random_search_summary.csv")


if __name__ == "__main__":
    main()
