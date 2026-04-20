#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-validated hyperparameter sweeps for the GP/Wishart model.

Each run:
- 5 randomized trial folds (train/test split)
- sweep four distinct lambdas (gp/wp x angle/sf) and number of components (P)
- report cross-validated log probability of test trials
- plot log probability vs smoothness (geometric mean of lambdas) and vs P
"""
import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import json
import itertools
from pathlib import Path

import jax
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import optim

import inference
import models
import utils

DATA_PATH = "data_sated_animal_1.npz"
X_KEY = "x"
Y_KEY = "y"

SEED = 10
NUM_FOLDS = 5
TRAIN_TRIAL_PROP = 0.8
TRAIN_CONDITION_PROP = 1.0

ADAM_STEP = 0.001
ITERATIONS = 50000
NUM_PARTICLES = 1
MC_DRAWS = 50

GAMMA = 1e-5
BETA_WP = 1.0
WP_SAMPLE_DIAG = GAMMA
OPTIMIZE_L = True
PERIOD = None  # set to None to infer from x

# LAMBDA_GRID = {
#     "gp_angle": [1.5, 2.0, 2.5],
#     "gp_sf": [1.0, 3.0, 6.0],
#     "wp_angle": [0.25, 0.5, 1.0, 3.0],
#     "wp_sf": [1.0, 1.5],
# }
LAMBDA_GRID = {
    "gp_angle": [1.8],
    "gp_sf":    [3.0, 6.0, 12.0, 20.0,25.0],
    "wp_angle": [0.5,1.0, 3.0, 5.0,10.0],
    "wp_sf":    [12.0, 20.0,25.0],
}
# LAMBDA_GRID = {
#     "gp_angle": [1.8],
#     "gp_sf":    [3.0, 6.0, 12.0, 20.0,25.0],
#     "wp_angle": [0.5,1.0, 3.0, 5.0,10.0],
#     "wp_sf":    [1.0, 1.5, 6.0],
# }

SEARCH_STRATEGY = "random"  # "grid" or "random"
N_RANDOM_SAMPLES = 50
LAMBDA_RANGES = {
    "gp_angle": (0.1, 50.0),
    "gp_sf": (0.001, 25.0),
    "wp_angle": (5.0, 50.0),
    "wp_sf": (0.1,10.0),
}
LAMBDA_SAMPLE = "uniform"  # or "uniform"

P_VALUES = [0,2,3]
FULL_GRID_SEARCH = True  # if True, grid search over lambdas x P
P_FOR_LAMBDA_SWEEP = 0
LAMBDA_FOR_P_SWEEP = None  # if dict, use those lambdas; if None, use best combo

OUTPUT_DIR = "outputs"
RESULTS_PATH = "outputs/hyperparam_cv_results_feb_17.json"

def estimate_beta_gp(y_train):
    mu = y_train.mean(axis=0)
    mu_var = jnp.var(mu, axis=0)
    v = mu_var + 1e-12
    v_geo = jnp.exp(jnp.mean(jnp.log(v)))
    return float(jnp.sqrt(v_geo))

def build_kernels(hyperparams, period):
    periodic_gp_angle = lambda a, b: hyperparams["gamma_gp_angle"] * (a == b) + hyperparams["beta_gp_angle"] * jnp.exp(
        -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyperparams["lambda_gp_angle"]
    )
    square_gp_sf = lambda a, b: hyperparams["gamma_gp_sf"] * (a == b) + hyperparams["beta_gp_sf"] * jnp.exp(
        -((a - b) ** 2) / hyperparams["lambda_gp_sf"]
    )
    kernel_gp = lambda x, y: periodic_gp_angle(x[0], y[0]) * square_gp_sf(x[1], y[1])

    periodic_wp_angle = lambda a, b: hyperparams["gamma_wp_angle"] * (a == b) + hyperparams["beta_wp_angle"] * jnp.exp(
        -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyperparams["lambda_wp_angle"]
    )
    square_wp_sf = lambda a, b: hyperparams["gamma_wp_sf"] * (a == b) + hyperparams["beta_wp_sf"] * jnp.exp(
        -((a - b) ** 2) / hyperparams["lambda_wp_sf"]
    )
    kernel_wp = lambda x, y: periodic_wp_angle(x[0], y[0]) * square_wp_sf(x[1], y[1])

    return kernel_gp, kernel_wp


def fit_posterior(x_train, y_train, hyperparams, period, seed):
    K, C, N = y_train.shape
    kernel_gp, kernel_wp = build_kernels(hyperparams, period)

    gp = models.GaussianProcess(kernel=kernel_gp, N=N)

    y_centered = y_train - y_train.mean(0, keepdims=True)
    y_flat = y_centered.reshape(K * C, N)
    if y_flat.shape[0] < 2:
        empirical = WP_SAMPLE_DIAG * jnp.eye(N, dtype=jnp.float64)
    else:
        empirical = jnp.cov(y_flat.T)
    V_picked = empirical + WP_SAMPLE_DIAG * jnp.eye(N, dtype=jnp.float64)

    diag_scale = 0.1 if hyperparams["p"] > 0 else 1.25
    wp = models.WishartLRDProcess(
        kernel=kernel_wp,
        P=int(hyperparams["p"]),
        V=V_picked,
        optimize_L=OPTIMIZE_L,
        diag_scale=diag_scale,
    )

    lik = models.NormalConditionalLikelihood(N)
    joint = models.JointGaussianWishartProcess(gp, wp, lik)

    init = {"G": y_train.mean(0).T[:, None]}
    varfam = inference.VariationalNormal(joint.model, init=init)
    optimizer = optim.Adam(ADAM_STEP)

    varfam.infer(
        optimizer,
        x_train,
        y_train,
        n_iter=ITERATIONS,
        key=jax.random.PRNGKey(seed),
        num_particles=NUM_PARTICLES,
    )

    joint.update_params(varfam.posterior)
    posterior = models.NormalGaussianWishartPosterior(joint, varfam, x_train)
    return posterior, lik


def mc_log_prob_trials(posterior, x_eval, y_eval, vi_samples, gp_samples, seed):
    if y_eval.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "lp": None,
        }

    with numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(seed)):
        lp = posterior.log_prob(x_eval, y_eval, vi_samples=vi_samples, gp_samples=gp_samples)

    lp = jnp.asarray(lp)
    return {
        "mean": float(lp.mean()),
        "std": float(lp.std()),
        "lp": np.asarray(lp).tolist(),
    }


def sample_lambdas(rng):
    def draw(low, high):
        if LAMBDA_SAMPLE == "loguniform":
            if low <= 0 or high <= 0:
                raise ValueError("loguniform sampling needs positive ranges")
            return float(10 ** rng.uniform(np.log10(low), np.log10(high)))
        if LAMBDA_SAMPLE == "uniform":
            return float(rng.uniform(low, high))
        raise ValueError(f"Unsupported LAMBDA_SAMPLE: {LAMBDA_SAMPLE}")

    return {
        "gp_angle": draw(*LAMBDA_RANGES["gp_angle"]),
        "gp_sf": draw(*LAMBDA_RANGES["gp_sf"]),
        "wp_angle": draw(*LAMBDA_RANGES["wp_angle"]),
        "wp_sf": draw(*LAMBDA_RANGES["wp_sf"]),
    }


def iter_lambda_combos(grid):
    keys = ["gp_angle", "gp_sf", "wp_angle", "wp_sf"]
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def run_cv_for_params(x_full, y_full, lambdas, p_val, fold_seeds):
    fold_results = []
    for fold_seed in fold_seeds:
        split = utils.split_data(
            x_full,
            y_full,
            train_trial_prop=TRAIN_TRIAL_PROP,
            train_condition_prop=TRAIN_CONDITION_PROP,
            seed=fold_seed,
        )
        x_tr, y_tr, _, _, _, y_te, *_ = split
        y_test = y_te["x"]

        x_tr = jnp.asarray(x_tr)
        y_tr = jnp.asarray(y_tr)
        y_test = jnp.asarray(y_test)

        period = PERIOD
        if period is None:
            period = int(np.unique(np.asarray(x_tr)[:, 0]).size)

        beta_gp = estimate_beta_gp(y_tr)
        hyperparams = {
            "gamma_gp_angle": GAMMA,
            "beta_gp_angle": beta_gp,
            "lambda_gp_angle": float(lambdas["gp_angle"]),
            "gamma_gp_sf": GAMMA,
            "beta_gp_sf": beta_gp,
            "lambda_gp_sf": float(lambdas["gp_sf"]),
            "gamma_wp_angle": GAMMA,
            "beta_wp_angle": BETA_WP,
            "lambda_wp_angle": float(lambdas["wp_angle"]),
            "gamma_wp_sf": GAMMA,
            "beta_wp_sf": BETA_WP,
            "lambda_wp_sf": float(lambdas["wp_sf"]),
            "p": int(p_val),
        }

        posterior, lik = fit_posterior(x_tr, y_tr, hyperparams, period, fold_seed)
        ll_stats = mc_log_prob_trials(
            posterior,
            x_tr,
            y_test,
            vi_samples=MC_DRAWS,
            gp_samples=1,
            seed=fold_seed + 1000,
        )
        score = ll_stats["mean"]
        fold_results.append(
            {
                "seed": int(fold_seed),
                "beta_gp": float(beta_gp),
                "period": int(period),
                "x_train_shape": list(x_tr.shape),
                "y_train_shape": list(y_tr.shape),
                "y_test_shape": list(y_test.shape),
                "score_mean": float(score),
                "score_std": float(ll_stats["std"]),
                "score_lp": ll_stats["lp"],
            }
        )

    return fold_results


def summarize_folds(folds):
    scores = np.asarray([fold["score_mean"] for fold in folds], dtype=float)
    return {
        "mean": float(np.nanmean(scores)) if scores.size else float("nan"),
        "std": float(np.nanstd(scores)) if scores.size else float("nan"),
        "scores": scores.tolist(),
    }



def main():
    data = np.load(DATA_PATH)
    if X_KEY not in data or Y_KEY not in data:
        raise KeyError(f"Expected keys {X_KEY!r} and {Y_KEY!r} in {DATA_PATH}")
    x_full = np.asarray(data[X_KEY])
    y_full = np.asarray(data[Y_KEY])

    if x_full.ndim == 1:
        x_full = x_full[:, None]
    if x_full.ndim != 2 or x_full.shape[1] != 2:
        raise ValueError(f"Expected x to have shape (C, 2). Got {x_full.shape}.")

    fold_seeds = [SEED + i for i in range(NUM_FOLDS)]

    combo_results = []

    if SEARCH_STRATEGY == "grid":
        for lambdas in iter_lambda_combos(LAMBDA_GRID):
            p_iter = P_VALUES if FULL_GRID_SEARCH else [P_FOR_LAMBDA_SWEEP]
            for p_val in p_iter:
                folds = run_cv_for_params(x_full, y_full, lambdas, p_val, fold_seeds)
                summary = summarize_folds(folds)
                combo_results.append(
                    {
                        "lambdas": {k: float(v) for k, v in lambdas.items()},
                        "p": int(p_val),
                        "folds": folds,
                        "score_summary": summary,
                        "mean": summary["mean"],
                        "std": summary["std"],
                    }
                )
    elif SEARCH_STRATEGY == "random":
        rng = np.random.default_rng(SEED)
        for sample_id in range(N_RANDOM_SAMPLES):
            lambdas = sample_lambdas(rng)
            p_val = int(rng.choice(P_VALUES))
            folds = run_cv_for_params(x_full, y_full, lambdas, p_val, fold_seeds)
            summary = summarize_folds(folds)
            combo_results.append(
                {
                    "sample_id": int(sample_id),
                    "lambdas": {k: float(v) for k, v in lambdas.items()},
                    "p": int(p_val),
                    "folds": folds,
                    "score_summary": summary,
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )
    else:
        raise ValueError(f"Unsupported SEARCH_STRATEGY: {SEARCH_STRATEGY}")

    best_combo = None
    if combo_results:
        best_idx = int(np.nanargmax(np.asarray([item["mean"] for item in combo_results])))
        best_combo = combo_results[best_idx]

    p_entries = None
    lambda_for_p = None
    if SEARCH_STRATEGY == "grid" and not FULL_GRID_SEARCH:
        best_lambda = best_combo["lambdas"] if best_combo else None
        if isinstance(LAMBDA_FOR_P_SWEEP, dict):
            lambda_for_p = LAMBDA_FOR_P_SWEEP
        else:
            lambda_for_p = best_lambda
        if lambda_for_p is None:
            lambda_for_p = {k: float(v[0]) for k, v in LAMBDA_GRID.items()}

        p_entries = []
        for p_val in P_VALUES:
            folds = run_cv_for_params(x_full, y_full, lambda_for_p, p_val, fold_seeds)
            summary = summarize_folds(folds)
            p_entries.append(
                {
                    "p": int(p_val),
                    "folds": folds,
                    "score_summary": summary,
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )

    output = {
        "data_summary": {
            "x_shape": list(x_full.shape),
            "y_shape": list(y_full.shape),
        },
        "lambda_sweep": {
            "entries": combo_results,
            "p_fixed": None if SEARCH_STRATEGY == "random" else (None if FULL_GRID_SEARCH else int(P_FOR_LAMBDA_SWEEP)),
            "full_grid": bool(FULL_GRID_SEARCH) if SEARCH_STRATEGY == "grid" else None,
            "strategy": SEARCH_STRATEGY,
        },
        "p_sweep": None if SEARCH_STRATEGY == "random" or FULL_GRID_SEARCH else {
            "entries": p_entries,
            "lambda_fixed": lambda_for_p,
        },
        "best_combo": best_combo,
        "best_lambda": None if best_combo is None else best_combo["lambdas"],
        "best_p": None if best_combo is None else best_combo["p"],
        "fold_seeds": fold_seeds,
        "config": {
            "data_path": DATA_PATH,
            "train_trial_prop": TRAIN_TRIAL_PROP,
            "train_condition_prop": TRAIN_CONDITION_PROP,
            "num_folds": NUM_FOLDS,
            "iterations": ITERATIONS,
            "adam_step": ADAM_STEP,
            "num_particles": NUM_PARTICLES,
            "mc_draws": MC_DRAWS,
            "gamma": GAMMA,
            "beta_wp": BETA_WP,
            "period": PERIOD,
            "lambda_grid": LAMBDA_GRID,
            "lambda_ranges": LAMBDA_RANGES,
            "lambda_sample": LAMBDA_SAMPLE,
            "n_random_samples": N_RANDOM_SAMPLES,
            "search_strategy": SEARCH_STRATEGY,
            "full_grid_search": FULL_GRID_SEARCH,
        },
    }

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
