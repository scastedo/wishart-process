#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated per-animal hyperparameter search for the GP/Wishart model.

Stage 1 searches GP lambdas with closed-form leave-one-angle-out interpolation.
Stage 2 freezes the best GP lambdas and searches WP lambdas with trial holdout:
first a cheap screen, then a confirmation pass for the best screen settings.
"""
import argparse
import csv
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

np = None
jax = None
jnp = None
numpyro = None
optim = None
inference = None
models = None


def ensure_numpy():
    global np
    if np is not None:
        return

    import numpy as _np

    np = _np


def ensure_model_libraries():
    global jax, jnp, numpyro, optim, inference, models
    if jax is not None:
        return

    ensure_numpy()

    import jax as _jax
    from jax import config as _config

    _config.update("jax_enable_x64", True)
    _config.update("jax_default_matmul_precision", "highest")

    import jax.numpy as _jnp
    import numpyro as _numpyro
    from numpyro import optim as _optim

    import inference as _inference
    import models as _models

    jax = _jax
    jnp = _jnp
    numpyro = _numpyro
    optim = _optim
    inference = _inference
    models = _models


# -------------------------
# Editable defaults
# -------------------------
DEFAULT_ANIMALS = list(range(14))
FOOD_RESTRICTED_SATED = [1, 2, 3, 6, 7, 8, 11, 12]
CONTROL_SATED = [0, 4, 5, 9, 10, 13]

DATA_PATH = "../Data/predictions_fullTrace_sated.npy"
ANGLE_METADATA_PATH = "../Data/metadata_deconv/stimAngle_sated.mat"
SF_METADATA_PATH = "../Data/metadata_deconv/stimSpatFreq_sated.mat"

OUTPUT_DIR = "outputs/animal_hyperparam_search"
START = 40
STOP = 80
SEED = 10

PERIOD = 12
GAMMA = 1e-5
BETA_WP = 1.0
WP_SAMPLE_DIAG = GAMMA
ADAM_STEP = 0.001
NUM_PARTICLES = 1
P_VALUE = 0

GP_ANGLE_GRID = list(range(5, 36))
GP_SF_GRID = list(range(10, 41))

WP_SCREEN_ANGLE_GRID = [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]
WP_SCREEN_SF_GRID = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
WP_SCREEN_FOLDS = [0, 4, 8]
WP_SCREEN_ITERATIONS = 15000
WP_SCREEN_MC_DRAWS = 10

WP_CONFIRM_TOP_K = 4
WP_CONFIRM_ITERATIONS = 30000
WP_CONFIRM_MC_DRAWS = 20


# -------------------------
# Small serialization helpers
# -------------------------
def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and value.__class__.__module__.startswith("jax") and hasattr(value, "tolist"):
        return np.asarray(value).tolist()
    if np is not None:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
    return value


def atomic_json_dump(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    tmp_path.replace(path)


def summarize_scores(scores):
    ensure_numpy()
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "n": int(arr.size),
            "scores": arr.tolist(),
        }
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "se": float(np.nanstd(arr) / np.sqrt(np.sum(~np.isnan(arr)))),
        "n": int(np.sum(~np.isnan(arr))),
        "scores": arr.tolist(),
    }


def choose_best(entries, key="mean"):
    ensure_numpy()
    if not entries:
        return None
    scores = np.asarray([entry.get(key, float("nan")) for entry in entries], dtype=float)
    if np.all(np.isnan(scores)):
        return None
    return entries[int(np.nanargmax(scores))]


# -------------------------
# Data loading / preprocessing
# -------------------------
def load_raw_sated_data(data_path, angle_metadata_path, sf_metadata_path):
    ensure_numpy()
    from scipy.io import loadmat

    sated_deconv = np.load(data_path, allow_pickle=True)
    angle_metadata = loadmat(angle_metadata_path, simplify_cells=True)
    sf_metadata = loadmat(sf_metadata_path, simplify_cells=True)
    return (
        sated_deconv,
        angle_metadata["order_of_stim_arossAnimals"],
        sf_metadata["stimSpatFreq_arossAnimals"],
    )


def resort_preprocessing(datum, angle_arr, sf_arr, animal):
    ensure_numpy()
    data = np.copy(datum[animal, :])
    neurons = data[0].shape[0]
    reshape_data = np.full((60, neurons, data[0].shape[1]), np.nan)
    for i in range(60):
        reshape_data[i, :, :] = data[i]

    reshape_data = reshape_data.reshape(60, neurons, 12, 120)
    reshape_data = np.transpose(reshape_data, (1, 2, 0, 3))

    # Match the existing SNR script: remove the first two neurons.
    reshape_data = reshape_data[2:, :, :, :]

    max_trial = np.argmax(np.isnan(reshape_data[0, 1, :, 0]))
    if max_trial > 0:
        reshape_data = reshape_data[:, :, :max_trial, :]

    angles = np.copy(angle_arr[animal])
    for itrials in range(angles.shape[1]):
        order = (angles[:, itrials] - 1).astype(int)
        reshape_data[:, :, itrials, :] = reshape_data[:, order, itrials, :]

    reshaped_data = []
    sfs = np.copy(sf_arr[animal])
    for experiment in range(1, 6):
        mask = sfs == experiment
        reshaped_data.append(reshape_data[:, :, mask, :])

    max_trials = max(exp.shape[2] for exp in reshaped_data)
    for i, exp in enumerate(reshaped_data):
        if exp.shape[2] < max_trials:
            padding = max_trials - exp.shape[2]
            reshaped_data[i] = np.pad(
                exp,
                ((0, 0), (0, 0), (0, padding), (0, 0)),
                mode="constant",
                constant_values=np.nan,
            )

    return np.stack(reshaped_data, axis=2)


def animal_group(animal):
    if animal in FOOD_RESTRICTED_SATED:
        return "food_restricted_sated"
    if animal in CONTROL_SATED:
        return "control_sated"
    return "unknown"


def preprocess_animal(datum, angle_arr, sf_arr, animal, start, stop):
    ensure_model_libraries()
    test_data = resort_preprocessing(datum, angle_arr, sf_arr, animal)[:, :, :, :, start:stop]
    resp = jnp.nanmean(test_data, axis=-1).transpose(3, 1, 2, 0)
    resp = resp[~jnp.isnan(resp).any(axis=(1, 2, 3))]

    if resp.ndim != 4:
        raise ValueError(f"Animal {animal}: expected response array with 4 dims, got {resp.shape}.")

    k_trials, n_angles, n_sfs, n_neurons = resp.shape
    if k_trials < 2:
        raise ValueError(f"Animal {animal}: needs at least 2 valid trials, got {k_trials}.")

    angles = jnp.arange(n_angles)
    sfs = jnp.log2(jnp.array([0.02, 0.04, 0.08, 0.16, 0.32]))
    if len(sfs) != n_sfs:
        raise ValueError(f"Animal {animal}: expected {len(sfs)} SFs, got {n_sfs}.")

    x_full = jnp.stack(jnp.meshgrid(angles, sfs, indexing="ij"), axis=-1).reshape(-1, 2)
    y_full = resp.reshape(k_trials, n_angles * n_sfs, n_neurons).astype(np.float64, copy=False)

    return np.asarray(x_full), np.asarray(y_full), {
        "animal": int(animal),
        "group": animal_group(int(animal)),
        "start": int(start),
        "stop": int(stop),
        "num_trials": int(k_trials),
        "num_angles": int(n_angles),
        "num_sfs": int(n_sfs),
        "num_conditions": int(n_angles * n_sfs),
        "num_neurons": int(n_neurons),
        "x_shape": list(x_full.shape),
        "y_shape": list(y_full.shape),
    }


# -------------------------
# Kernels / model fitting
# -------------------------
def estimate_beta_gp(y_train):
    ensure_model_libraries()
    mu = jnp.asarray(y_train).mean(axis=0)
    mu_var = jnp.var(mu, axis=0)
    v = mu_var + 1e-12
    v_geo = jnp.exp(jnp.mean(jnp.log(v)))
    return float(jnp.sqrt(v_geo))


def make_hyperparams(gp_angle, gp_sf, wp_angle, wp_sf, p, beta_gp):
    return {
        "gamma_gp_angle": GAMMA,
        "beta_gp_angle": float(beta_gp),
        "lambda_gp_angle": float(gp_angle),
        "gamma_gp_sf": GAMMA,
        "beta_gp_sf": float(beta_gp),
        "lambda_gp_sf": float(gp_sf),
        "gamma_wp_angle": GAMMA,
        "beta_wp_angle": BETA_WP,
        "lambda_wp_angle": float(wp_angle),
        "gamma_wp_sf": GAMMA,
        "beta_wp_sf": BETA_WP,
        "lambda_wp_sf": float(wp_sf),
        "p": int(p),
    }


def build_kernels(hyperparams, period):
    ensure_model_libraries()
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


def fit_posterior(x_train, y_train, hyperparams, period, seed, iterations):
    ensure_model_libraries()
    k_trials, _, n_neurons = y_train.shape
    kernel_gp, kernel_wp = build_kernels(hyperparams, period)

    gp = models.GaussianProcess(kernel=kernel_gp, N=n_neurons)

    y_centered = y_train - y_train.mean(0, keepdims=True)
    y_flat = y_centered.reshape(k_trials * y_train.shape[1], n_neurons)
    if y_flat.shape[0] < 2:
        empirical = WP_SAMPLE_DIAG * jnp.eye(n_neurons, dtype=jnp.float64)
    else:
        empirical = jnp.cov(y_flat.T)
    v_picked = empirical + WP_SAMPLE_DIAG * jnp.eye(n_neurons, dtype=jnp.float64)

    diag_scale = 0.1 if hyperparams["p"] > 0 else 1.25
    wp = models.WishartLRDProcess(
        kernel=kernel_wp,
        P=int(hyperparams["p"]),
        V=v_picked,
        optimize_L=True,
        diag_scale=diag_scale,
    )

    lik = models.NormalConditionalLikelihood(n_neurons)
    joint = models.JointGaussianWishartProcess(gp, wp, lik)

    init = {"G": y_train.mean(0).T[:, None]}
    varfam = inference.VariationalNormal(joint.model, init=init)
    optimizer = optim.Adam(ADAM_STEP)

    varfam.infer(
        optimizer,
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        n_iter=int(iterations),
        key=jax.random.PRNGKey(int(seed)),
        num_particles=NUM_PARTICLES,
    )

    joint.update_params(varfam.posterior)
    posterior = models.NormalGaussianWishartPosterior(joint, varfam, jnp.asarray(x_train))
    return posterior


def mc_log_prob_trials(posterior, x_eval, y_eval, vi_samples, gp_samples, seed):
    if y_eval.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "lp": None}

    with numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(int(seed))):
        lp = posterior.log_prob(
            jnp.asarray(x_eval),
            jnp.asarray(y_eval),
            vi_samples=int(vi_samples),
            gp_samples=int(gp_samples),
        )

    lp = jnp.asarray(lp)
    return {
        "mean": float(lp.mean()),
        "std": float(lp.std()),
        "lp": np.asarray(lp).tolist(),
    }


# -------------------------
# GP interpolation score
# -------------------------
def split_leave_one_angle_out(x_full, y_full, fold_id):
    x_np = np.asarray(x_full)
    unique_angles = np.sort(np.unique(x_np[:, 0]))
    test_angle = unique_angles[int(fold_id)]
    train_mask = ~np.isclose(x_np[:, 0], test_angle)
    test_mask = np.isclose(x_np[:, 0], test_angle)
    return (
        x_full[train_mask],
        y_full[:, train_mask, :],
        x_full[test_mask],
        y_full[:, test_mask, :],
        np.asarray([test_angle]),
    )


def align_mean_shape(mu_hat, mu_true):
    mu_hat = jnp.asarray(mu_hat)
    if mu_hat.shape == mu_true.shape:
        return mu_hat
    if mu_hat.T.shape == mu_true.shape:
        return mu_hat.T
    raise ValueError(f"Predicted mean shape {mu_hat.shape} does not match target {mu_true.shape}.")


def circular_angle_distance(a, b, period):
    diff = np.abs(a - b)
    return np.minimum(diff, period - diff)


def circular_linear_angle_baseline(x_train, y_train, x_eval, period):
    x_train_np = np.asarray(x_train)
    x_eval_np = np.asarray(x_eval)
    mu_train = np.asarray(y_train).mean(axis=0)

    preds = []
    for x_row in x_eval_np:
        same_sf = np.isclose(x_train_np[:, 1], x_row[1])
        candidates = np.where(same_sf)[0]
        if candidates.size == 0:
            candidates = np.arange(x_train_np.shape[0])

        angles = x_train_np[candidates, 0]
        forward_from_candidate = (x_row[0] - angles) % period
        forward_to_candidate = (angles - x_row[0]) % period

        prev_idx = candidates[np.argmin(forward_from_candidate)]
        next_idx = candidates[np.argmin(forward_to_candidate)]
        prev_dist = float(np.min(forward_from_candidate))
        next_dist = float(np.min(forward_to_candidate))
        denom = prev_dist + next_dist

        if denom == 0:
            pred = mu_train[prev_idx]
        else:
            pred = (next_dist / denom) * mu_train[prev_idx] + (prev_dist / denom) * mu_train[next_idx]
        preds.append(pred)

    return jnp.asarray(np.stack(preds, axis=0))


def prediction_metrics(mu_hat, mu_true, y_eval):
    mu_hat = align_mean_shape(mu_hat, mu_true)
    err = mu_hat - mu_true
    mse = jnp.mean(err ** 2)

    trial_var = jnp.var(jnp.asarray(y_eval), axis=0)
    sem_var = trial_var / max(y_eval.shape[0], 1)
    noise_norm_mse = jnp.mean((err ** 2) / (sem_var + 1e-8))

    a = mu_hat.reshape(-1) - mu_hat.mean()
    b = mu_true.reshape(-1) - mu_true.mean()
    corr = jnp.sum(a * b) / (jnp.sqrt(jnp.sum(a ** 2) * jnp.sum(b ** 2)) + 1e-8)

    return {
        "mse": float(mse),
        "rmse": float(jnp.sqrt(mse)),
        "noise_norm_mse": float(noise_norm_mse),
        "corr": float(corr),
    }


def empirical_gp_mean_baseline(x_train, y_train, x_eval, hyperparams, period):
    kernel_gp, _ = build_kernels(hyperparams, period)
    gp = models.GaussianProcess(kernel=kernel_gp, N=y_train.shape[-1])
    mu_train = jnp.asarray(y_train).mean(axis=0)
    return gp.posterior_mode(jnp.asarray(x_train), mu_train, jnp.asarray(x_eval))


def empirical_gp_fold_score(x_train, y_train, x_eval, y_eval, hyperparams, period):
    mu_true = jnp.asarray(y_eval).mean(axis=0)
    gp_metrics = prediction_metrics(
        empirical_gp_mean_baseline(x_train, y_train, x_eval, hyperparams, period),
        mu_true,
        y_eval,
    )
    linear_metrics = prediction_metrics(
        circular_linear_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )
    return {
        "score": -gp_metrics["mse"],
        "mse": gp_metrics["mse"],
        "rmse": gp_metrics["rmse"],
        "corr": gp_metrics["corr"],
        "noise_norm_mse": gp_metrics["noise_norm_mse"],
        "circular_linear_mse": linear_metrics["mse"],
        "gp_minus_circular_linear_mse": gp_metrics["mse"] - linear_metrics["mse"],
    }


def run_gp_search(x_full, y_full, gp_angle_grid, gp_sf_grid):
    ensure_model_libraries()
    fold_ids = list(range(len(np.sort(np.unique(np.asarray(x_full)[:, 0])))))
    entries = []

    for gp_angle, gp_sf in itertools.product(gp_angle_grid, gp_sf_grid):
        fold_scores = []
        fold_mse = []
        fold_corr = []
        fold_improvement = []
        fold_results = []

        for fold_id in fold_ids:
            x_train, y_train, x_eval, y_eval, test_angles = split_leave_one_angle_out(
                x_full, y_full, fold_id
            )
            beta_gp = estimate_beta_gp(y_train)
            hyperparams = make_hyperparams(
                gp_angle=gp_angle,
                gp_sf=gp_sf,
                wp_angle=WP_SCREEN_ANGLE_GRID[0],
                wp_sf=WP_SCREEN_SF_GRID[0],
                p=P_VALUE,
                beta_gp=beta_gp,
            )
            metrics = empirical_gp_fold_score(
                x_train,
                y_train,
                x_eval,
                y_eval,
                hyperparams,
                PERIOD,
            )
            fold_scores.append(metrics["score"])
            fold_mse.append(metrics["mse"])
            fold_corr.append(metrics["corr"])
            fold_improvement.append(metrics["gp_minus_circular_linear_mse"])
            fold_results.append(
                {
                    "fold": int(fold_id),
                    "test_angles": test_angles.tolist(),
                    "beta_gp": float(beta_gp),
                    "score": float(metrics["score"]),
                    "mse": float(metrics["mse"]),
                    "corr": float(metrics["corr"]),
                    "gp_minus_circular_linear_mse": float(metrics["gp_minus_circular_linear_mse"]),
                }
            )

        score_summary = summarize_scores(fold_scores)
        mse_summary = summarize_scores(fold_mse)
        corr_summary = summarize_scores(fold_corr)
        improvement_summary = summarize_scores(fold_improvement)
        entries.append(
            {
                "lambdas": {
                    "gp_angle": float(gp_angle),
                    "gp_sf": float(gp_sf),
                },
                "folds": fold_results,
                "score_summary": score_summary,
                "mse_summary": mse_summary,
                "corr_summary": corr_summary,
                "gp_minus_circular_linear_mse_summary": improvement_summary,
                "mean": score_summary["mean"],
                "std": score_summary["std"],
                "se": score_summary["se"],
            }
        )

    best = choose_best(entries, key="mean")
    return {
        "entries": entries,
        "best": best,
        "fold_ids": fold_ids,
        "config": {
            "score_metric": "negative_empirical_gp_mean_mse",
            "gp_angle_grid": [float(v) for v in gp_angle_grid],
            "gp_sf_grid": [float(v) for v in gp_sf_grid],
            "p": int(P_VALUE),
            "period": int(PERIOD),
        },
    }


# -------------------------
# WP trial-holdout score
# -------------------------
def split_leave_one_trial_out(x_full, y_full, fold_id):
    n_trials = y_full.shape[0]
    test_trial = int(fold_id) % n_trials

    train_mask = np.ones(n_trials, dtype=bool)
    train_mask[test_trial] = False

    return (
        x_full,
        y_full[train_mask, :, :],
        x_full,
        y_full[[test_trial], :, :],
        np.where(train_mask)[0],
        np.asarray([test_trial]),
    )


def run_wp_combo(
    x_full,
    y_full,
    gp_angle,
    gp_sf,
    wp_angle,
    wp_sf,
    fold_ids,
    iterations,
    mc_draws,
    seed_offset,
):
    folds = []
    for fold_id in fold_ids:
        x_train, y_train, x_eval, y_eval, train_trials, test_trials = split_leave_one_trial_out(
            x_full,
            y_full,
            fold_id,
        )
        beta_gp = estimate_beta_gp(y_train)
        hyperparams = make_hyperparams(
            gp_angle=gp_angle,
            gp_sf=gp_sf,
            wp_angle=wp_angle,
            wp_sf=wp_sf,
            p=P_VALUE,
            beta_gp=beta_gp,
        )
        fit_seed = int(seed_offset + fold_id)
        posterior = fit_posterior(
            x_train,
            y_train,
            hyperparams,
            PERIOD,
            seed=fit_seed,
            iterations=iterations,
        )
        ll_stats = mc_log_prob_trials(
            posterior,
            x_eval,
            y_eval,
            vi_samples=mc_draws,
            gp_samples=1,
            seed=fit_seed + 1000,
        )
        folds.append(
            {
                "fold": int(fold_id),
                "train_trials": train_trials.tolist(),
                "test_trials": test_trials.tolist(),
                "beta_gp": float(beta_gp),
                "score_mean": float(ll_stats["mean"]),
                "score_std": float(ll_stats["std"]),
                "score_metric": "trial_holdout_log_prob",
                "score_lp": ll_stats["lp"],
            }
        )

    summary = summarize_scores([fold["score_mean"] for fold in folds])
    return {
        "lambdas": {
            "gp_angle": float(gp_angle),
            "gp_sf": float(gp_sf),
            "wp_angle": float(wp_angle),
            "wp_sf": float(wp_sf),
        },
        "p": int(P_VALUE),
        "folds": folds,
        "score_summary": summary,
        "mean": summary["mean"],
        "std": summary["std"],
        "se": summary["se"],
        "iterations": int(iterations),
        "mc_draws": int(mc_draws),
    }


def run_wp_stage(
    x_full,
    y_full,
    gp_angle,
    gp_sf,
    screen_angle_grid,
    screen_sf_grid,
    screen_folds,
    screen_iterations,
    screen_mc_draws,
    confirm_top_k,
    confirm_iterations,
    confirm_mc_draws,
    seed,
    progress_callback=None,
):
    n_trials = y_full.shape[0]
    screen_fold_ids = [int(fold) for fold in screen_folds if int(fold) < n_trials]
    if not screen_fold_ids:
        screen_fold_ids = [0]

    all_fold_ids = list(range(n_trials))
    screen_entries = []

    for combo_id, (wp_angle, wp_sf) in enumerate(itertools.product(screen_angle_grid, screen_sf_grid)):
        entry = run_wp_combo(
            x_full,
            y_full,
            gp_angle,
            gp_sf,
            wp_angle,
            wp_sf,
            screen_fold_ids,
            screen_iterations,
            screen_mc_draws,
            seed_offset=seed + 10000 + combo_id * 100,
        )
        entry["stage"] = "screen"
        screen_entries.append(entry)
        if progress_callback is not None:
            progress_callback("wp_screen", screen_entries, None)

    ranked_screen = sorted(
        screen_entries,
        key=lambda item: item["mean"] if not np.isnan(item["mean"]) else -np.inf,
        reverse=True,
    )
    confirm_lambdas = []
    seen = set()
    for entry in ranked_screen:
        wp_angle = entry["lambdas"]["wp_angle"]
        wp_sf = entry["lambdas"]["wp_sf"]
        key = (wp_angle, wp_sf)
        if key not in seen:
            confirm_lambdas.append(key)
            seen.add(key)
        if len(confirm_lambdas) >= int(confirm_top_k):
            break

    confirm_entries = []
    for combo_id, (wp_angle, wp_sf) in enumerate(confirm_lambdas):
        entry = run_wp_combo(
            x_full,
            y_full,
            gp_angle,
            gp_sf,
            wp_angle,
            wp_sf,
            all_fold_ids,
            confirm_iterations,
            confirm_mc_draws,
            seed_offset=seed + 20000 + combo_id * 100,
        )
        entry["stage"] = "confirm"
        confirm_entries.append(entry)
        if progress_callback is not None:
            progress_callback("wp_confirm", screen_entries, confirm_entries)

    best = choose_best(confirm_entries, key="mean")
    return {
        "screen": {
            "entries": screen_entries,
            "best": choose_best(screen_entries, key="mean"),
            "fold_ids": screen_fold_ids,
        },
        "confirm": {
            "entries": confirm_entries,
            "best": best,
            "fold_ids": all_fold_ids,
        },
        "best": best,
        "config": {
            "score_metric": "trial_holdout_log_prob",
            "screen_wp_angle_grid": [float(v) for v in screen_angle_grid],
            "screen_wp_sf_grid": [float(v) for v in screen_sf_grid],
            "screen_fold_ids": screen_fold_ids,
            "screen_iterations": int(screen_iterations),
            "screen_mc_draws": int(screen_mc_draws),
            "confirm_top_k": int(confirm_top_k),
            "confirm_fold_ids": all_fold_ids,
            "confirm_iterations": int(confirm_iterations),
            "confirm_mc_draws": int(confirm_mc_draws),
            "p": int(P_VALUE),
            "period": int(PERIOD),
        },
    }


# -------------------------
# Orchestration / outputs
# -------------------------
def make_empty_result(animal, args):
    return {
        "status": "started",
        "animal": int(animal),
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "data_summary": None,
        "gp_stage": None,
        "wp_stage": None,
        "final_best_hyperparams": None,
        "config": {
            "data_path": args.data_path,
            "angle_metadata_path": args.angle_metadata_path,
            "sf_metadata_path": args.sf_metadata_path,
            "start": int(args.start),
            "stop": int(args.stop),
            "seed": int(args.seed),
            "gamma": float(GAMMA),
            "beta_wp": float(BETA_WP),
            "wp_sample_diag": float(WP_SAMPLE_DIAG),
            "adam_step": float(ADAM_STEP),
            "num_particles": int(NUM_PARTICLES),
            "p": int(P_VALUE),
            "period": int(PERIOD),
        },
    }


def result_path_for_animal(output_dir, animal):
    return Path(output_dir) / f"animal_{int(animal):02d}_hyperparams.json"


def final_hyperparams_from_best(best_gp, best_wp):
    if best_gp is None or best_wp is None:
        return None
    return {
        "lambda_gp_angle": float(best_gp["lambdas"]["gp_angle"]),
        "lambda_gp_sf": float(best_gp["lambdas"]["gp_sf"]),
        "lambda_wp_angle": float(best_wp["lambdas"]["wp_angle"]),
        "lambda_wp_sf": float(best_wp["lambdas"]["wp_sf"]),
        "p": int(P_VALUE),
        "gamma_gp_angle": float(GAMMA),
        "gamma_gp_sf": float(GAMMA),
        "gamma_wp_angle": float(GAMMA),
        "gamma_wp_sf": float(GAMMA),
        "beta_wp_angle": float(BETA_WP),
        "beta_wp_sf": float(BETA_WP),
        "beta_gp": "estimated_per_fold_or_fit_from_training_data",
    }


def process_animal(animal, datum, angle_arr, sf_arr, args):
    out_path = result_path_for_animal(args.output_dir, animal)
    if out_path.exists() and not args.force:
        with open(out_path, "r") as f:
            existing = json.load(f)
        if existing.get("status") == "completed":
            print(f"Animal {animal}: already completed, skipping ({out_path}).", flush=True)
            return existing

    result = make_empty_result(animal, args)
    atomic_json_dump(result, out_path)

    try:
        print(f"Animal {animal}: preprocessing.", flush=True)
        x_full, y_full, data_summary = preprocess_animal(
            datum,
            angle_arr,
            sf_arr,
            animal,
            args.start,
            args.stop,
        )
        result["data_summary"] = data_summary
        result["status"] = "preprocessed"
        result["updated_at"] = timestamp()
        atomic_json_dump(result, out_path)

        print(
            f"Animal {animal}: GP search over {len(args.gp_angle_grid) * len(args.gp_sf_grid)} combos.",
            flush=True,
        )
        gp_stage = run_gp_search(
            x_full,
            y_full,
            args.gp_angle_grid,
            args.gp_sf_grid,
        )
        result["gp_stage"] = gp_stage
        result["status"] = "gp_completed"
        result["updated_at"] = timestamp()
        atomic_json_dump(result, out_path)

        best_gp = gp_stage["best"]
        if best_gp is None:
            raise RuntimeError(f"Animal {animal}: GP search did not produce a best setting.")

        best_gp_angle = best_gp["lambdas"]["gp_angle"]
        best_gp_sf = best_gp["lambdas"]["gp_sf"]
        print(
            f"Animal {animal}: WP search with GP=({best_gp_angle:g}, {best_gp_sf:g}).",
            flush=True,
        )

        def save_wp_progress(stage_name, screen_entries, confirm_entries):
            partial_wp = {
                "screen": {
                    "entries": screen_entries,
                    "best": choose_best(screen_entries, key="mean"),
                    "fold_ids": [
                        int(fold)
                        for fold in args.wp_screen_folds
                        if int(fold) < y_full.shape[0]
                    ],
                },
                "confirm": None
                if confirm_entries is None
                else {
                    "entries": confirm_entries,
                    "best": choose_best(confirm_entries, key="mean"),
                    "fold_ids": list(range(y_full.shape[0])),
                },
                "best": None if confirm_entries is None else choose_best(confirm_entries, key="mean"),
                "partial_stage": stage_name,
            }
            result["wp_stage"] = partial_wp
            result["status"] = stage_name
            result["updated_at"] = timestamp()
            atomic_json_dump(result, out_path)

        wp_stage = run_wp_stage(
            x_full,
            y_full,
            best_gp_angle,
            best_gp_sf,
            args.wp_screen_angle_grid,
            args.wp_screen_sf_grid,
            args.wp_screen_folds,
            args.wp_screen_iterations,
            args.wp_screen_mc_draws,
            args.wp_confirm_top_k,
            args.wp_confirm_iterations,
            args.wp_confirm_mc_draws,
            args.seed + int(animal) * 1000,
            progress_callback=save_wp_progress,
        )
        result["wp_stage"] = wp_stage
        result["final_best_hyperparams"] = final_hyperparams_from_best(best_gp, wp_stage["best"])
        result["status"] = "completed"
        result["updated_at"] = timestamp()
        atomic_json_dump(result, out_path)
        print(f"Animal {animal}: completed ({out_path}).", flush=True)
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["updated_at"] = timestamp()
        atomic_json_dump(result, out_path)
        raise


def summary_row(result, output_dir):
    animal = int(result["animal"])
    data_summary = result.get("data_summary") or {}
    gp_best = ((result.get("gp_stage") or {}).get("best")) or {}
    wp_best = ((result.get("wp_stage") or {}).get("best")) or {}
    gp_lambdas = gp_best.get("lambdas") or {}
    wp_lambdas = wp_best.get("lambdas") or {}
    gp_summary = gp_best.get("score_summary") or {}
    wp_summary = wp_best.get("score_summary") or {}
    final_hp = result.get("final_best_hyperparams") or {}

    return {
        "animal": animal,
        "group": data_summary.get("group", animal_group(animal)),
        "num_trials": data_summary.get("num_trials"),
        "num_neurons": data_summary.get("num_neurons"),
        "best_gp_angle": gp_lambdas.get("gp_angle"),
        "best_gp_sf": gp_lambdas.get("gp_sf"),
        "best_wp_angle": wp_lambdas.get("wp_angle"),
        "best_wp_sf": wp_lambdas.get("wp_sf"),
        "p": final_hp.get("p", P_VALUE),
        "gp_score": gp_summary.get("mean"),
        "gp_mse": (gp_best.get("mse_summary") or {}).get("mean"),
        "wp_log_prob_mean": wp_summary.get("mean"),
        "wp_log_prob_se": wp_summary.get("se"),
        "status": result.get("status"),
        "result_path": str(result_path_for_animal(output_dir, animal)),
    }


def write_combined_summaries(output_dir, results):
    output_dir = Path(output_dir)
    rows = [summary_row(result, output_dir) for result in results]

    json_path = output_dir / "best_hyperparams_summary.json"
    csv_path = output_dir / "best_hyperparams_summary.csv"
    atomic_json_dump(rows, json_path)

    fieldnames = [
        "animal",
        "group",
        "num_trials",
        "num_neurons",
        "best_gp_angle",
        "best_gp_sf",
        "best_wp_angle",
        "best_wp_sf",
        "p",
        "gp_score",
        "gp_mse",
        "wp_log_prob_mean",
        "wp_log_prob_se",
        "status",
        "result_path",
    ]
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(csv_path)


def parse_csv_numbers(text, cast=float):
    if text is None or text == "":
        return None
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run per-animal GP then WP hyperparameter searches.",
    )
    parser.add_argument("--animals", default=None, help="Comma-separated animal ids, e.g. 1,2,3.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--angle-metadata-path", default=ANGLE_METADATA_PATH)
    parser.add_argument("--sf-metadata-path", default=SF_METADATA_PATH)
    parser.add_argument("--start", type=int, default=START)
    parser.add_argument("--stop", type=int, default=STOP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--gp-angle-grid", default=None, help="Comma-separated GP angle grid override.")
    parser.add_argument("--gp-sf-grid", default=None, help="Comma-separated GP SF grid override.")
    parser.add_argument("--wp-screen-angle-grid", default=None, help="Comma-separated WP screen angle grid override.")
    parser.add_argument("--wp-screen-sf-grid", default=None, help="Comma-separated WP screen SF grid override.")
    parser.add_argument("--wp-screen-folds", default=None, help="Comma-separated WP screen folds override.")
    parser.add_argument("--wp-screen-iterations", type=int, default=WP_SCREEN_ITERATIONS)
    parser.add_argument("--wp-screen-mc-draws", type=int, default=WP_SCREEN_MC_DRAWS)
    parser.add_argument("--wp-confirm-top-k", type=int, default=WP_CONFIRM_TOP_K)
    parser.add_argument("--wp-confirm-iterations", type=int, default=WP_CONFIRM_ITERATIONS)
    parser.add_argument("--wp-confirm-mc-draws", type=int, default=WP_CONFIRM_MC_DRAWS)

    args = parser.parse_args()

    args.animals = parse_csv_numbers(args.animals, int) or DEFAULT_ANIMALS
    args.gp_angle_grid = parse_csv_numbers(args.gp_angle_grid, float) or [float(v) for v in GP_ANGLE_GRID]
    args.gp_sf_grid = parse_csv_numbers(args.gp_sf_grid, float) or [float(v) for v in GP_SF_GRID]
    args.wp_screen_angle_grid = parse_csv_numbers(args.wp_screen_angle_grid, float) or WP_SCREEN_ANGLE_GRID
    args.wp_screen_sf_grid = parse_csv_numbers(args.wp_screen_sf_grid, float) or WP_SCREEN_SF_GRID
    args.wp_screen_folds = parse_csv_numbers(args.wp_screen_folds, int) or WP_SCREEN_FOLDS

    return args


def dry_run(args):
    output_dir = Path(args.output_dir)
    gp_combos = len(args.gp_angle_grid) * len(args.gp_sf_grid)
    wp_screen_combos = len(args.wp_screen_angle_grid) * len(args.wp_screen_sf_grid)
    wp_screen_fits = wp_screen_combos * len(args.wp_screen_folds)
    wp_confirm_fits = f"{int(args.wp_confirm_top_k)} x all valid trial folds"

    print("Dry run only. No data loaded and no files written.")
    print(f"Animals: {args.animals}")
    print(f"Output dir: {output_dir}")
    print(f"GP combos per animal: {gp_combos}")
    print(f"WP screen combos per animal: {wp_screen_combos}")
    print(f"WP screen fits per animal: {wp_screen_fits}")
    print(f"WP confirm fits per animal: {wp_confirm_fits}")
    print(f"WP screen iterations: {args.wp_screen_iterations}")
    print(f"WP confirm iterations: {args.wp_confirm_iterations}")
    for animal in args.animals:
        print(f"Animal {animal}: {result_path_for_animal(output_dir, animal)}")


def main():
    args = parse_args()

    if args.dry_run:
        dry_run(args)
        return

    ensure_model_libraries()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("Loading raw SATED data and metadata.", flush=True)
    datum, angle_arr, sf_arr = load_raw_sated_data(
        args.data_path,
        args.angle_metadata_path,
        args.sf_metadata_path,
    )

    results = []
    for animal in args.animals:
        result = process_animal(animal, datum, angle_arr, sf_arr, args)
        results.append(result)
        write_combined_summaries(args.output_dir, results)

    # Include previously completed selected animals in the final combined summary.
    by_animal = {int(result["animal"]): result for result in results}
    for animal in args.animals:
        if int(animal) in by_animal:
            continue
        path = result_path_for_animal(args.output_dir, animal)
        if path.exists():
            with open(path, "r") as f:
                by_animal[int(animal)] = json.load(f)

    ordered_results = [by_animal[int(animal)] for animal in args.animals if int(animal) in by_animal]
    write_combined_summaries(args.output_dir, ordered_results)


if __name__ == "__main__":
    main()
