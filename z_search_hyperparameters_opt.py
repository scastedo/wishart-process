#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-validated hyperparameter sweeps for the GP/Wishart model.

- mean-angle mode: leave one angle out and score GP interpolation of empirical means
- WP mode: keep all conditions, leave one trial out, and score posterior predictive log prob
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
NUM_ANGLE_PARITY_FOLDS = 2
ANGLE_CV_STRATEGY = "leave_one_angle_out"  # "leave_one_angle_out" or "parity"
TRIAL_HOLDOUT_FOLDS = None  # None means leave each trial out once
TRAIN_TRIAL_PROP = 1.0
TRAIN_CONDITION_PROP = 0.8
EVAL_MODE = "trial_holdout_log_prob"  # "mean_angle_interpolation", "trial_holdout_log_prob", or "posterior_predictive_log_prob"
MEAN_INTERP_SCORE_SOURCE = "empirical_gp_mean"  # "empirical_gp_mean" or "model"

ADAM_STEP = 0.001
ITERATIONS = 30000
NUM_PARTICLES = 1
MC_DRAWS = 20

GAMMA = 1e-5
BETA_WP = 1.0
WP_SAMPLE_DIAG = GAMMA
OPTIMIZE_L = True
PERIOD = 12  # set to None to infer from x

LAMBDA_GRID = {
    "gp_angle": [17.0],
    "gp_sf": [24.0],
    "wp_angle": [18.0, 20.0, 24.0, 30.0],
    "wp_sf": [2.0],
}


SEARCH_STRATEGY = "grid"  # "grid" or "random"
N_RANDOM_SAMPLES = 50
LAMBDA_RANGES = {
    "gp_angle": (0.01, 5.0),
    "gp_sf": (0.01, 5.0),
    "wp_angle": (0.01, 5.0),
    "wp_sf": (0.01,5.0),
}
LAMBDA_SAMPLE = "uniform"  # or "uniform"

P_VALUES = [0]
FULL_GRID_SEARCH = True  # if True, grid search over lambdas x P
P_FOR_LAMBDA_SWEEP = 0
LAMBDA_FOR_P_SWEEP = None  # if dict, use those lambdas; if None, use best combo

OUTPUT_DIR = "outputs"
RESULTS_PATH = "outputs/wp_trial_loo_cv_gp17_24_full_wpangle_next.json"

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


def make_hyperparams(lambdas, p_val, beta_gp):
    return {
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


def split_angle_parity(x_full, y_full, fold_id):
    x_np = np.asarray(x_full)
    unique_angles = np.sort(np.unique(x_np[:, 0]))
    train_angles = unique_angles[int(fold_id)::NUM_ANGLE_PARITY_FOLDS]

    train_mask = np.isin(x_np[:, 0], train_angles)
    test_mask = ~train_mask
    test_angles = unique_angles[~np.isin(unique_angles, train_angles)]

    return (
        x_full[train_mask],
        y_full[:, train_mask, :],
        x_full[test_mask],
        y_full[:, test_mask, :],
        train_angles,
        test_angles,
    )


def split_leave_one_angle_out(x_full, y_full, fold_id):
    x_np = np.asarray(x_full)
    unique_angles = np.sort(np.unique(x_np[:, 0]))
    test_angle = unique_angles[int(fold_id)]
    train_angles = unique_angles[~np.isclose(unique_angles, test_angle)]

    test_mask = np.isclose(x_np[:, 0], test_angle)
    train_mask = ~test_mask

    return (
        x_full[train_mask],
        y_full[:, train_mask, :],
        x_full[test_mask],
        y_full[:, test_mask, :],
        train_angles,
        np.asarray([test_angle]),
    )


def split_mean_angle_interpolation(x_full, y_full, fold_id):
    if ANGLE_CV_STRATEGY == "leave_one_angle_out":
        return split_leave_one_angle_out(x_full, y_full, fold_id)
    if ANGLE_CV_STRATEGY == "parity":
        return split_angle_parity(x_full, y_full, fold_id)
    raise ValueError(f"Unsupported ANGLE_CV_STRATEGY: {ANGLE_CV_STRATEGY}")


def mean_angle_fold_ids(x_full):
    if ANGLE_CV_STRATEGY == "leave_one_angle_out":
        unique_angles = np.sort(np.unique(np.asarray(x_full)[:, 0]))
        return list(range(len(unique_angles)))
    if ANGLE_CV_STRATEGY == "parity":
        return list(range(NUM_ANGLE_PARITY_FOLDS))
    raise ValueError(f"Unsupported ANGLE_CV_STRATEGY: {ANGLE_CV_STRATEGY}")


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


def nearest_angle_baseline(x_train, y_train, x_eval, period):
    x_train_np = np.asarray(x_train)
    x_eval_np = np.asarray(x_eval)
    mu_train = np.asarray(y_train).mean(axis=0)

    preds = []
    for x_row in x_eval_np:
        same_sf = np.isclose(x_train_np[:, 1], x_row[1])
        candidates = np.where(same_sf)[0]
        if candidates.size == 0:
            candidates = np.arange(x_train_np.shape[0])

        distances = circular_angle_distance(x_train_np[candidates, 0], x_row[0], period)
        nearest = candidates[np.isclose(distances, distances.min())]
        preds.append(mu_train[nearest].mean(axis=0))

    return jnp.asarray(np.stack(preds, axis=0))


def nearest_single_angle_baseline(x_train, y_train, x_eval, period):
    x_train_np = np.asarray(x_train)
    x_eval_np = np.asarray(x_eval)
    mu_train = np.asarray(y_train).mean(axis=0)

    preds = []
    for x_row in x_eval_np:
        same_sf = np.isclose(x_train_np[:, 1], x_row[1])
        candidates = np.where(same_sf)[0]
        if candidates.size == 0:
            candidates = np.arange(x_train_np.shape[0])

        distances = circular_angle_distance(x_train_np[candidates, 0], x_row[0], period)
        preds.append(mu_train[candidates[np.argmin(distances)]])

    return jnp.asarray(np.stack(preds, axis=0))


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


def empirical_gp_mean_baseline(x_train, y_train, x_eval, hyperparams, period):
    kernel_gp, _ = build_kernels(hyperparams, period)
    gp = models.GaussianProcess(kernel=kernel_gp, N=y_train.shape[-1])
    mu_train = jnp.asarray(y_train).mean(axis=0)
    return gp.posterior_mode(jnp.asarray(x_train), mu_train, jnp.asarray(x_eval))


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


def mean_interp_score(posterior, x_train, y_train, x_eval, y_eval, hyperparams, period):
    mu_true = jnp.asarray(y_eval).mean(axis=0)
    mu_hat, _, _ = posterior.mode(jnp.asarray(x_eval))
    mu_train_true = jnp.asarray(y_train).mean(axis=0)
    mu_train_hat, _, _ = posterior.mode(jnp.asarray(x_train))

    model_metrics = prediction_metrics(mu_hat, mu_true, y_eval)
    train_model_metrics = prediction_metrics(mu_train_hat, mu_train_true, y_train)
    nearest_metrics = prediction_metrics(
        nearest_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )
    nearest_single_metrics = prediction_metrics(
        nearest_single_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )
    linear_metrics = prediction_metrics(
        circular_linear_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )
    empirical_gp_metrics = prediction_metrics(
        empirical_gp_mean_baseline(x_train, y_train, x_eval, hyperparams, period),
        mu_true,
        y_eval,
    )

    return {
        "score": -model_metrics["mse"],
        "mse": model_metrics["mse"],
        "rmse": model_metrics["rmse"],
        "noise_norm_mse": model_metrics["noise_norm_mse"],
        "corr": model_metrics["corr"],
        "model": model_metrics,
        "train_model": train_model_metrics,
        "model_minus_circular_linear_mse": model_metrics["mse"] - linear_metrics["mse"],
        "empirical_gp_minus_circular_linear_mse": empirical_gp_metrics["mse"] - linear_metrics["mse"],
        "nearest_angle": nearest_metrics,
        "nearest_single_angle": nearest_single_metrics,
        "circular_linear_angle": linear_metrics,
        "empirical_gp_mean": empirical_gp_metrics,
    }


def empirical_gp_mean_interp_score(x_train, y_train, x_eval, y_eval, hyperparams, period):
    mu_true = jnp.asarray(y_eval).mean(axis=0)
    empirical_gp_metrics = prediction_metrics(
        empirical_gp_mean_baseline(x_train, y_train, x_eval, hyperparams, period),
        mu_true,
        y_eval,
    )
    nearest_metrics = prediction_metrics(
        nearest_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )
    nearest_single_metrics = prediction_metrics(
        nearest_single_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )
    linear_metrics = prediction_metrics(
        circular_linear_angle_baseline(x_train, y_train, x_eval, period),
        mu_true,
        y_eval,
    )

    return {
        "score": -empirical_gp_metrics["mse"],
        "mse": empirical_gp_metrics["mse"],
        "rmse": empirical_gp_metrics["rmse"],
        "noise_norm_mse": empirical_gp_metrics["noise_norm_mse"],
        "corr": empirical_gp_metrics["corr"],
        "score_source": "empirical_gp_mean",
        "empirical_gp_minus_circular_linear_mse": empirical_gp_metrics["mse"] - linear_metrics["mse"],
        "nearest_angle": nearest_metrics,
        "nearest_single_angle": nearest_single_metrics,
        "circular_linear_angle": linear_metrics,
        "empirical_gp_mean": empirical_gp_metrics,
    }


def run_cv_for_params(x_full, y_full, lambdas, p_val, fold_ids):
    fold_results = []
    for fold_id in fold_ids:
        if EVAL_MODE == "mean_angle_interpolation":
            x_tr, y_tr, x_test, y_test, train_angles, test_angles = split_mean_angle_interpolation(
                x_full, y_full, fold_id
            )

            x_tr = jnp.asarray(x_tr)
            y_tr = jnp.asarray(y_tr)
            x_test = jnp.asarray(x_test)
            y_test = jnp.asarray(y_test)

            period = PERIOD
            beta_gp = estimate_beta_gp(y_tr)
            hyperparams = make_hyperparams(lambdas, p_val, beta_gp)

            if MEAN_INTERP_SCORE_SOURCE == "empirical_gp_mean":
                metrics = empirical_gp_mean_interp_score(
                    x_tr,
                    y_tr,
                    x_test,
                    y_test,
                    hyperparams,
                    period,
                )
            elif MEAN_INTERP_SCORE_SOURCE == "model":
                fit_seed = SEED + 100 + int(fold_id)
                posterior, _ = fit_posterior(x_tr, y_tr, hyperparams, period, fit_seed)
                metrics = mean_interp_score(posterior, x_tr, y_tr, x_test, y_test, hyperparams, period)
                metrics["score_source"] = "model"
            else:
                raise ValueError(f"Unsupported MEAN_INTERP_SCORE_SOURCE: {MEAN_INTERP_SCORE_SOURCE}")
            score = metrics["score"]
            score_metric = f"negative_{metrics['score_source']}_mse"

            fold_results.append(
                {
                    "fold": int(fold_id),
                    "train_angles": np.asarray(train_angles).tolist(),
                    "test_angles": np.asarray(test_angles).tolist(),
                    "beta_gp": float(beta_gp),
                    "period": int(period),
                    "x_train_shape": list(x_tr.shape),
                    "y_train_shape": list(y_tr.shape),
                    "y_test_shape": list(y_test.shape),
                    "score_mean": float(score),
                    "score_metric": score_metric,
                    "mean_metrics": metrics,
                }
            )
        elif EVAL_MODE == "posterior_predictive_log_prob":
            split = utils.split_data(
                x_full,
                y_full,
                train_trial_prop=TRAIN_TRIAL_PROP,
                train_condition_prop=TRAIN_CONDITION_PROP,
                seed=fold_id,
            )
            x_tr, y_tr, _, _, x_test, y_te, *_ = split
            y_test = y_te["x_test"]

            x_tr = jnp.asarray(x_tr)
            y_tr = jnp.asarray(y_tr)
            y_test = jnp.asarray(y_test)
            x_test = jnp.asarray(x_test)

            period = PERIOD
            beta_gp = estimate_beta_gp(y_tr)
            hyperparams = make_hyperparams(lambdas, p_val, beta_gp)

            posterior, _ = fit_posterior(x_tr, y_tr, hyperparams, period, fold_id)
            ll_stats = mc_log_prob_trials(
                posterior,
                x_test,
                y_test,
                vi_samples=MC_DRAWS,
                gp_samples=1,
                seed=fold_id + 1000,
            )
            score = ll_stats["mean"]
            fold_results.append(
                {
                    "seed": int(fold_id),
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
        elif EVAL_MODE == "trial_holdout_log_prob":
            x_tr, y_tr, x_test, y_test, train_trials, test_trials = split_leave_one_trial_out(
                x_full,
                y_full,
                fold_id,
            )

            x_tr = jnp.asarray(x_tr)
            y_tr = jnp.asarray(y_tr)
            x_test = jnp.asarray(x_test)
            y_test = jnp.asarray(y_test)

            period = PERIOD
            beta_gp = estimate_beta_gp(y_tr)
            hyperparams = make_hyperparams(lambdas, p_val, beta_gp)

            posterior, _ = fit_posterior(x_tr, y_tr, hyperparams, period, fold_id)
            ll_stats = mc_log_prob_trials(
                posterior,
                x_test,
                y_test,
                vi_samples=MC_DRAWS,
                gp_samples=1,
                seed=fold_id + 1000,
            )
            score = ll_stats["mean"]
            fold_results.append(
                {
                    "fold": int(fold_id),
                    "train_trials": np.asarray(train_trials).tolist(),
                    "test_trials": np.asarray(test_trials).tolist(),
                    "beta_gp": float(beta_gp),
                    "period": int(period),
                    "x_train_shape": list(x_tr.shape),
                    "y_train_shape": list(y_tr.shape),
                    "x_test_shape": list(x_test.shape),
                    "y_test_shape": list(y_test.shape),
                    "score_mean": float(score),
                    "score_std": float(ll_stats["std"]),
                    "score_metric": "trial_holdout_log_prob",
                    "score_lp": ll_stats["lp"],
                }
            )
        else:
            raise ValueError(f"Unsupported EVAL_MODE: {EVAL_MODE}")

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

    if EVAL_MODE == "mean_angle_interpolation":
        fold_ids = mean_angle_fold_ids(x_full)
    elif EVAL_MODE == "trial_holdout_log_prob":
        n_trial_folds = y_full.shape[0] if TRIAL_HOLDOUT_FOLDS is None else min(int(TRIAL_HOLDOUT_FOLDS), y_full.shape[0])
        fold_ids = list(range(n_trial_folds))
    else:
        fold_ids = [SEED + i for i in range(NUM_FOLDS)]

    combo_results = []

    if SEARCH_STRATEGY == "grid":
        for lambdas in iter_lambda_combos(LAMBDA_GRID):
            p_iter = P_VALUES if FULL_GRID_SEARCH else [P_FOR_LAMBDA_SWEEP]
            for p_val in p_iter:
                folds = run_cv_for_params(x_full, y_full, lambdas, p_val, fold_ids)
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
            folds = run_cv_for_params(x_full, y_full, lambdas, p_val, fold_ids)
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

    if EVAL_MODE == "mean_angle_interpolation":
        score_metric = f"negative_{MEAN_INTERP_SCORE_SOURCE}_mse"
    elif EVAL_MODE == "trial_holdout_log_prob":
        score_metric = "trial_holdout_log_prob"
    else:
        score_metric = "posterior_predictive_log_prob"

    p_entries = None
    lambda_for_p = None
    if SEARCH_STRATEGY == "grid" and not FULL_GRID_SEARCH and EVAL_MODE == "posterior_predictive_log_prob":
        best_lambda = best_combo["lambdas"] if best_combo else None
        if isinstance(LAMBDA_FOR_P_SWEEP, dict):
            lambda_for_p = LAMBDA_FOR_P_SWEEP
        else:
            lambda_for_p = best_lambda
        if lambda_for_p is None:
            lambda_for_p = {k: float(v[0]) for k, v in LAMBDA_GRID.items()}

        p_entries = []
        for p_val in P_VALUES:
            folds = run_cv_for_params(x_full, y_full, lambda_for_p, p_val, fold_ids)
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
        "p_sweep": None if p_entries is None else {
            "entries": p_entries,
            "lambda_fixed": lambda_for_p,
        },
        "best_combo": best_combo,
        "best_lambda": None if best_combo is None else best_combo["lambdas"],
        "best_p": None if best_combo is None else best_combo["p"],
        "fold_ids": fold_ids,
        "config": {
            "data_path": DATA_PATH,
            "eval_mode": EVAL_MODE,
            "score_metric": score_metric,
            "mean_interp_score_source": MEAN_INTERP_SCORE_SOURCE,
            "train_trial_prop": TRAIN_TRIAL_PROP,
            "train_condition_prop": TRAIN_CONDITION_PROP,
            "num_folds": NUM_FOLDS,
            "num_angle_parity_folds": NUM_ANGLE_PARITY_FOLDS,
            "angle_cv_strategy": ANGLE_CV_STRATEGY,
            "num_angle_folds": len(fold_ids) if EVAL_MODE == "mean_angle_interpolation" else None,
            "trial_holdout_folds": TRIAL_HOLDOUT_FOLDS,
            "num_trial_holdout_folds": len(fold_ids) if EVAL_MODE == "trial_holdout_log_prob" else None,
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
