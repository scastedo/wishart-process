#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate searched hyperparameters, fit the best on full data, and run diagnostics.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro import optim
import yaml

import inference
import models
import utils
import visualizations


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return cfg


def _load_results(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_npz(npz_path: str, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if x_key not in data or y_key not in data:
        raise KeyError(f"Expected keys {x_key!r} and {y_key!r} in {npz_path}")
    return np.asarray(data[x_key]), np.asarray(data[y_key])


def _standardize_x(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim != 2:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")
    return x


def _kernel_value(
    a: jnp.ndarray,
    b: jnp.ndarray,
    kernel_type: str,
    gamma: float,
    beta: float,
    lam: float,
    period: float,
) -> jnp.ndarray:
    kernel_type = kernel_type.lower()
    if kernel_type == "periodic":
        base = jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2
        return gamma * (a == b) + beta * jnp.exp(-base / lam)
    if kernel_type == "rbf":
        return gamma * (a == b) + beta * jnp.exp(-((a - b) ** 2) / lam)
    raise ValueError(f"Unsupported kernel type: {kernel_type}")


def _build_kernel(
    dim_specs: List[Dict[str, Any]],
    hp: Dict[str, Dict[str, Dict[str, float]]],
    kind: str,
    default_periods: Dict[str, float],
):
    def kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        out = 1.0
        for dim_idx, dim in enumerate(dim_specs):
            name = dim["name"]
            ktype = dim["type"]
            period = dim.get("period", default_periods[name])
            params = hp[kind][name]
            out = out * _kernel_value(
                x[dim_idx],
                y[dim_idx],
                ktype,
                params["gamma"],
                params["beta"],
                params["lambda"],
                period,
            )
        return out

    return kernel


def _default_periods(x: np.ndarray, dim_specs: List[Dict[str, Any]]) -> Dict[str, float]:
    periods = {}
    for dim_idx, dim in enumerate(dim_specs):
        name = dim["name"]
        if "period" in dim and dim["period"] is not None:
            periods[name] = float(dim["period"])
            continue
        values = np.asarray(x[:, dim_idx])
        periods[name] = float(values.max() - values.min() + 1.0)
    return periods


def _init_scale_from_data(y_train: jnp.ndarray, jitter_frac: float) -> jnp.ndarray:
    y_flat = y_train.reshape(-1, y_train.shape[-1])
    y_flat = y_flat - y_flat.mean(0, keepdims=True)
    sigma_bar = (y_flat.T @ y_flat) / max(y_flat.shape[0] - 1, 1)
    eps = jitter_frac * jnp.trace(sigma_bar) / sigma_bar.shape[0]
    return jnp.linalg.cholesky(sigma_bar + eps * jnp.eye(sigma_bar.shape[0]))


def _mc_log_lik(
    posterior: models.NormalGaussianWishartPosterior,
    likelihood,
    x_eval: jnp.ndarray,
    y_eval: jnp.ndarray,
    n_mc: int,
    seed: int,
) -> float:
    def one_draw(rng_key):
        with numpyro.handlers.seed(rng_seed=rng_key):
            mu, sigma, _ = posterior.sample(x_eval)
        return likelihood.log_prob(y_eval, mu, sigma).mean()

    keys = jax.random.split(jax.random.PRNGKey(seed), n_mc)
    return float(jax.vmap(one_draw)(keys).mean())


def _empirical_stats(y: np.ndarray, jitter: float) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    mu = y.mean(0)
    c = y.shape[1]
    n = y.shape[2]
    sigma = np.zeros((c, n, n), dtype=float)
    for i in range(c):
        cov = np.cov(y[:, i, :], rowvar=False)
        cov = cov + jitter * np.eye(n)
        sigma[i] = cov
    return mu, sigma


def _cov_error_stats(sigma_pred: np.ndarray, sigma_emp: np.ndarray) -> Dict[str, float]:
    diffs_op = []
    diffs_fro = []
    for i in range(sigma_pred.shape[0]):
        diffs_op.append(np.linalg.norm(sigma_pred[i] - sigma_emp[i], ord=2))
        diffs_fro.append(np.linalg.norm(sigma_pred[i] - sigma_emp[i], ord="fro"))
    return {
        "op_mean": float(np.mean(diffs_op)),
        "op_median": float(np.median(diffs_op)),
        "fro_mean": float(np.mean(diffs_fro)),
        "fro_median": float(np.median(diffs_fro)),
    }


def _mean_mse(mu_pred: np.ndarray, mu_emp: np.ndarray) -> float:
    return float(np.mean(np.sum((mu_pred - mu_emp) ** 2, axis=1)))


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    params = copy.deepcopy(params)
    model = params.setdefault("model", {})
    if "P" in model:
        try:
            model["P"] = int(model["P"])
        except ValueError:
            model["P"] = int(float(model["P"]))
    return params


def _fit_model(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    dim_specs: List[Dict[str, Any]],
    default_periods: Dict[str, float],
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    seed: int,
) -> Tuple[models.JointGaussianWishartProcess, Any, models.NormalGaussianWishartPosterior, Any]:
    n = y_train.shape[-1]

    kernel_gp = _build_kernel(dim_specs, params, "gp", default_periods)
    kernel_wp = _build_kernel(dim_specs, params, "wp", default_periods)

    likelihood_name = cfg["model"].get("likelihood", "NormalConditionalLikelihood")
    likelihood = getattr(models, likelihood_name)(n)

    process_name = cfg["model"].get("process", "WishartLRDProcess")
    p_val = int(params["model"].get("P", cfg["model"].get("P", 1)))
    v_scale = float(params["model"].get("V_scale", cfg["model"].get("V_scale", 1e-1)))
    diag_scale = float(params["model"].get("diag_scale", cfg["model"].get("diag_scale", 1.0)))
    optimize_l = bool(cfg["model"].get("optimize_L", True))

    gp = models.GaussianProcess(kernel=kernel_gp, N=n)
    wp_class = getattr(models, process_name)
    wp = wp_class(
        kernel=kernel_wp,
        P=p_val,
        V=v_scale * jnp.eye(n),
        optimize_L=optimize_l,
        diag_scale=diag_scale,
    )

    init_l = _init_scale_from_data(y_train, cfg["model"].get("init_L_jitter_frac", 1e-3))
    wp.L = init_l

    joint = models.JointGaussianWishartProcess(gp, wp, likelihood)

    guide_name = cfg["inference"].get("guide", "VariationalNormal")
    guide_class = getattr(inference, guide_name)

    init = None
    if likelihood_name == "NormalConditionalLikelihood":
        init = {"G": y_train.mean(0).T[:, None]}
    if likelihood_name == "PoissonConditionalLikelihood":
        likelihood.initialize_rate(y_train)
        init_g = likelihood.gain_inverse_fn(y_train.mean(0).T[:, None]) - likelihood.rate[:, None, None]
        init = {"G": init_g, "g": init_g.transpose(1, 2, 0).repeat(y_train.shape[0], 0)}

    guide = guide_class(joint.model, init=init)
    optimizer = optim.Adam(float(cfg["inference"].get("step_size", 1e-2)))
    guide.infer(
        optimizer,
        x_train,
        y_train,
        n_iter=int(cfg["inference"].get("n_iter", 15000)),
        key=jax.random.PRNGKey(seed),
        num_particles=int(cfg["inference"].get("num_particles", 1)),
    )

    joint.update_params(guide.posterior)
    posterior = models.NormalGaussianWishartPosterior(joint, guide, x_train)
    return joint, guide, posterior, likelihood


def _eval_split_loglik(
    posterior: models.NormalGaussianWishartPosterior,
    likelihood,
    x_train: jnp.ndarray,
    y_val: Dict[str, jnp.ndarray],
    x_val: jnp.ndarray,
    eval_cfg: Dict[str, Any],
) -> float:
    split_mode = eval_cfg.get("split", "conditions").lower()
    n_mc = int(eval_cfg.get("n_mc", 16))
    seed = int(eval_cfg.get("seed", 0))

    if split_mode == "conditions":
        return _mc_log_lik(posterior, likelihood, x_val, y_val["x_test"], n_mc, seed)
    if split_mode == "trials":
        return _mc_log_lik(posterior, likelihood, x_train, y_val["x"], n_mc, seed)
    if split_mode == "both":
        ll_a = _mc_log_lik(posterior, likelihood, x_val, y_val["x_test"], n_mc, seed)
        ll_b = _mc_log_lik(posterior, likelihood, x_train, y_val["x"], n_mc, seed + 1)
        return 0.5 * (ll_a + ll_b)
    raise ValueError(f"Unsupported eval split: {split_mode}")


def _predict_with_jitter(
    joint: models.JointGaussianWishartProcess,
    guide,
    x_train: jnp.ndarray,
    x_eval: jnp.ndarray,
    base: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    def gp_predict_jitter(gp, X, Y, x, base_scale):
        K_X_X = gp.evaluate_kernel(X, X)
        scale = float(jnp.mean(jnp.diag(K_X_X)))
        jitter = base_scale * max(scale, 1.0)
        K_X_X = K_X_X + jitter * jnp.eye(len(X))
        K_X_x = gp.evaluate_kernel(x, X)
        K_x_x = gp.evaluate_kernel(x, x) + jitter * jnp.eye(len(x))

        Ki = jnp.linalg.inv(K_X_X)
        f = jnp.einsum("ij,jm->mi", K_X_x.T @ Ki, Y)
        K = K_x_x - K_X_x.T @ Ki @ K_X_x + jitter * jnp.eye(len(x))
        return f, K

    def wp_predict_jitter(wp, X, F, x, base_scale, rng_key):
        K_X_X = wp.evaluate_kernel(X, X)
        scale = float(jnp.mean(jnp.diag(K_X_X)))
        jitter = base_scale * max(scale, 1.0)
        K_X_X = K_X_X + jitter * jnp.eye(len(X))
        K_X_x = wp.evaluate_kernel(x, X)
        K_x_x = wp.evaluate_kernel(x, x) + jitter * jnp.eye(len(x))

        Ki = jnp.linalg.inv(K_X_X)
        f = jnp.einsum("ij,mnj->mni", K_X_x.T @ Ki, F)
        K = K_x_x - K_X_x.T @ Ki @ K_X_x + jitter * jnp.eye(len(x))
        F_new = numpyro.distributions.MultivariateNormal(f, covariance_matrix=K).sample(rng_key)
        sigma = wp.f2sigma(F_new)
        return F_new, sigma

    with numpyro.handlers.seed(rng_seed=seed):
        F_draw, G_draw = guide.sample()

    mu_hat, _ = gp_predict_jitter(joint.gp, x_train, G_draw.squeeze().T, x_eval, base)
    mu_hat = np.asarray(mu_hat).T
    _, sigma_hat = wp_predict_jitter(joint.wp, x_train, F_draw, x_eval, base, jax.random.PRNGKey(seed + 1))
    return mu_hat, np.asarray(sigma_hat)


def _predict_posterior(
    posterior: models.NormalGaussianWishartPosterior,
    joint: models.JointGaussianWishartProcess,
    guide,
    x_train: jnp.ndarray,
    x_eval: jnp.ndarray,
    use_jitter: bool,
    jitter_base: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if use_jitter:
        return _predict_with_jitter(joint, guide, x_train, x_eval, jitter_base, seed)
    with numpyro.handlers.seed(rng_seed=seed):
        mu, sigma, _ = posterior.sample(x_eval)
    return np.asarray(mu), np.asarray(sigma)


def _split_train_val_test(
    x: np.ndarray,
    y: np.ndarray,
    split_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    test_cfg = split_cfg["test"]
    val_cfg = split_cfg["val"]

    x_train, y_train, _, _, x_test, y_test, *_ = utils.split_data(
        x,
        y,
        train_trial_prop=test_cfg["train_trial_prop"],
        train_condition_prop=test_cfg["train_condition_prop"],
        seed=test_cfg["seed"],
    )

    x_train, y_train, _, _, x_val, y_val, *_ = utils.split_data(
        x_train,
        y_train,
        train_trial_prop=val_cfg["train_trial_prop"],
        train_condition_prop=val_cfg["train_condition_prop"],
        seed=val_cfg["seed"],
    )

    return x_train, y_train, x_val, y_val, x_test, y_test


def _reshape_to_grid(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    angle_vals = np.sort(np.unique(x[:, 0]))
    sf_vals = np.sort(np.unique(x[:, 1]))
    c1 = len(angle_vals)
    c2 = len(sf_vals)
    n = y.shape[2]
    k = y.shape[0]

    mu_grid = np.zeros((c1, c2, n), dtype=float)
    sigma_grid = np.zeros((c1, c2, n, n), dtype=float)
    y_grid = np.zeros((k, c1, c2, n), dtype=float)

    angle_to_idx = {val: i for i, val in enumerate(angle_vals)}
    sf_to_idx = {val: i for i, val in enumerate(sf_vals)}

    for c in range(x.shape[0]):
        ai = angle_to_idx[x[c, 0]]
        si = sf_to_idx[x[c, 1]]
        mu_grid[ai, si] = mu[c]
        sigma_grid[ai, si] = sigma[c]
        y_grid[:, ai, si] = y[:, c]

    return angle_vals, sf_vals, mu_grid, sigma_grid, y_grid


def _plot_eigenvalues(
    sigma_pred: np.ndarray,
    sigma_emp: np.ndarray,
    top_k: int,
    out_path: Path,
) -> None:
    evals_pred = []
    evals_emp = []
    for i in range(sigma_pred.shape[0]):
        ev_pred = np.linalg.eigh(sigma_pred[i])[0][::-1]
        ev_emp = np.linalg.eigh(sigma_emp[i])[0][::-1]
        evals_pred.append(ev_pred[:top_k])
        evals_emp.append(ev_emp[:top_k])

    avg_pred = np.mean(np.stack(evals_pred, axis=0), axis=0)
    avg_emp = np.mean(np.stack(evals_emp, axis=0), axis=0)

    plt.figure(figsize=(6, 4))
    plt.scatter(np.arange(len(avg_emp)), avg_emp, label="empirical", s=12)
    plt.scatter(np.arange(len(avg_pred)), avg_pred, label="wishart", s=12)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.title("Top eigenvalues (avg over angles)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path) + ".png", dpi=150)
    plt.savefig(str(out_path) + ".pdf")
    plt.close("all")


def _select_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    for item in results:
        val = item.get("val_loglik")
        if val is None or not np.isfinite(val):
            continue
        if best is None or val > best["val_loglik"]:
            best = item
    if best is None:
        raise ValueError("No finite val_loglik found in results.")
    return best


def _evaluate_candidates(
    candidates: List[Dict[str, Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: Dict[str, np.ndarray],
    dim_specs: List[Dict[str, Any]],
    default_periods: Dict[str, float],
    cfg: Dict[str, Any],
    seed_base: int,
) -> List[Dict[str, Any]]:
    results = []
    for idx, cand in enumerate(candidates):
        params = _normalize_params(cand["params"])
        run_seed = seed_base + idx
        try:
            joint, guide, posterior, likelihood = _fit_model(
                jnp.asarray(x_train),
                jnp.asarray(y_train),
                dim_specs,
                default_periods,
                params,
                cfg,
                run_seed,
            )
            val_ll = _eval_split_loglik(
                posterior,
                likelihood,
                jnp.asarray(x_train),
                {k: jnp.asarray(v) for k, v in y_val.items()},
                jnp.asarray(x_val),
                cfg["eval"],
            )
            result = {
                "id": cand.get("id", idx),
                "val_loglik": float(val_ll),
                "params": params,
                "status": "ok",
            }
        except Exception as exc:
            result = {
                "id": cand.get("id", idx),
                "val_loglik": None,
                "params": params,
                "status": "fail",
                "error": str(exc),
            }
        results.append(result)
    return results


def _run_holdout_checks(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    dim_specs: List[Dict[str, Any]],
    default_periods: Dict[str, float],
    output_dir: Path,
    use_jitter: bool,
    jitter_base: float,
    seed: int,
) -> Dict[str, Any]:
    test_cfg = cfg["splits"]["test"]
    x_train, y_train, _, _, x_test, y_test, *_ = utils.split_data(
        x,
        y,
        train_trial_prop=test_cfg["train_trial_prop"],
        train_condition_prop=test_cfg["train_condition_prop"],
        seed=test_cfg["seed"],
    )

    joint, guide, posterior, likelihood = _fit_model(
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        dim_specs,
        default_periods,
        params,
        cfg,
        seed,
    )

    mu_train, sigma_train = _predict_posterior(
        posterior,
        joint,
        guide,
        jnp.asarray(x_train),
        jnp.asarray(x_train),
        use_jitter,
        jitter_base,
        seed,
    )
    mu_test, sigma_test = _predict_posterior(
        posterior,
        joint,
        guide,
        jnp.asarray(x_train),
        jnp.asarray(x_test),
        use_jitter,
        jitter_base,
        seed + 1,
    )

    mu_emp_train, sigma_emp_train = _empirical_stats(y_test["x"], jitter=1e-6)
    mu_emp_test, sigma_emp_test = _empirical_stats(y_test["x_test"], jitter=1e-6)

    holdout = {
        "trials": {
            "cov_error": _cov_error_stats(sigma_train, sigma_emp_train),
            "mean_mse": _mean_mse(mu_train, mu_emp_train),
            "loglik_wishart": float(
                _mc_log_lik(
                    posterior,
                    likelihood,
                    jnp.asarray(x_train),
                    jnp.asarray(y_test["x"]),
                    int(cfg["eval"].get("n_mc", 16)),
                    int(cfg["eval"].get("seed", 0)),
                )
            ),
            "loglik_empirical": float(
                likelihood.log_prob(
                    jnp.asarray(y_test["x"]),
                    jnp.asarray(mu_emp_train),
                    jnp.asarray(sigma_emp_train),
                ).mean()
            ),
        },
        "conditions": {
            "cov_error": _cov_error_stats(sigma_test, sigma_emp_test),
            "mean_mse": _mean_mse(mu_test, mu_emp_test),
            "loglik_wishart": float(
                _mc_log_lik(
                    posterior,
                    likelihood,
                    jnp.asarray(x_test),
                    jnp.asarray(y_test["x_test"]),
                    int(cfg["eval"].get("n_mc", 16)),
                    int(cfg["eval"].get("seed", 0)) + 1,
                )
            ),
            "loglik_empirical": float(
                likelihood.log_prob(
                    jnp.asarray(y_test["x_test"]),
                    jnp.asarray(mu_emp_test),
                    jnp.asarray(sigma_emp_test),
                ).mean()
            ),
        },
    }

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_cfg = cfg.get("analysis", {})
    pc_std = float(analysis_cfg.get("pc_std", 0.1))
    pc_dotsize = float(analysis_cfg.get("pc_dotsize", 8))

    visualizations.visualize_pc(
        mu_test[:, None],
        sigma_test,
        pc=y_test["x_test"].reshape(-1, y.shape[2]),
        title_str="Holdout conditions (wishart)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "holdout_conditions_wishart"),
    )
    visualizations.visualize_pc(
        mu_emp_test[:, None],
        sigma_emp_test,
        pc=y_test["x_test"].reshape(-1, y.shape[2]),
        title_str="Holdout conditions (empirical)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "holdout_conditions_empirical"),
    )

    visualizations.visualize_pc(
        mu_train[:, None],
        sigma_train,
        pc=y_test["x"].reshape(-1, y.shape[2]),
        title_str="Holdout trials (wishart)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "holdout_trials_wishart"),
    )
    visualizations.visualize_pc(
        mu_emp_train[:, None],
        sigma_emp_train,
        pc=y_test["x"].reshape(-1, y.shape[2]),
        title_str="Holdout trials (empirical)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "holdout_trials_empirical"),
    )

    return holdout


def _run_full_fit(
    x: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    dim_specs: List[Dict[str, Any]],
    default_periods: Dict[str, float],
    output_dir: Path,
    use_jitter: bool,
    jitter_base: float,
    seed: int,
) -> Dict[str, Any]:
    cfg_full = copy.deepcopy(cfg)
    if "full_fit" in cfg_full:
        cfg_full["inference"].update(cfg_full["full_fit"])

    joint, guide, posterior, _ = _fit_model(
        jnp.asarray(x),
        jnp.asarray(y),
        dim_specs,
        default_periods,
        params,
        cfg_full,
        seed,
    )

    mu_hat, sigma_hat = _predict_posterior(
        posterior,
        joint,
        guide,
        jnp.asarray(x),
        jnp.asarray(x),
        use_jitter,
        jitter_base,
        seed,
    )

    mu_emp, sigma_emp = _empirical_stats(y, jitter=1e-6)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    analysis_cfg = cfg.get("analysis", {})
    pc_std = float(analysis_cfg.get("pc_std", 0.1))
    pc_dotsize = float(analysis_cfg.get("pc_dotsize", 8))

    visualizations.visualize_pc(
        mu_hat[:, None],
        sigma_hat,
        pc=y.reshape(-1, y.shape[2]),
        title_str="Full data (wishart)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "full_wishart_pc"),
    )

    visualizations.visualize_pc(
        mu_emp[:, None],
        sigma_emp,
        pc=y.reshape(-1, y.shape[2]),
        title_str="Full data (empirical)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "full_empirical_pc"),
    )

    angles, sfs, mu_grid, sigma_grid, y_grid = _reshape_to_grid(x, mu_hat, sigma_hat, y)
    _, _, mu_emp_grid, sigma_emp_grid, _ = _reshape_to_grid(x, mu_emp, sigma_emp, y)

    mu_ring = mu_grid.mean(axis=1)
    sigma_ring = sigma_grid.mean(axis=1)
    y_ring = y_grid.mean(axis=2)
    ring_pc = y_ring.reshape(y_ring.shape[0] * y_ring.shape[1], -1)

    visualizations.visualize_pc(
        mu_ring[:, None],
        sigma_ring,
        pc=ring_pc,
        title_str="Angles ring (avg over SF)",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / "angles_ring_pc"),
    )

    sf_index = int(analysis_cfg.get("sf_index", 0))
    if "sf_value" in analysis_cfg:
        sf_value = analysis_cfg["sf_value"]
        sf_idx_match = np.where(np.isclose(sfs, sf_value))[0]
        if len(sf_idx_match) > 0:
            sf_index = int(sf_idx_match[0])

    sf_index = max(0, min(sf_index, len(sfs) - 1))

    mu_sf = mu_grid[:, sf_index, :]
    sigma_sf = sigma_grid[:, sf_index, :, :]
    y_sf = y_grid[:, :, sf_index, :]
    pc_sf = y_sf.reshape(y_sf.shape[0] * y_sf.shape[1], -1)

    visualizations.visualize_pc(
        mu_sf[:, None],
        sigma_sf,
        pc=pc_sf,
        title_str=f"Angles at SF index {sf_index}",
        dotsize=pc_dotsize,
        std=pc_std,
        save=True,
        file=str(plots_dir / f"angles_pc_sf_{sf_index}"),
    )

    sigma_emp_sf = sigma_emp_grid[:, sf_index, :, :]
    top_k = int(analysis_cfg.get("eig_top_k", min(50, sigma_emp_sf.shape[-1])))
    _plot_eigenvalues(
        sigma_sf,
        sigma_emp_sf,
        top_k=top_k,
        out_path=plots_dir / f"eigvals_sf_{sf_index}",
    )

    return {
        "angles": angles.tolist(),
        "sfs": sfs.tolist(),
        "sf_index": sf_index,
        "eig_top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hyperparams and run diagnostics")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--results", required=True, help="Search results JSON path")
    parser.add_argument("--output-dir", default="outputs/hp_eval", help="Output directory")
    parser.add_argument("--reuse-results", action="store_true", help="Skip re-evaluation and reuse val_logliks")
    parser.add_argument("--max-candidates", type=int, default=None, help="Limit candidates to evaluate")
    parser.add_argument("--use-jitter", action="store_true", help="Use jittered predictive sampling")
    parser.add_argument("--jitter-base", type=float, default=1e-12, help="Base jitter scale")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    results_json = _load_results(args.results)

    data_cfg = cfg["data"]
    x, y = _load_npz(data_cfg["npz_path"], data_cfg.get("x_key", "x"), data_cfg.get("y_key", "y"))
    x = _standardize_x(x)

    dim_specs = cfg["kernels"]["dims"]
    if x.shape[1] != len(dim_specs):
        raise ValueError(f"x has {x.shape[1]} dims, but kernels.dims has {len(dim_specs)} entries")

    default_periods = _default_periods(x, dim_specs)

    candidates = results_json.get("results", [])
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    if not candidates:
        raise ValueError("No candidates found in results JSON.")

    x_train, y_train, x_val, y_val, _, _ = _split_train_val_test(x, y, cfg["splits"])

    if args.reuse_results:
        eval_results = []
        for cand in candidates:
            eval_results.append(
                {
                    "id": cand.get("id"),
                    "val_loglik": cand.get("val_loglik"),
                    "params": _normalize_params(cand.get("params", {})),
                    "status": cand.get("status", "ok"),
                }
            )
    else:
        eval_results = _evaluate_candidates(
            candidates,
            x_train,
            y_train,
            x_val,
            y_val,
            dim_specs,
            default_periods,
            cfg,
            seed_base=int(cfg.get("run_seed", 0)),
        )

    best = _select_best(eval_results)
    best_params = _normalize_params(best["params"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holdout = _run_holdout_checks(
        x,
        y,
        best_params,
        cfg,
        dim_specs,
        default_periods,
        output_dir,
        args.use_jitter,
        args.jitter_base,
        seed=int(cfg.get("run_seed", 0)),
    )

    full_analysis = _run_full_fit(
        x,
        y,
        best_params,
        cfg,
        dim_specs,
        default_periods,
        output_dir,
        args.use_jitter,
        args.jitter_base,
        seed=int(cfg.get("run_seed", 0)) + 10,
    )

    summary = {
        "best": best,
        "holdout": holdout,
        "full_analysis": full_analysis,
        "eval_results": eval_results,
    }

    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
