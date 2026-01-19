#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone hyperparameter search for Wishart/GP models using held-out log-likelihood.

This follows the paper's kernel forms (gamma/beta/lambda) and cross-validates
on held-out log-likelihood to reduce overfitting. The default process is the
low-rank-plus-diagonal Wishart (WishartLRDProcess), with L initialized from the
grand empirical covariance and optionally optimized.
"""
import os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import optim
import yaml

import inference
import models
import utils


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


def _init_scale_from_data(y_train: jnp.ndarray, jitter_frac: float) -> jnp.ndarray:
    y_flat = y_train.reshape(-1, y_train.shape[-1])
    y_flat = y_flat - y_flat.mean(0, keepdims=True)
    sigma_bar = (y_flat.T @ y_flat) / max(y_flat.shape[0] - 1, 1)
    eps = jitter_frac * jnp.trace(sigma_bar) / sigma_bar.shape[0]
    return jnp.linalg.cholesky(sigma_bar + eps * jnp.eye(sigma_bar.shape[0],dtype= jnp.float64))


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


def _spec_to_values(spec: Dict[str, Any]) -> List[Any]:
    if "values" in spec:
        return spec["values"]
    raise ValueError("Grid search requires 'values' for every parameter.")


def _sample_spec(spec: Dict[str, Any], rng: np.random.Generator) -> Any:
    if "values" in spec:
        return rng.choice(spec["values"])

    dist = spec.get("dist", "loguniform")
    low = float(spec["low"])
    high = float(spec["high"])
    if dist == "loguniform":
        if low <= 0 or high <= 0:
            raise ValueError("loguniform requires low/high > 0")
        return float(10 ** rng.uniform(math.log10(low), math.log10(high)))
    if dist == "uniform":
        return float(rng.uniform(low, high))
    if dist == "int":
        return int(rng.integers(int(low), int(high) + 1))
    raise ValueError(f"Unsupported dist: {dist}")


def _collect_param_specs(cfg: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    specs = {}
    for dim in cfg["kernels"]["dims"]:
        dim_name = dim["name"]
        for kind in ("gp", "wp"):
            if kind not in dim:
                raise KeyError(f"Missing {kind} in kernel spec for dim {dim_name}")
            for key in ("gamma", "beta", "lambda"):
                if key not in dim[kind]:
                    raise KeyError(f"Missing {key} in {kind} spec for dim {dim_name}")
                specs[(kind, dim_name, key)] = dim[kind][key]

    for key, spec in cfg.get("search", {}).get("model", {}).items():
        specs[("model", key, "value")] = spec

    return specs


def _params_from_values(
    keys: List[Tuple[str, str, str]], values: Iterable[Any]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    params = {"gp": {}, "wp": {}, "model": {}}
    for (group, name, key), value in zip(keys, values):
        if group == "model":
            params["model"][name] = value
        else:
            params[group].setdefault(name, {})[key] = value
    return params


def _generate_candidates(cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    specs = _collect_param_specs(cfg)
    keys = list(specs.keys())

    strategy = cfg.get("search", {}).get("strategy", "random").lower()
    if strategy == "grid":
        value_lists = [_spec_to_values(specs[key]) for key in keys]
        for combo in itertools.product(*value_lists):
            yield _params_from_values(keys, combo)
        return

    n_samples = int(cfg.get("search", {}).get("n_samples", 25))
    seed = int(cfg.get("search", {}).get("seed", 0))
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        values = [_sample_spec(specs[key], rng) for key in keys]
        yield _params_from_values(keys, values)


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


def _eval_split(
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


def _fit_and_score(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_val: jnp.ndarray,
    y_val: Dict[str, jnp.ndarray],
    dim_specs: List[Dict[str, Any]],
    default_periods: Dict[str, float],
    params: Dict[str, Any],
    cfg: Dict[str, Any],
    run_seed: int,
) -> float:
    n = y_train.shape[-1]

    kernel_gp = _build_kernel(dim_specs, params, "gp", default_periods)
    kernel_wp = _build_kernel(dim_specs, params, "wp", default_periods)

    likelihood_name = cfg["model"].get("likelihood", "NormalConditionalLikelihood")
    likelihood = eval(f"models.{likelihood_name}")(n)

    process_name = cfg["model"].get("process", "WishartLRDProcess")
    p_val = int(params["model"].get("P", cfg["model"].get("P", 1)))
    v_scale = float(params["model"].get("V_scale", cfg["model"].get("V_scale", 1e-1)))
    diag_scale = float(params["model"].get("diag_scale", cfg["model"].get("diag_scale", 1.0)))
    optimize_l = bool(cfg["model"].get("optimize_L", True))

    gp = models.GaussianProcess(kernel=kernel_gp, N=n)
    wp = eval(f"models.{process_name}")(
        kernel=kernel_wp,
        P=p_val,
        V=v_scale * jnp.eye(n, dtype=jnp.float64),
        optimize_L=optimize_l,
        diag_scale=diag_scale,
    )

    init_l = _init_scale_from_data(y_train, cfg["model"].get("init_L_jitter_frac", 1e-3))
    wp.L = init_l

    joint = models.JointGaussianWishartProcess(gp, wp, likelihood)

    guide_name = cfg["inference"].get("guide", "VariationalNormal")
    init = None
    if likelihood_name == "NormalConditionalLikelihood":
        init = {"G": y_train.mean(0).T[:, None]}
    if likelihood_name == "PoissonConditionalLikelihood":
        likelihood.initialize_rate(y_train)
        init_g = likelihood.gain_inverse_fn(y_train.mean(0).T[:, None]) - likelihood.rate[:, None, None]
        init = {"G": init_g, "g": init_g.transpose(1, 2, 0).repeat(y_train.shape[0], 0)}

    guide = eval(f"inference.{guide_name}")(joint.model, init=init)

    optimizer = optim.Adam(float(cfg["inference"].get("step_size", 1e-2)))
    guide.infer(
        optimizer,
        x_train,
        y_train,
        n_iter=int(cfg["inference"].get("n_iter", 15000)),
        key=jax.random.PRNGKey(run_seed),
        num_particles=int(cfg["inference"].get("num_particles", 1)),
    )

    joint.update_params(guide.posterior)
    posterior = models.NormalGaussianWishartPosterior(joint, guide, x_train)

    return _eval_split(posterior, likelihood, x_train, y_val, x_val, cfg["eval"])


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    return cfg


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


def run_search(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = cfg["data"]
    x, y = _load_npz(data_cfg["npz_path"], data_cfg.get("x_key", "x"), data_cfg.get("y_key", "y"))
    x = _standardize_x(x)

    dim_specs = cfg["kernels"]["dims"]
    if x.shape[1] != len(dim_specs):
        raise ValueError(f"x has {x.shape[1]} dims, but kernels.dims has {len(dim_specs)} entries")

    x_train, y_train, x_val, y_val, x_test, y_test = _split_train_val_test(
        x, y, cfg["splits"]
    )

    default_periods = _default_periods(x_train, dim_specs)

    results = []
    best = None
    base_seed = int(cfg.get("run_seed", 0))

    for idx, params in enumerate(_generate_candidates(cfg)):
        run_seed = base_seed + idx
        try:
            score = _fit_and_score(
                jnp.asarray(x_train),
                jnp.asarray(y_train),
                jnp.asarray(x_val),
                {k: jnp.asarray(v) for k, v in y_val.items()},
                dim_specs,
                default_periods,
                params,
                cfg,
                run_seed,
            )
            result = {"id": idx, "val_loglik": score, "params": params, "status": "ok"}
        except Exception as exc:
            result = {"id": idx, "val_loglik": None, "params": params, "status": "fail", "error": str(exc)}

        results.append(result)
        if result["status"] == "ok" and (best is None or result["val_loglik"] > best["val_loglik"]):
            best = result

    test_loglik = None
    if cfg.get("eval", {}).get("run_test", False) and best is not None and len(x_test) > 0:
        test_loglik = _fit_and_score(
            jnp.asarray(x_train),
            jnp.asarray(y_train),
            jnp.asarray(x_test),
            {k: jnp.asarray(v) for k, v in y_test.items()},
            dim_specs,
            default_periods,
            best["params"],
            cfg,
            base_seed + len(results) + 1,
        )

    output = {
        "best": best,
        "results": results,
        "test_loglik": test_loglik,
        "data_summary": {
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "x_train_shape": list(x_train.shape),
            "y_train_shape": list(y_train.shape),
            "x_val_shape": list(x_val.shape),
            "x_test_shape": list(x_test.shape),
        },
    }

    return _jsonable(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for Wishart/GP models")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    output = run_search(cfg)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
