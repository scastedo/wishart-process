#!/usr/bin/env python3
"""Direct temporal-averaging Gaussian--Wishart analysis.

Raw calcium frames are averaged within trials before each duration-specific
Gaussian--Wishart fit.  The script is resumable and can run inference and
plotting separately::

    python temporal_averaging_wishart.py --stage all
    python temporal_averaging_wishart.py --stage fit
    python temporal_averaging_wishart.py --stage plot

The default analysis uses 1--10 frame windows centered near frame 60, 40
neurons per matched subsample, 10 subsamples per animal, and the per-animal
hyperparameters already stored in ``outputs/animal_hyperparam_search``.

Before inference, the script verifies that the complete raw-trial identities
are identical across all durations for each animal.  For decomposition,
conditions are averaged within mode and technical subsamples are then averaged
within animal.  NNLS and leave-one-mode-out diagnostics operate on each
animal's mean mode spectrum before FR/CTR means and SEMs are formed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "wishart-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls


FOOD_RESTRICTED_ANIMALS = (1, 2, 3, 6, 7, 8, 11, 12)
CONTROL_ANIMALS = (0, 4, 5, 9, 10, 13)
GROUP_ANIMALS = {
    "FR": FOOD_RESTRICTED_ANIMALS,
    "CTR": CONTROL_ANIMALS,
}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / (
    "wishart_temporal_avg_center60_equal40_sated"
)
DEFAULT_HYPERPARAM_DIR = PROJECT_ROOT / "outputs/animal_hyperparam_search"

FRAME_RATE_HZ = 20.0
RAW_TRACE_FRAMES = 120
TARGET_GAP_DEGREES = 30.0
N_SUBSAMPLE = 40
SHARED_BETA_GP = 0.03
FIT_SCHEMA_VERSION = 1
GAMMA = 1e-5
ADAM_STEP_SIZE = 0.001
NUM_PARTICLES = 1
SUBSAMPLE_SEED_BASE = 13
INFERENCE_SEED_BASE = 13
MODE_START = 1  # Exclude the largest covariance eigenmode from decomposition.
MODE_PLOT_DURATIONS = (1, 5, 10)
N_EIGENVALUE_BINS = 8

GROUP_COLORS = {
    "FR": "tab:red",
    "CTR": "tab:blue",
}

REQUIRED_FIT_KEYS = {
    "animal",
    "group",
    "dataset_state",
    "start",
    "stop",
    "repeat_idx",
    "n_subsample",
    "n_full_neurons",
    "neuron_indices",
    "n_total_trials",
    "n_complete_trials",
    "complete_trial_indices",
    "small_angle",
    "eigs_per_condition",
    "overlaps_small",
    "losses",
    "averaging_frames",
    "averaging_seconds",
    "window_center_frame",
    "iterations",
    "subsample_seed",
    "inference_seed",
    "shared_beta_gp",
    "fit_schema_version",
    "frame_rate_hz",
    "window_sample_midpoint_frame",
    "inference_config",
    "hyperparams",
    "hyperparams_source",
    "hyperparams_sha256",
}


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    hyperparam_dir: Path
    repeats: int
    max_frames: int
    center_frame: int
    iterations: int

    @property
    def animals(self) -> tuple[int, ...]:
        return FOOD_RESTRICTED_ANIMALS + CONTROL_ANIMALS

    @property
    def expected_fit_count(self) -> int:
        return self.max_frames * self.repeats * len(self.animals)


def expected_inference_config(config: RunConfig) -> dict[str, object]:
    """Return every setting that can change a duration-specific fit."""

    return {
        "fit_schema_version": FIT_SCHEMA_VERSION,
        "dataset_state": "sated",
        "target_gap_degrees": TARGET_GAP_DEGREES,
        "n_subsample": N_SUBSAMPLE,
        "shared_beta_gp": SHARED_BETA_GP,
        "gamma": GAMMA,
        "adam_step_size": ADAM_STEP_SIZE,
        "num_particles": NUM_PARTICLES,
        "iterations": config.iterations,
        "subsample_seed_base": SUBSAMPLE_SEED_BASE,
        "inference_seed_base": INFERENCE_SEED_BASE,
    }


@lru_cache(maxsize=None)
def expected_animal_hyperparams(
    source_path_string: str,
) -> tuple[dict[str, float | int], str]:
    """Load and fingerprint the effective parameters used by inference."""

    source_path = Path(source_path_string)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = payload.get("final_best_hyperparams")
    if not isinstance(selected, dict):
        raise ValueError(
            f"Missing final_best_hyperparams in {source_path}"
        )

    def required_float(key: str) -> float:
        value = selected.get(key)
        if value is None:
            raise ValueError(f"Missing {key} in {source_path}")
        return float(value)

    def float_or_default(key: str, default: float) -> float:
        value = selected.get(key)
        return float(default if value is None else value)

    if selected.get("p") is None:
        raise ValueError(f"Missing p in {source_path}")
    effective: dict[str, float | int] = {
        "lambda_gp_angle": required_float("lambda_gp_angle"),
        "gamma_gp_angle": float_or_default("gamma_gp_angle", GAMMA),
        "beta_gp_angle": SHARED_BETA_GP,
        "lambda_gp_sf": required_float("lambda_gp_sf"),
        "gamma_gp_sf": float_or_default("gamma_gp_sf", GAMMA),
        "beta_gp_sf": SHARED_BETA_GP,
        "lambda_wp_angle": required_float("lambda_wp_angle"),
        "gamma_wp_angle": float_or_default("gamma_wp_angle", GAMMA),
        "beta_wp_angle": float_or_default("beta_wp_angle", 1.0),
        "lambda_wp_sf": required_float("lambda_wp_sf"),
        "gamma_wp_sf": float_or_default("gamma_wp_sf", GAMMA),
        "beta_wp_sf": float_or_default("beta_wp_sf", 1.0),
        "p": int(selected["p"]),
    }
    canonical = json.dumps(
        effective,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    signature = hashlib.sha256(canonical).hexdigest()
    return effective, signature


def hyperparam_path(config: RunConfig, animal: int) -> Path:
    return config.hyperparam_dir / f"animal_{animal:02d}_hyperparams.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and plot duration-specific Gaussian--Wishart models after "
            "directly averaging raw calcium frames within each trial."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "fit", "plot"),
        default="all",
        help="Run inference, plotting, or both (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute valid existing fits during the fit stage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Fit and figure directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--hyperparam-dir",
        type=Path,
        default=DEFAULT_HYPERPARAM_DIR,
        help=(
            "Directory containing animal_XX_hyperparams.json files "
            f"(default: {DEFAULT_HYPERPARAM_DIR})."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Matched neuron subsamples per animal and duration (default: 10).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Fit durations from one through this many frames (default: 10).",
    )
    parser.add_argument(
        "--center-frame",
        type=int,
        default=60,
        help="Center frame used for all duration windows (default: 60).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=40_000,
        help="Variational iterations per fit (default: 40000).",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> RunConfig:
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.center_frame < 0:
        raise ValueError("--center-frame must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    config = RunConfig(
        output_dir=args.output_dir,
        hyperparam_dir=args.hyperparam_dir,
        repeats=args.repeats,
        max_frames=args.max_frames,
        center_frame=args.center_frame,
        iterations=args.iterations,
    )
    validate_windows(config)
    return config


def averaging_window(center_frame: int, n_frames: int) -> tuple[int, int]:
    """Return a nearly centered half-open frame interval of exact length."""

    start = int(center_frame) - int(n_frames) // 2
    stop = start + int(n_frames)
    return start, stop


def validate_windows(config: RunConfig) -> None:
    windows = []
    for n_frames in range(1, config.max_frames + 1):
        start, stop = averaging_window(config.center_frame, n_frames)
        if start < 0:
            raise ValueError(
                f"Duration {n_frames} produces a negative start frame: {start}"
            )
        if stop > RAW_TRACE_FRAMES:
            raise ValueError(
                f"Window {start}:{stop} exceeds the {RAW_TRACE_FRAMES}-frame "
                "raw calcium trace"
            )
        if stop - start != n_frames:
            raise AssertionError("Averaging-window length is incorrect")
        midpoint = 0.5 * (start + stop)
        if abs(midpoint - config.center_frame) > 0.5:
            raise AssertionError(
                f"Window {start}:{stop} is not centered near "
                f"frame {config.center_frame}"
            )
        windows.append((start, stop))

    if len(set(windows)) != config.max_frames:
        raise AssertionError("Duration windows are not unique")


def fit_filename(
    group: str,
    animal: int,
    repeat_idx: int,
    n_frames: int,
    start: int,
    stop: int,
) -> str:
    return (
        f"{group}_avg{n_frames:02d}_f{start:03d}_{stop:03d}_"
        f"a{animal:02d}_r{repeat_idx:02d}_overlaps_eigs.pkl"
    )


def fit_path(
    config: RunConfig,
    group: str,
    animal: int,
    repeat_idx: int,
    n_frames: int,
) -> Path:
    start, stop = averaging_window(config.center_frame, n_frames)
    return config.output_dir / fit_filename(
        group,
        animal,
        repeat_idx,
        n_frames,
        start,
        stop,
    )


def expected_fit_specs(config: RunConfig) -> Iterable[dict[str, object]]:
    for n_frames in range(1, config.max_frames + 1):
        start, stop = averaging_window(config.center_frame, n_frames)
        for group, animals in GROUP_ANIMALS.items():
            for animal in animals:
                for repeat_idx in range(config.repeats):
                    yield {
                        "group": group,
                        "animal": int(animal),
                        "repeat_idx": int(repeat_idx),
                        "n_frames": int(n_frames),
                        "start": int(start),
                        "stop": int(stop),
                        "path": fit_path(
                            config,
                            group,
                            animal,
                            repeat_idx,
                            n_frames,
                        ),
                    }


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    return result


def _gap_index(saved: dict) -> int:
    gaps = np.atleast_1d(np.asarray(saved["small_angle"], dtype=float))
    matches = np.flatnonzero(np.isclose(gaps, TARGET_GAP_DEGREES))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {TARGET_GAP_DEGREES:g}-degree entry, got {gaps}"
        )
    return int(matches[0])


def validate_saved_fit(
    path: Path,
    spec: dict[str, object],
    config: RunConfig,
    *,
    return_data: bool = False,
) -> tuple[bool, str, dict | None]:
    if not path.is_file():
        return False, "missing", None

    try:
        saved = load_pickle(path)
        missing = sorted(REQUIRED_FIT_KEYS.difference(saved))
        if missing:
            raise ValueError(f"missing keys: {missing}")

        scalar_expectations = {
            "group": str(spec["group"]),
            "animal": int(spec["animal"]),
            "repeat_idx": int(spec["repeat_idx"]),
            "start": int(spec["start"]),
            "stop": int(spec["stop"]),
            "averaging_frames": int(spec["n_frames"]),
            "n_subsample": N_SUBSAMPLE,
            "iterations": config.iterations,
        }
        for key, expected in scalar_expectations.items():
            actual = saved[key]
            if actual != expected:
                raise ValueError(f"{key}={actual!r}, expected {expected!r}")

        if str(saved["dataset_state"]).lower() != "sated":
            raise ValueError("dataset_state is not 'sated'")

        expected_seconds = int(spec["n_frames"]) / FRAME_RATE_HZ
        if not np.isclose(float(saved["averaging_seconds"]), expected_seconds):
            raise ValueError("averaging_seconds does not match duration")
        if not np.isclose(
            float(saved["window_center_frame"]),
            config.center_frame,
        ):
            raise ValueError("window_center_frame does not match configuration")
        expected_sample_midpoint = 0.5 * (
            int(spec["start"]) + int(spec["stop"]) - 1
        )
        if not np.isclose(
            float(saved["window_sample_midpoint_frame"]),
            expected_sample_midpoint,
        ):
            raise ValueError("window_sample_midpoint_frame is incorrect")
        if int(saved["fit_schema_version"]) != FIT_SCHEMA_VERSION:
            raise ValueError("fit_schema_version does not match this script")
        if not np.isclose(float(saved["frame_rate_hz"]), FRAME_RATE_HZ):
            raise ValueError("frame_rate_hz does not match this script")
        if saved["inference_config"] != expected_inference_config(config):
            raise ValueError("inference_config does not match this run")

        n_full_neurons = int(saved["n_full_neurons"])
        if n_full_neurons < N_SUBSAMPLE:
            raise ValueError("n_full_neurons is smaller than n_subsample")
        neuron_indices = np.asarray(saved["neuron_indices"], dtype=int)
        if neuron_indices.shape != (N_SUBSAMPLE,):
            raise ValueError(
                f"neuron_indices has shape {neuron_indices.shape}, "
                f"expected {(N_SUBSAMPLE,)}"
            )
        if len(np.unique(neuron_indices)) != N_SUBSAMPLE:
            raise ValueError("neuron_indices are not unique")
        if not np.all(np.diff(neuron_indices) > 0):
            raise ValueError("neuron_indices are not strictly sorted")
        if np.any(neuron_indices < 0) or np.any(
            neuron_indices >= n_full_neurons
        ):
            raise ValueError("neuron_indices are outside the available neurons")

        n_total_trials = int(saved["n_total_trials"])
        n_complete_trials = int(saved["n_complete_trials"])
        complete_trial_indices = np.asarray(
            saved["complete_trial_indices"],
            dtype=int,
        )
        if complete_trial_indices.ndim != 1:
            raise ValueError("complete_trial_indices must be one-dimensional")
        if n_complete_trials != len(complete_trial_indices):
            raise ValueError("n_complete_trials does not match stored indices")
        if n_complete_trials <= 0 or n_total_trials < n_complete_trials:
            raise ValueError("stored trial counts are invalid")
        if len(np.unique(complete_trial_indices)) != n_complete_trials:
            raise ValueError("complete_trial_indices are not unique")
        if (
            np.any(complete_trial_indices < 0)
            or np.any(complete_trial_indices >= n_total_trials)
            or not np.all(np.diff(complete_trial_indices) > 0)
        ):
            raise ValueError("complete_trial_indices are invalid or unsorted")

        expected_subsample_seed = SUBSAMPLE_SEED_BASE + 1000 * int(
            spec["animal"]
        ) + int(spec["repeat_idx"])
        expected_inference_seed = INFERENCE_SEED_BASE + 1000 * int(
            spec["animal"]
        ) + int(spec["repeat_idx"])
        if int(saved["subsample_seed"]) != expected_subsample_seed:
            raise ValueError("subsample_seed does not match the matched design")
        if int(saved["inference_seed"]) != expected_inference_seed:
            raise ValueError("inference_seed does not match the matched design")
        expected_neuron_indices = np.sort(
            np.random.default_rng(expected_subsample_seed).choice(
                n_full_neurons,
                N_SUBSAMPLE,
                replace=False,
            )
        )
        if not np.array_equal(neuron_indices, expected_neuron_indices):
            raise ValueError("neuron_indices do not match their stored seed")
        if not np.isclose(float(saved["shared_beta_gp"]), SHARED_BETA_GP):
            raise ValueError("shared_beta_gp does not match the June 3 setting")
        hyperparams = saved["hyperparams"]
        if not isinstance(hyperparams, dict):
            raise ValueError("hyperparams is not a dictionary")
        expected_hyperparams, expected_hyperparams_sha256 = (
            expected_animal_hyperparams(
                str(hyperparam_path(config, int(spec["animal"])).resolve())
            )
        )
        for key, expected_value in expected_hyperparams.items():
            actual_value = hyperparams.get(key)
            if actual_value is None or not np.isclose(
                float(actual_value),
                float(expected_value),
            ):
                raise ValueError(f"effective hyperparameter {key} has changed")
        if saved["hyperparams_sha256"] != expected_hyperparams_sha256:
            raise ValueError("hyperparameter fingerprint has changed")
        expected_hyperparam_name = (
            f"animal_{int(spec['animal']):02d}_hyperparams.json"
        )
        if Path(saved["hyperparams_source"]).name != expected_hyperparam_name:
            raise ValueError("hyperparams_source does not match animal")

        losses = np.asarray(saved["losses"], dtype=float)
        if losses.shape != (config.iterations,):
            raise ValueError(
                f"losses has shape {losses.shape}, "
                f"expected {(config.iterations,)}"
            )
        if not np.isfinite(losses).all():
            raise ValueError("losses contain non-finite values")

        gap_idx = _gap_index(saved)
        eigenvalues = np.asarray(saved["eigs_per_condition"], dtype=float)
        overlaps = np.asarray(saved["overlaps_small"], dtype=float)
        if eigenvalues.shape != (12, 5, N_SUBSAMPLE, 1):
            raise ValueError(
                f"eigs_per_condition has invalid shape {eigenvalues.shape}"
            )
        if overlaps.shape != (1, 12, 5, N_SUBSAMPLE, 1):
            raise ValueError(
                f"overlaps_small has invalid shape {overlaps.shape}"
            )
        if overlaps.shape[0] <= gap_idx:
            raise ValueError("overlaps_small does not contain target gap")
        if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
            raise ValueError("eigs_per_condition must be finite and positive")
        if not np.isfinite(overlaps).all() or np.any(overlaps < 0):
            raise ValueError("overlaps_small must be finite and non-negative")

    except Exception as exc:  # A corrupt file must be recomputed, not trusted.
        return False, str(exc), None

    return True, "valid", saved if return_data else None


def atomic_pickle_dump(
    value: object,
    path: Path,
    spec: dict[str, object],
    config: RunConfig,
) -> None:
    """Validate a temporary pickle before atomically replacing its target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        valid, reason, _ = validate_saved_fit(temporary, spec, config)
        if not valid:
            raise ValueError(
                f"Refusing to replace {path.name} with an invalid fit: {reason}"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def configure_inference(config: RunConfig):
    import z_castedo_snr_script_dynamic_equal_neurons as inference_runner

    inference_runner.DATASET_STATE = "sated"
    inference_runner.small_angle = [TARGET_GAP_DEGREES]
    inference_runner.N_SUBSAMPLE = N_SUBSAMPLE
    inference_runner.REPEATS = config.repeats
    inference_runner.HYPERPARAM_DIR = config.hyperparam_dir
    inference_runner.HYPERPARAM_OVERRIDES = {}
    inference_runner.SHARED_BETA_GP = SHARED_BETA_GP
    inference_runner.ITERATIONS = config.iterations
    inference_runner.NUM_PART = NUM_PARTICLES
    inference_runner.GAMMA = GAMMA
    inference_runner.ADAM_OPT = ADAM_STEP_SIZE
    inference_runner.SUBSAMPLE_SEED = SUBSAMPLE_SEED_BASE
    inference_runner.INFERENCE_SEED = INFERENCE_SEED_BASE
    inference_runner.configure_base_inputs()
    return inference_runner


def compute_matched_trial_metadata(
    inference_runner,
    config: RunConfig,
) -> dict[int, dict[str, object]]:
    """Confirm every duration uses the same complete raw trials per animal."""

    deconv, angle, sf = inference_runner.load_state_inputs()
    metadata: dict[int, dict[str, object]] = {}
    for animal in config.animals:
        animal_data = np.asarray(
            inference_runner.base.resort_preprocessing(
                deconv,
                angle,
                sf,
                animal,
            )
        )
        if animal_data.ndim != 5:
            raise ValueError(
                f"Animal {animal}: expected five raw-data axes, got "
                f"{animal_data.shape}"
            )
        if animal_data.shape[-1] < RAW_TRACE_FRAMES:
            raise ValueError(
                f"Animal {animal}: raw trace has only "
                f"{animal_data.shape[-1]} frames"
            )

        duration_indices = {}
        for n_frames in range(1, config.max_frames + 1):
            start, stop = averaging_window(config.center_frame, n_frames)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                averaged = np.nanmean(
                    animal_data[..., start:stop],
                    axis=-1,
                )
            # Raw axes are neuron, angle, spatial frequency, trial, frame.
            complete = ~np.isnan(averaged).any(axis=(0, 1, 2))
            duration_indices[n_frames] = np.flatnonzero(complete).astype(int)

        reference = duration_indices[1]
        mismatched = [
            n_frames
            for n_frames, indices in duration_indices.items()
            if not np.array_equal(indices, reference)
        ]
        if mismatched:
            raise ValueError(
                f"Animal {animal}: complete-trial identities differ from the "
                f"one-frame window at durations {mismatched}. The reused "
                "inference function cannot guarantee a matched-duration "
                "comparison for these data."
            )
        if len(reference) == 0:
            raise ValueError(f"Animal {animal}: no common complete trials")
        metadata[int(animal)] = {
            "n_full_neurons": int(animal_data.shape[0]),
            "n_total_trials": int(animal_data.shape[3]),
            "n_complete_trials": int(len(reference)),
            "complete_trial_indices": reference,
        }
    return metadata


def audit_matched_saved_metadata(config: RunConfig) -> list[tuple[Path, str]]:
    """Validate the cached set and its matched duration design."""

    invalid: list[tuple[Path, str]] = []
    neuron_lookup: dict[
        tuple[str, int, int],
        tuple[int, tuple[int, ...]],
    ] = {}
    trial_lookup: dict[tuple[str, int], tuple[int, tuple[int, ...]]] = {}
    for spec in expected_fit_specs(config):
        path = Path(spec["path"])
        valid, reason, saved = validate_saved_fit(
            path,
            spec,
            config,
            return_data=True,
        )
        if not valid or saved is None:
            invalid.append((path, reason))
            continue

        group = str(spec["group"])
        animal = int(spec["animal"])
        repeat_idx = int(spec["repeat_idx"])
        match_key = (group, animal, repeat_idx)
        neuron_design = (
            int(saved["n_full_neurons"]),
            tuple(np.asarray(saved["neuron_indices"], dtype=int)),
        )
        previous_neurons = neuron_lookup.setdefault(match_key, neuron_design)
        if previous_neurons != neuron_design:
            invalid.append((path, "neuron subset differs across durations"))

        trial_key = (group, animal)
        trials = (
            int(saved["n_total_trials"]),
            tuple(np.asarray(saved["complete_trial_indices"], dtype=int)),
        )
        previous_trials = trial_lookup.setdefault(trial_key, trials)
        if previous_trials != trials:
            invalid.append((path, "complete-trial set differs across durations"))
    return invalid


def run_fit_stage(config: RunConfig, force: bool) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    specs = list(expected_fit_specs(config))
    if len(specs) != config.expected_fit_count:
        raise AssertionError("Unexpected number of fit specifications")

    if not force:
        invalid_cached = audit_matched_saved_metadata(config)
        if not invalid_cached:
            print(
                f"Fit stage complete: all {len(specs)} files are already valid.",
                flush=True,
            )
            return
        print(
            f"Fit stage found {len(invalid_cached)} missing, invalid, or "
            "unmatched cached files.",
            flush=True,
        )

    inference_runner = configure_inference(config)

    inference_runner.validate_neuron_counts(config.animals, N_SUBSAMPLE)
    matched_trial_metadata = compute_matched_trial_metadata(
        inference_runner,
        config,
    )
    animal_hyperparams = inference_runner.base.preload_search_hyperparams(
        config.animals,
        config.hyperparam_dir,
    )

    completed = 0
    recomputed_invalid = 0
    for index, spec in enumerate(specs, start=1):
        path = Path(spec["path"])
        valid, reason, existing = validate_saved_fit(
            path,
            spec,
            config,
            return_data=True,
        )
        expected_trials = matched_trial_metadata[int(spec["animal"])]
        if valid and existing is not None:
            stored_trial_design = (
                int(existing["n_full_neurons"]),
                int(existing["n_total_trials"]),
                tuple(
                    np.asarray(
                        existing["complete_trial_indices"],
                        dtype=int,
                    )
                ),
            )
            current_trial_design = (
                int(expected_trials["n_full_neurons"]),
                int(expected_trials["n_total_trials"]),
                tuple(
                    np.asarray(
                        expected_trials["complete_trial_indices"],
                        dtype=int,
                    )
                ),
            )
            if stored_trial_design != current_trial_design:
                valid = False
                reason = "neuron/trial design differs from current raw data"
        if valid and not force:
            completed += 1
            print(
                f"[{index}/{len(specs)}] Skip valid fit: {path.name}",
                flush=True,
            )
            continue
        if path.exists() and not valid:
            recomputed_invalid += 1
            print(
                f"[{index}/{len(specs)}] Recompute invalid fit "
                f"({reason}): {path.name}",
                flush=True,
            )
        else:
            action = "Force refit" if force and path.exists() else "Fit"
            print(
                f"[{index}/{len(specs)}] {action}: {path.name}",
                flush=True,
            )

        animal = int(spec["animal"])
        repeat_idx = int(spec["repeat_idx"])
        hyperparams, hyperparams_source = animal_hyperparams[animal]
        _, hyperparams_sha256 = expected_animal_hyperparams(
            str(hyperparam_path(config, animal).resolve())
        )
        saved = inference_runner.analysis(
            animal=animal,
            start=int(spec["start"]),
            stop=int(spec["stop"]),
            repeat_idx=repeat_idx,
            n_subsample=N_SUBSAMPLE,
            save_dir=None,
            fname_prefix=None,
            animal_hyperparams=hyperparams,
            hyperparams_source=hyperparams_source,
        )
        saved.update(
            {
                "group": str(spec["group"]),
                "averaging_frames": int(spec["n_frames"]),
                "averaging_seconds": int(spec["n_frames"]) / FRAME_RATE_HZ,
                "window_center_frame": int(config.center_frame),
                "window_sample_midpoint_frame": 0.5
                * (int(spec["start"]) + int(spec["stop"]) - 1),
                "window_design": "fixed_center",
                "iterations": int(config.iterations),
                "shared_beta_gp": float(SHARED_BETA_GP),
                "fit_schema_version": FIT_SCHEMA_VERSION,
                "frame_rate_hz": FRAME_RATE_HZ,
                "inference_config": expected_inference_config(config),
                "hyperparams_sha256": hyperparams_sha256,
                **matched_trial_metadata[animal],
                "trial_filtering": (
                    "nanmean_within_window_then_drop_trials_with_remaining_nan"
                ),
            }
        )
        atomic_pickle_dump(saved, path, spec, config)

        valid, reason, _ = validate_saved_fit(path, spec, config)
        if not valid:
            raise RuntimeError(
                f"Newly written fit failed validation: {path}: {reason}"
            )
        completed += 1

    invalid_final = audit_matched_saved_metadata(config)
    if invalid_final:
        examples = "; ".join(
            f"{path.name}: {reason}" for path, reason in invalid_final[:5]
        )
        raise RuntimeError(
            f"Fit-stage design audit failed for {len(invalid_final)} files: "
            f"{examples}"
        )

    print(
        f"Fit stage complete: {completed}/{len(specs)} valid files; "
        f"recomputed invalid files={recomputed_invalid}.",
        flush=True,
    )


def _scaled_combined_fit(
    overlap: np.ndarray,
    mean_abs_projection: np.ndarray,
    eigenvalue: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit alpha*E[z^2] + beta*E[|z|] with stable column scaling."""

    scale = float(
        np.median(mean_abs_projection[mean_abs_projection > 0])
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot scale a non-positive signal projection")
    scaled_quadratic_predictor = overlap / scale**2
    scaled_linear_predictor = mean_abs_projection / scale
    scaled_alpha, scaled_beta = nnls(
        np.column_stack(
            [scaled_quadratic_predictor, scaled_linear_predictor]
        ),
        eigenvalue,
    )[0]
    quadratic = scaled_alpha * scaled_quadratic_predictor
    linear = scaled_beta * scaled_linear_predictor
    prediction = quadratic + linear
    alpha = float(scaled_alpha / scale**2)
    beta = float(scaled_beta / scale)
    return alpha, beta, linear, quadratic, prediction


def analyze_saved_fit(
    saved: dict,
    spec: dict[str, object],
) -> tuple[dict, list[dict]]:
    """Extract trace and condition-averaged values for each covariance mode."""

    group = str(spec["group"])
    animal = int(spec["animal"])
    repeat_idx = int(spec["repeat_idx"])
    n_frames = int(spec["n_frames"])
    seconds = n_frames / FRAME_RATE_HZ
    identity = {
        "group": group,
        "animal": animal,
        "repeat": repeat_idx,
        "n_frames": n_frames,
        "seconds": seconds,
    }

    gap_idx = _gap_index(saved)
    eigenvalues = np.asarray(saved["eigs_per_condition"], dtype=float)[..., 0]
    overlaps = np.asarray(saved["overlaps_small"], dtype=float)[gap_idx, ..., 0]
    trace_row = {
        **identity,
        "trace": float(np.mean(np.sum(eigenvalues, axis=2))),
    }

    mode_value_rows = []
    for mode in range(N_SUBSAMPLE):
        mode_eigenvalues = eigenvalues[..., mode]
        mode_overlaps = overlaps[..., mode]
        mode_abs_projections = np.sqrt(mode_overlaps)
        mode_value_rows.append(
            {
                **identity,
                "mode": int(mode),
                "eigenvalue": float(np.mean(mode_eigenvalues)),
                "overlap": float(np.mean(mode_overlaps)),
                "mean_abs_projection": float(
                    np.mean(mode_abs_projections)
                ),
                "n_conditions": int(mode_eigenvalues.size),
            }
        )

    mode_frame = pd.DataFrame(mode_value_rows)
    numeric = mode_frame[
        ["eigenvalue", "overlap", "mean_abs_projection"]
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(numeric).all()
        or np.any(mode_frame["eigenvalue"] <= 0)
        or np.any(mode_frame["overlap"] < 0)
        or np.any(mode_frame["mean_abs_projection"] < 0)
    ):
        raise ValueError(
            f"Invalid condition-averaged mode values for {group} animal "
            f"{animal}, repeat {repeat_idx}, duration {n_frames}"
        )
    if not np.all(mode_frame["n_conditions"] == 60):
        raise AssertionError("Every mode must contain all 60 conditions")

    return trace_row, mode_value_rows


def load_fit_records(config: RunConfig) -> dict[str, pd.DataFrame]:
    trace_rows = []
    mode_value_rows = []
    manifest_rows = []
    neuron_lookup: dict[
        tuple[str, int, int],
        tuple[int, tuple[int, ...]],
    ] = {}
    seed_lookup: dict[tuple[str, int, int], tuple[int, int]] = {}
    trial_lookup: dict[tuple[str, int], tuple[int, tuple[int, ...]]] = {}
    invalid = []

    specs = list(expected_fit_specs(config))
    for spec in specs:
        path = Path(spec["path"])
        valid, reason, saved = validate_saved_fit(
            path,
            spec,
            config,
            return_data=True,
        )
        if not valid or saved is None:
            invalid.append((path, reason))
            continue

        group = str(spec["group"])
        animal = int(spec["animal"])
        repeat_idx = int(spec["repeat_idx"])
        match_key = (group, animal, repeat_idx)
        neuron_design = (
            int(saved["n_full_neurons"]),
            tuple(np.asarray(saved["neuron_indices"], dtype=int)),
        )
        previous_neurons = neuron_lookup.setdefault(match_key, neuron_design)
        if previous_neurons != neuron_design:
            raise ValueError(
                "Neuron subsets do not match across durations for "
                f"{group} animal {animal}, repeat {repeat_idx}"
            )
        seeds = (int(saved["subsample_seed"]), int(saved["inference_seed"]))
        previous_seeds = seed_lookup.setdefault(match_key, seeds)
        if previous_seeds != seeds:
            raise ValueError(
                "Seeds do not match across durations for "
                f"{group} animal {animal}, repeat {repeat_idx}"
            )
        trial_key = (group, animal)
        trial_design = (
            int(saved["n_total_trials"]),
            tuple(np.asarray(saved["complete_trial_indices"], dtype=int)),
        )
        previous_trials = trial_lookup.setdefault(trial_key, trial_design)
        if previous_trials != trial_design:
            raise ValueError(
                "Complete-trial identities do not match across durations for "
                f"{group} animal {animal}"
            )

        trace_row, fit_mode_rows = analyze_saved_fit(saved, spec)
        trace_rows.append(trace_row)
        mode_value_rows.extend(fit_mode_rows)
        manifest_rows.append(
            {
                "path": str(path),
                "group": group,
                "animal": animal,
                "repeat": repeat_idx,
                "n_frames": int(spec["n_frames"]),
                "start": int(spec["start"]),
                "stop": int(spec["stop"]),
                "seconds": int(spec["n_frames"]) / FRAME_RATE_HZ,
                "subsample_seed": seeds[0],
                "inference_seed": seeds[1],
                "n_full_neurons": int(saved["n_full_neurons"]),
                "n_total_trials": int(saved["n_total_trials"]),
                "n_complete_trials": int(saved["n_complete_trials"]),
                "final_loss": float(np.asarray(saved["losses"])[-1]),
                "n_modes": N_SUBSAMPLE,
            }
        )

    if invalid:
        examples = "\n".join(
            f"  - {path}: {reason}" for path, reason in invalid[:10]
        )
        raise RuntimeError(
            f"Plot stage requires all {config.expected_fit_count} valid fits; "
            f"found {len(invalid)} missing or invalid files.\n{examples}\n"
            "Run with --stage fit to create or repair them."
        )
    if len(manifest_rows) != config.expected_fit_count:
        raise AssertionError("Loaded fit count does not match expectation")

    return {
        "repeat_traces": pd.DataFrame(trace_rows),
        "repeat_mode_values": pd.DataFrame(mode_value_rows),
        "manifest": pd.DataFrame(manifest_rows),
    }


def calculate_trace_scaling(
    repeat_traces: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = (
        repeat_traces[repeat_traces["n_frames"] == 1][
            ["group", "animal", "repeat", "trace"]
        ]
        .rename(columns={"trace": "one_frame_trace"})
    )
    scaled_repeats = repeat_traces.merge(
        baseline,
        on=["group", "animal", "repeat"],
        how="left",
        validate="many_to_one",
    )
    scaled_repeats["normalized_trace"] = (
        scaled_repeats["trace"] / scaled_repeats["one_frame_trace"]
    )
    if (
        not np.isfinite(scaled_repeats["normalized_trace"]).all()
        or np.any(scaled_repeats["one_frame_trace"] <= 0)
    ):
        raise ValueError("Covariance trace baseline is missing or non-positive")

    animal_traces = (
        scaled_repeats.groupby(
            ["group", "animal", "n_frames", "seconds"],
            as_index=False,
        )
        .agg(
            trace=("trace", "mean"),
            normalized_trace=("normalized_trace", "mean"),
            n_repeats=("repeat", "nunique"),
        )
    )
    summary = (
        animal_traces.groupby(
            ["group", "n_frames", "seconds"],
            as_index=False,
        )
        .agg(
            trace_mean=("normalized_trace", "mean"),
            trace_sem=("normalized_trace", "sem"),
            n_animals=("animal", "nunique"),
        )
    )
    return scaled_repeats, animal_traces, summary


def aggregate_animal_mode_values(
    repeat_modes: pd.DataFrame,
    config: RunConfig,
) -> pd.DataFrame:
    """Average conditions first and technical repeats second, within animal."""

    key_columns = [
        "group",
        "animal",
        "repeat",
        "n_frames",
        "seconds",
        "mode",
    ]
    if repeat_modes.duplicated(key_columns).any():
        raise ValueError("Repeat-mode table contains duplicate design cells")
    expected_rows = config.expected_fit_count * N_SUBSAMPLE
    if len(repeat_modes) != expected_rows:
        raise ValueError(
            f"Repeat-mode table has {len(repeat_modes)} rows, "
            f"expected {expected_rows}"
        )
    if not np.all(repeat_modes["n_conditions"] == 60):
        raise ValueError("Every repeat-mode row must average 60 conditions")

    spectrum_sizes = repeat_modes.groupby(
        ["group", "animal", "repeat", "n_frames"]
    )["mode"].nunique()
    if not np.all(spectrum_sizes.to_numpy() == N_SUBSAMPLE):
        raise ValueError("A repeat spectrum is missing covariance modes")
    repeat_counts = repeat_modes.groupby(
        ["group", "animal", "n_frames", "mode"]
    )["repeat"].nunique()
    if not np.all(repeat_counts.to_numpy() == config.repeats):
        raise ValueError("Animal spectra do not contain every technical repeat")

    animal_modes = (
        repeat_modes.groupby(
            ["group", "animal", "n_frames", "seconds", "mode"],
            as_index=False,
        )
        .agg(
            eigenvalue=("eigenvalue", "mean"),
            overlap=("overlap", "mean"),
            mean_abs_projection=("mean_abs_projection", "mean"),
            n_repeats=("repeat", "nunique"),
            n_condition_values=("n_conditions", "sum"),
        )
    )
    expected_animal_rows = (
        len(config.animals) * config.max_frames * N_SUBSAMPLE
    )
    if len(animal_modes) != expected_animal_rows:
        raise ValueError(
            f"Animal-mode table has {len(animal_modes)} rows, "
            f"expected {expected_animal_rows}"
        )
    if not np.all(animal_modes["n_repeats"] == config.repeats):
        raise ValueError("Animal-mode repeat count does not match configuration")
    if not np.all(
        animal_modes["n_condition_values"] == 60 * config.repeats
    ):
        raise ValueError("Animal-mode condition count is incomplete")
    return animal_modes


def analyze_mode_spectra(
    mode_values: pd.DataFrame,
    *,
    include_repeat: bool,
    analysis_level: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit and cross-validate linear--quadratic models across mode ranks."""

    group_columns = ["group", "animal"]
    if include_repeat:
        group_columns.append("repeat")
    group_columns.extend(["n_frames", "seconds"])

    fit_rows = []
    cv_rows = []
    lomo_rows = []
    expected_modes = np.arange(MODE_START, N_SUBSAMPLE)

    for keys, spectrum in mode_values.groupby(group_columns, sort=False):
        identity = dict(zip(group_columns, keys))
        spectrum = spectrum.sort_values("mode")
        retained = spectrum[spectrum["mode"] >= MODE_START].copy()
        modes = retained["mode"].to_numpy(dtype=int)
        if not np.array_equal(modes, expected_modes):
            raise ValueError(
                f"{analysis_level} spectrum has modes {modes.tolist()}, "
                f"expected {expected_modes.tolist()}"
            )

        eigenvalue = retained["eigenvalue"].to_numpy(dtype=float)
        overlap = retained["overlap"].to_numpy(dtype=float)
        mean_abs_projection = retained[
            "mean_abs_projection"
        ].to_numpy(dtype=float)
        numeric = np.column_stack(
            [eigenvalue, overlap, mean_abs_projection]
        )
        if not np.isfinite(numeric).all() or np.any(numeric <= 0):
            raise ValueError(
                f"{analysis_level} spectrum contains invalid fit values"
            )

        alpha, beta, linear, quadratic, prediction = _scaled_combined_fit(
            overlap,
            mean_abs_projection,
            eigenvalue,
        )
        fitted_total = float(np.sum(prediction))
        if not np.isfinite(fitted_total) or fitted_total <= 0:
            raise ValueError("Combined model has a non-positive fitted total")
        linear_fraction = float(np.sum(linear) / fitted_total)
        quadratic_fraction = float(np.sum(quadratic) / fitted_total)
        if not (
            0 <= linear_fraction <= 1
            and 0 <= quadratic_fraction <= 1
            and np.isclose(linear_fraction + quadratic_fraction, 1.0)
        ):
            raise AssertionError("Invalid linear--quadratic fractions")

        residual_ss = float(np.sum((eigenvalue - prediction) ** 2))
        total_ss = float(
            np.sum((eigenvalue - np.mean(eigenvalue)) ** 2)
        )
        fit_rows.append(
            {
                **identity,
                "analysis_level": analysis_level,
                "alpha": alpha,
                "beta": beta,
                "linear_fraction": linear_fraction,
                "quadratic_fraction": quadratic_fraction,
                "r_squared": (
                    1.0 - residual_ss / total_ss
                    if total_ss > 0
                    else np.nan
                ),
                "n_valid_modes": int(len(modes)),
            }
        )

        squared_errors = {
            "linear_only": [],
            "quadratic_only": [],
            "combined": [],
        }
        actual_values = []
        held_modes = []
        for held_index, held_mode in enumerate(modes):
            train_mask = modes != held_mode
            test_mask = ~train_mask
            if train_mask[held_index] or not test_mask[held_index]:
                raise AssertionError("Held-out mode leaked into training data")
            training_modes = modes[train_mask]
            if (
                held_mode in training_modes
                or len(training_modes) != len(modes) - 1
            ):
                raise AssertionError("LOMO training set has incorrect modes")

            train_projection = mean_abs_projection[train_mask]
            scale = float(np.median(train_projection[train_projection > 0]))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("LOMO training scale is non-positive")
            train_quadratic = overlap[train_mask] / scale**2
            train_linear = train_projection / scale
            test_quadratic = overlap[test_mask] / scale**2
            test_linear = mean_abs_projection[test_mask] / scale
            train_target = eigenvalue[train_mask]
            test_target = eigenvalue[test_mask]

            scaled_alpha, scaled_beta = nnls(
                np.column_stack([train_quadratic, train_linear]),
                train_target,
            )[0]
            linear_coefficient = nnls(
                train_linear[:, None],
                train_target,
            )[0][0]
            quadratic_coefficient = nnls(
                train_quadratic[:, None],
                train_target,
            )[0][0]

            linear_part = scaled_beta * test_linear
            quadratic_part = scaled_alpha * test_quadratic
            predictions = {
                "linear_only": linear_coefficient * test_linear,
                "quadratic_only": quadratic_coefficient * test_quadratic,
                "combined": linear_part + quadratic_part,
            }
            for model, model_prediction in predictions.items():
                squared_errors[model].append(
                    float(np.square(test_target - model_prediction)[0])
                )
            actual_values.append(float(test_target[0]))
            held_modes.append(int(held_mode))

            combined_prediction = float(predictions["combined"][0])
            if not np.isfinite(combined_prediction) or combined_prediction <= 0:
                raise ValueError("LOMO combined prediction is non-positive")
            lomo_linear_fraction = float(linear_part[0] / combined_prediction)
            lomo_quadratic_fraction = float(
                quadratic_part[0] / combined_prediction
            )
            if not (
                0 <= lomo_linear_fraction <= 1
                and 0 <= lomo_quadratic_fraction <= 1
                and np.isclose(
                    lomo_linear_fraction + lomo_quadratic_fraction,
                    1.0,
                )
            ):
                raise AssertionError("Invalid LOMO component fractions")

            lomo_rows.append(
                {
                    **identity,
                    "analysis_level": analysis_level,
                    "mode": int(held_mode),
                    "eigenvalue": float(test_target[0]),
                    "overlap": float(overlap[test_mask][0]),
                    "mean_abs_projection": float(
                        mean_abs_projection[test_mask][0]
                    ),
                    "alpha_training": float(scaled_alpha / scale**2),
                    "beta_training": float(scaled_beta / scale),
                    "linear_fraction": lomo_linear_fraction,
                    "quadratic_fraction": lomo_quadratic_fraction,
                    "combined_prediction": combined_prediction,
                    "linear_only_prediction": float(
                        predictions["linear_only"][0]
                    ),
                    "quadratic_only_prediction": float(
                        predictions["quadratic_only"][0]
                    ),
                    "n_training_modes": int(train_mask.sum()),
                }
            )

        if not np.array_equal(np.sort(held_modes), expected_modes):
            raise AssertionError("LOMO did not hold every retained mode once")
        actual_rms = float(np.sqrt(np.mean(np.square(actual_values))))
        if not np.isfinite(actual_rms) or actual_rms <= 0:
            raise ValueError("LOMO target RMS is non-positive")
        for model, errors in squared_errors.items():
            if len(errors) != len(expected_modes):
                raise AssertionError("LOMO prediction count is incomplete")
            rmse = float(np.sqrt(np.mean(errors)))
            cv_rows.append(
                {
                    **identity,
                    "analysis_level": analysis_level,
                    "model": model,
                    "rmse": rmse,
                    "normalized_rmse": rmse / actual_rms,
                    "n_test_points": int(len(actual_values)),
                    "n_held_modes": int(len(held_modes)),
                }
            )

    return (
        pd.DataFrame(fit_rows),
        pd.DataFrame(cv_rows),
        pd.DataFrame(lomo_rows),
    )


def summarize_duration_fractions(
    animal_fits: pd.DataFrame,
) -> pd.DataFrame:
    return (
        animal_fits.groupby(
            ["group", "n_frames", "seconds"],
            as_index=False,
        )
        .agg(
            linear_mean=("linear_fraction", "mean"),
            linear_sem=("linear_fraction", "sem"),
            quadratic_mean=("quadratic_fraction", "mean"),
            quadratic_sem=("quadratic_fraction", "sem"),
            r_squared_mean=("r_squared", "mean"),
            n_animals=("animal", "nunique"),
        )
    )


def summarize_lomo_cv(
    animal_errors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    error_wide = animal_errors.pivot(
        index=["group", "animal", "n_frames", "seconds"],
        columns="model",
        values="normalized_rmse",
    ).reset_index()
    error_wide["combined_vs_linear_improvement"] = (
        error_wide["linear_only"] - error_wide["combined"]
    )
    error_wide["combined_vs_quadratic_improvement"] = (
        error_wide["quadratic_only"] - error_wide["combined"]
    )
    error_summary = (
        animal_errors.groupby(
            ["group", "n_frames", "seconds", "model"],
            as_index=False,
        )
        .agg(
            nrmse_mean=("normalized_rmse", "mean"),
            nrmse_sem=("normalized_rmse", "sem"),
            n_animals=("animal", "nunique"),
        )
    )
    improvement_summary = (
        error_wide.groupby(
            ["group", "n_frames", "seconds"],
            as_index=False,
        )
        .agg(
            combined_vs_linear_mean=(
                "combined_vs_linear_improvement",
                "mean",
            ),
            combined_vs_linear_sem=(
                "combined_vs_linear_improvement",
                "sem",
            ),
            combined_vs_quadratic_mean=(
                "combined_vs_quadratic_improvement",
                "mean",
            ),
            combined_vs_quadratic_sem=(
                "combined_vs_quadratic_improvement",
                "sem",
            ),
            n_animals=("animal", "nunique"),
        )
    )
    return error_wide, error_summary, improvement_summary


def calculate_mode_summary(
    animal_lomo_modes: pd.DataFrame,
    config: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    durations = [
        duration
        for duration in MODE_PLOT_DURATIONS
        if duration <= config.max_frames
    ]
    duration_rows = animal_lomo_modes[
        animal_lomo_modes["n_frames"].isin(durations)
    ].copy()
    observed_durations = set(duration_rows["n_frames"].unique())
    if observed_durations != set(durations):
        raise ValueError(
            "Auxiliary plot is missing requested durations: "
            f"expected {durations}, found {sorted(observed_durations)}"
        )
    fraction_values = duration_rows[
        ["linear_fraction", "quadratic_fraction"]
    ].to_numpy(dtype=float)
    if not np.isfinite(fraction_values).all():
        raise ValueError("Auxiliary LOMO fractions contain non-finite values")
    mode_counts = duration_rows.groupby(
        ["group", "animal", "n_frames"]
    )["mode"].nunique()
    if not np.all(mode_counts.to_numpy() == N_SUBSAMPLE - MODE_START):
        raise ValueError("Auxiliary animal spectra are missing held-out modes")
    animal_counts = (
        duration_rows.groupby(["group", "n_frames"])["animal"]
        .nunique()
        .to_dict()
    )
    expected_counts = {
        (group, duration): len(animals)
        for group, animals in GROUP_ANIMALS.items()
        for duration in durations
    }
    if animal_counts != expected_counts:
        raise ValueError(
            f"Auxiliary group-by-duration counts are {animal_counts}; "
            f"expected {expected_counts}"
        )

    selected = duration_rows[
        np.isfinite(duration_rows["linear_fraction"])
        & (duration_rows["eigenvalue"] > 0)
    ].copy()
    if selected.empty:
        raise ValueError("No valid points for eigenvalue-dependence figure")

    edges = np.unique(
        np.quantile(
            selected["eigenvalue"],
            np.linspace(0, 1, N_EIGENVALUE_BINS + 1),
        )
    )
    if len(edges) < 3:
        raise ValueError("Too few unique eigenvalue-bin edges")
    selected["eigen_bin"] = pd.cut(
        selected["eigenvalue"],
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    bin_positions = (
        selected.groupby("eigen_bin", observed=True, as_index=False)
        .agg(eigenvalue_mid=("eigenvalue", "median"))
    )
    animal_binned = (
        selected.groupby(
            ["group", "animal", "n_frames", "seconds", "eigen_bin"],
            observed=True,
            as_index=False,
        )
        .agg(
            linear_fraction=("linear_fraction", "mean"),
            quadratic_fraction=("quadratic_fraction", "mean"),
            n_modes=("mode", "nunique"),
        )
        .merge(bin_positions, on="eigen_bin", how="left", validate="many_to_one")
    )
    summary = (
        animal_binned.groupby(
            [
                "group",
                "n_frames",
                "seconds",
                "eigen_bin",
                "eigenvalue_mid",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            linear_mean=("linear_fraction", "mean"),
            linear_sem=("linear_fraction", "sem"),
            quadratic_mean=("quadratic_fraction", "mean"),
            quadratic_sem=("quadratic_fraction", "sem"),
            n_animals=("animal", "nunique"),
        )
    )
    summary = summary[summary["n_animals"] >= 3].reset_index(drop=True)
    return animal_binned, summary


def validate_group_sizes(frame: pd.DataFrame, label: str) -> None:
    counts = frame.groupby("group")["animal"].nunique().to_dict()
    expected = {"CTR": 6, "FR": 8}
    if counts != expected:
        raise ValueError(f"{label} has group sizes {counts}, expected {expected}")


def validate_group_duration_counts(
    frame: pd.DataFrame,
    label: str,
    config: RunConfig,
) -> None:
    counts = (
        frame.groupby(["group", "n_frames"])["animal"]
        .nunique()
        .to_dict()
    )
    expected = {
        (group, n_frames): len(animals)
        for group, animals in GROUP_ANIMALS.items()
        for n_frames in range(1, config.max_frames + 1)
    }
    if counts != expected:
        raise ValueError(f"{label} has counts {counts}, expected {expected}")


def validate_analysis_row_count(
    frame: pd.DataFrame,
    expected: int,
    label: str,
) -> None:
    if len(frame) != expected:
        raise ValueError(f"{label} has {len(frame)} rows, expected {expected}")


def plot_main_figure(
    trace_summary: pd.DataFrame,
    fraction_summary: pd.DataFrame,
    improvement_summary: pd.DataFrame,
    config: RunConfig,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8), dpi=200)
    trace_ax, fraction_ax, cv_ax = axes

    for group, color in GROUP_COLORS.items():
        trace_group = trace_summary[
            trace_summary["group"] == group
        ].sort_values("seconds")
        x = trace_group["seconds"].to_numpy()
        mean = trace_group["trace_mean"].to_numpy()
        sem = trace_group["trace_sem"].to_numpy()
        trace_ax.plot(x, mean, color=color, marker="o", lw=2, label=group)
        trace_ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

        fraction_group = fraction_summary[
            fraction_summary["group"] == group
        ].sort_values("seconds")
        x_fraction = fraction_group["seconds"].to_numpy()
        for term, linestyle, marker in (
            ("linear", "-", "o"),
            ("quadratic", "--", "s"),
        ):
            term_mean = fraction_group[f"{term}_mean"].to_numpy()
            term_sem = fraction_group[f"{term}_sem"].to_numpy()
            fraction_ax.plot(
                x_fraction,
                term_mean,
                color=color,
                linestyle=linestyle,
                marker=marker,
                lw=2,
                ms=4,
                label=f"{group}: {term}",
            )
            fraction_ax.fill_between(
                x_fraction,
                term_mean - term_sem,
                term_mean + term_sem,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

        improvement_group = improvement_summary[
            improvement_summary["group"] == group
        ].sort_values("seconds")
        x_cv = improvement_group["seconds"].to_numpy()
        for baseline, linestyle, marker in (
            ("linear", "-", "o"),
            ("quadratic", "--", "s"),
        ):
            improvement = improvement_group[
                f"combined_vs_{baseline}_mean"
            ].to_numpy()
            improvement_sem = improvement_group[
                f"combined_vs_{baseline}_sem"
            ].to_numpy()
            cv_ax.plot(
                x_cv,
                improvement,
                color=color,
                linestyle=linestyle,
                marker=marker,
                lw=2,
                ms=4,
                label=f"{group}: vs {baseline}",
            )
            cv_ax.fill_between(
                x_cv,
                improvement - improvement_sem,
                improvement + improvement_sem,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    n_frames = np.arange(1, config.max_frames + 1)
    trace_ax.plot(
        n_frames / FRAME_RATE_HZ,
        1.0 / n_frames,
        color="black",
        linestyle=":",
        lw=1.5,
        label=r"$1/n$ reference",
    )
    trace_ax.set_title("A  Inferred noise scaling", loc="left", fontweight="bold")
    trace_ax.set_xlabel("Temporal averaging duration (s)")
    trace_ax.set_ylabel("Covariance trace / one-frame trace")
    trace_ax.set_ylim(bottom=0)
    trace_ax.legend(frameon=False, fontsize=8)

    fraction_ax.axhline(0.5, color="0.4", linestyle=":", lw=1)
    fraction_ax.set_title(
        "B  Animal mode-spectrum decomposition",
        loc="left",
        fontweight="bold",
    )
    fraction_ax.set_xlabel("Temporal averaging duration (s)")
    fraction_ax.set_ylabel("Fraction of fitted eigenvalue term")
    fraction_ax.set_ylim(0, 1)
    fraction_ax.legend(frameon=False, fontsize=8, ncol=2)

    cv_ax.axhline(0, color="0.4", linestyle=":", lw=1)
    cv_ax.set_title(
        "C  Held-out-mode benefit",
        loc="left",
        fontweight="bold",
    )
    cv_ax.set_xlabel("Temporal averaging duration (s)")
    cv_ax.set_ylabel(r"Single-term NRMSE $-$ combined NRMSE")
    cv_ax.legend(frameon=False, fontsize=8, ncol=2)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        rf"Direct temporal averaging; centered near frame {config.center_frame}; "
        rf"$\Delta\theta={TARGET_GAP_DEGREES:g}^\circ$; largest mode excluded "
        "from animal-averaged mode spectra",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            config.output_dir / f"temporal_averaging_main.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_mode_figure(mode_summary: pd.DataFrame, config: RunConfig) -> None:
    durations = sorted(mode_summary["n_frames"].unique())
    bin_positions = (
        mode_summary[["eigen_bin", "eigenvalue_mid"]]
        .drop_duplicates()
        .sort_values("eigen_bin")
        .set_index("eigen_bin")
    )
    if bin_positions.index.duplicated().any():
        raise AssertionError("Eigenvalue bins do not have common x positions")
    duration_colors = dict(
        zip(
            durations,
            plt.cm.viridis(np.linspace(0.12, 0.88, len(durations))),
        )
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.5),
        dpi=200,
        sharex=True,
        sharey=True,
    )

    for axis, group in zip(axes, ("FR", "CTR")):
        for n_frames in durations:
            sub = mode_summary[
                (mode_summary["group"] == group)
                & (mode_summary["n_frames"] == n_frames)
            ].set_index("eigen_bin")
            # Reindex to every common bin so NaNs break the line wherever the
            # group-duration cell contains fewer than three animals.
            sub = bin_positions.join(
                sub.drop(columns="eigenvalue_mid"),
                how="left",
            )
            x = sub["eigenvalue_mid"].to_numpy()
            mean = sub["linear_mean"].to_numpy()
            sem = sub["linear_sem"].to_numpy()
            axis.plot(
                x,
                mean,
                color=duration_colors[n_frames],
                marker="o",
                lw=2,
                ms=4,
                label=f"{n_frames / FRAME_RATE_HZ:.2f} s",
            )
            axis.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=duration_colors[n_frames],
                alpha=0.14,
                linewidth=0,
            )

        axis.axhline(0.5, color="0.4", linestyle=":", lw=1)
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean inferred covariance eigenvalue")
        axis.set_title(f"{group} animals")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(title="Averaging duration", frameon=False, fontsize=8)

    axes[0].set_ylabel("LOMO model-implied linear share")
    fig.suptitle(
        rf"Model-implied linear share across noise modes; "
        rf"$\Delta\theta={TARGET_GAP_DEGREES:g}^\circ$; "
        "largest mode excluded",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            config.output_dir / f"temporal_averaging_mode_dependence.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def run_plot_stage(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_fit_records(config)
    repeat_trace_scaling, animal_traces, trace_summary = calculate_trace_scaling(
        data["repeat_traces"]
    )
    animal_mode_values = aggregate_animal_mode_values(
        data["repeat_mode_values"],
        config,
    )
    repeat_fits, repeat_cv_errors, repeat_lomo_modes = analyze_mode_spectra(
        data["repeat_mode_values"],
        include_repeat=True,
        analysis_level="repeat_mean_mode_spectrum",
    )
    animal_fits, animal_cv_errors, animal_lomo_modes = analyze_mode_spectra(
        animal_mode_values,
        include_repeat=False,
        analysis_level="animal_mean_mode_spectrum",
    )
    fraction_summary = summarize_duration_fractions(animal_fits)
    animal_cv_improvements, cv_summary, improvement_summary = summarize_lomo_cv(
        animal_cv_errors
    )
    mode_animals, mode_summary = calculate_mode_summary(
        animal_lomo_modes,
        config,
    )

    validate_group_sizes(animal_traces, "Trace scaling")
    validate_group_sizes(animal_fits, "Duration fits")
    validate_group_sizes(animal_cv_errors, "Cross-validation results")
    validate_group_sizes(mode_animals, "Mode-dependence results")
    validate_group_duration_counts(animal_traces, "Trace scaling", config)
    validate_group_duration_counts(animal_fits, "Duration fits", config)
    validate_group_duration_counts(
        animal_cv_errors,
        "Cross-validation results",
        config,
    )

    n_animal_durations = len(config.animals) * config.max_frames
    n_retained_modes = N_SUBSAMPLE - MODE_START
    validate_analysis_row_count(
        repeat_fits,
        config.expected_fit_count,
        "Repeat mode-spectrum fits",
    )
    validate_analysis_row_count(
        animal_fits,
        n_animal_durations,
        "Animal mode-spectrum fits",
    )
    validate_analysis_row_count(
        repeat_cv_errors,
        3 * config.expected_fit_count,
        "Repeat cross-validation table",
    )
    validate_analysis_row_count(
        animal_cv_errors,
        3 * n_animal_durations,
        "Animal cross-validation table",
    )
    validate_analysis_row_count(
        repeat_lomo_modes,
        n_retained_modes * config.expected_fit_count,
        "Repeat held-out-mode table",
    )
    validate_analysis_row_count(
        animal_lomo_modes,
        n_retained_modes * n_animal_durations,
        "Animal held-out-mode table",
    )
    for fits, label in (
        (repeat_fits, "Repeat fits"),
        (animal_fits, "Animal fits"),
    ):
        fractions = fits[
            ["linear_fraction", "quadratic_fraction"]
        ].to_numpy(dtype=float)
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions < 0)
            or np.any(fractions > 1)
            or not np.allclose(fractions.sum(axis=1), 1.0)
        ):
            raise ValueError(f"{label} contain invalid component fractions")

    tables = {
        "fit_manifest.csv": data["manifest"],
        "repeat_mode_values.csv": data["repeat_mode_values"],
        "animal_mode_values.csv": animal_mode_values,
        "repeat_trace_scaling.csv": repeat_trace_scaling,
        "animal_trace_scaling.csv": animal_traces,
        "group_trace_summary.csv": trace_summary,
        "repeat_duration_fits.csv": repeat_fits,
        "animal_duration_fits.csv": animal_fits,
        "group_duration_summary.csv": fraction_summary,
        "repeat_lomo_mode_fractions.csv": repeat_lomo_modes,
        "animal_lomo_mode_fractions.csv": animal_lomo_modes,
        "repeat_cv_errors.csv": repeat_cv_errors,
        "animal_cv_errors.csv": animal_cv_errors,
        "animal_cv_improvements.csv": animal_cv_improvements,
        "group_cv_error_summary.csv": cv_summary,
        "group_cv_improvement_summary.csv": improvement_summary,
        "animal_mode_bin_fractions.csv": mode_animals,
        "group_mode_bin_summary.csv": mode_summary,
    }
    for filename, frame in tables.items():
        atomic_csv_dump(frame, config.output_dir / filename)

    plot_main_figure(
        trace_summary,
        fraction_summary,
        improvement_summary,
        config,
    )
    plot_mode_figure(mode_summary, config)

    print(
        f"Plot stage complete. Figures and {len(tables)} tables saved in "
        f"{config.output_dir.resolve()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    print(
        f"Temporal-averaging analysis: stage={args.stage}, "
        f"expected fits={config.expected_fit_count}, "
        f"output={config.output_dir.resolve()}",
        flush=True,
    )

    if args.stage in ("all", "fit"):
        run_fit_stage(config, force=args.force)
    if args.stage in ("all", "plot"):
        run_plot_stage(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed fit files remain resumable.", file=sys.stderr)
        raise SystemExit(130)
