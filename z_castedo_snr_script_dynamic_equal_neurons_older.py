import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import optim

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import z_castedo_snr_script_dynamic as base


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
DATASET_STATE = "sated"  # "sated", "hungry", or "recovery"

FOOD_RESTRICTED_ANIMALS = [1, 2, 3, 6, 7, 8, 11, 12]
CONTROL_ANIMALS = [0, 4, 5, 9, 10, 13]

small_angle = [10.0,30.0]
N_SUBSAMPLE = 40
REPEATS = 10

RUN_DYNAMIC = False
STATIC_START = 40
STATIC_STOP = 80
TIME_WINDOW_SIZE = 5
N_TIME_WINDOWS = 10
TIME_WINDOWS = [
    (40, 45),
    (44, 49),
    (48, 53),
    (52, 57),
    (56, 61),
    (59, 64),
    (63, 68),
    (67, 72),
    (71, 76),
    (75, 80),
]

HYPERPARAM_DIR = Path("outputs/animal_hyperparam_search")
HYPERPARAM_OVERRIDES = {
    "lambda_gp_angle": 0.001,
    "lambda_gp_sf": 0.001,
}
SAVE_DIR = f"wishart_jun2_{'dynamic' if RUN_DYNAMIC else 'static'}_equal{N_SUBSAMPLE}_{DATASET_STATE}"

SUBSAMPLE_SEED = 13
INFERENCE_SEED = 13
GAMMA = 1e-5
SHARED_BETA_GP = 0.03
ADAM_OPT = 0.001
ITERATIONS = 40000
NUM_PART = 1


def configure_base_inputs():
    base.DATA_DIR = DATA_DIR
    base.DATASET_STATE = DATASET_STATE


def load_state_inputs():
    configure_base_inputs()
    return base.load_state_inputs()


def repeat_seed(base_seed, animal, repeat_idx):
    return int(base_seed) + 1000 * int(animal) + int(repeat_idx)


def sample_neuron_indices(n_full_neurons, n_subsample, seed):
    n_full_neurons = int(n_full_neurons)
    n_subsample = int(n_subsample)
    if n_subsample <= 0:
        raise ValueError(f"n_subsample must be positive, got {n_subsample}")
    if n_full_neurons < n_subsample:
        raise ValueError(
            f"Cannot subsample {n_subsample} neurons from only {n_full_neurons} available neurons"
        )

    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_full_neurons, n_subsample, replace=False)).astype(int)


def compute_noise_metrics_equal_neurons(mean_orig_mode, sigma_orig_mode, mu_test_hat, seed):
    outputs = [
        base.compute_noise_metrics_all(
            mean_orig_mode,
            sigma_orig_mode,
            mu_test_hat_gap,
            total_k="all",
            min_neurons=None,
            repeats=1,
            seed=seed,
            small_degree=24,
        )
        for mu_test_hat_gap in mu_test_hat
    ]
    output = outputs[0]
    output["dot_prod_small"] = np.stack([item["dot_prod_small"] for item in outputs], axis=0)
    output["norm_small"] = np.stack([item["norm_small"] for item in outputs], axis=0)
    return output


def validate_small_angles():
    if not 1 <= len(small_angle) <= 4:
        raise ValueError(f"small_angle must contain 1-4 degree gaps, got {small_angle}")
    if any(angle <= 0 or angle > 30 for angle in small_angle):
        raise ValueError(f"small_angle gaps must be in the range (0, 30], got {small_angle}")


def analysis(
    animal,
    start,
    stop,
    repeat_idx,
    n_subsample=N_SUBSAMPLE,
    save_dir=None,
    fname_prefix=None,
    animal_hyperparams=None,
    hyperparams_source=None,
):
    validate_small_angles()
    if animal_hyperparams is None:
        raise ValueError(f"Animal {animal}: analysis requires per-animal hyperparameters.")

    deconv, angle, sf = load_state_inputs()
    test_data = base.resort_preprocessing(deconv, angle, sf, animal)[:, :, :, :, start:stop]
    resp = jnp.nanmean(test_data, axis=-1).transpose(3, 1, 2, 0)  # K x C1 x C2 x N
    resp = resp[~jnp.isnan(resp).any(axis=(1, 2, 3))]
    if resp.shape[0] == 0:
        raise ValueError(f"Animal {animal}: no complete trials remain for window {start}:{stop}")

    _, _, _, n_full_neurons = [int(v) for v in resp.shape]
    subsample_seed = repeat_seed(SUBSAMPLE_SEED, animal, repeat_idx)
    inference_seed = repeat_seed(INFERENCE_SEED, animal, repeat_idx)
    neuron_indices = sample_neuron_indices(n_full_neurons, n_subsample, subsample_seed)
    resp = resp[..., jnp.asarray(neuron_indices)]

    seed_msg = f"subsample_seed={subsample_seed}, inference_seed={inference_seed}"
    print(
        f"Animal {animal}, repeat {repeat_idx:02d}, window {start}:{stop}, "
        f"N={n_full_neurons}->{int(n_subsample)} ({seed_msg})",
        flush=True,
    )

    k_trials, c1, c2, n_neurons = [int(v) for v in resp.shape]
    angles = jnp.arange(c1)
    sfs = jnp.array([0.02, 0.04, 0.08, 0.16, 0.32])
    sfs = jnp.log2(sfs)
    assert len(sfs) == c2, f"Expected {c2} SFs, got {len(sfs)}"

    x_full = jnp.stack(jnp.meshgrid(angles, sfs, indexing="ij"), axis=-1).reshape(-1, 2)
    y_full = resp.reshape(k_trials, c1 * c2, n_neurons).astype(jnp.float64)

    period = 12
    wp_sample_diag = GAMMA

    beta_gp = SHARED_BETA_GP

    if HYPERPARAM_OVERRIDES:
        animal_hyperparams = {**animal_hyperparams, **HYPERPARAM_OVERRIDES}
    hyperparams = base.make_effective_hyperparams(animal_hyperparams, beta_gp, GAMMA)
    diag_scale = 0.1 if hyperparams["p"] > 0 else 1.25

    periodic_gp_angle = lambda a, b: hyperparams["gamma_gp_angle"] * (
        a == b
    ) + hyperparams["beta_gp_angle"] * jnp.exp(
        -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyperparams["lambda_gp_angle"]
    )
    square_gp_sf = lambda a, b: hyperparams["gamma_gp_sf"] * (
        a == b
    ) + hyperparams["beta_gp_sf"] * jnp.exp(
        -(a - b) ** 2 / hyperparams["lambda_gp_sf"]
    )
    kernel_gp = lambda x, y: periodic_gp_angle(x[0], y[0]) * square_gp_sf(x[1], y[1])

    gp = base.models.GaussianProcess(kernel=kernel_gp, N=n_neurons)

    periodic_wp_angle = lambda a, b: hyperparams["gamma_wp_angle"] * (
        a == b
    ) + hyperparams["beta_wp_angle"] * jnp.exp(
        -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyperparams["lambda_wp_angle"]
    )
    square_wp_sf = lambda a, b: hyperparams["gamma_wp_sf"] * (
        a == b
    ) + hyperparams["beta_wp_sf"] * jnp.exp(
        -(a - b) ** 2 / hyperparams["lambda_wp_sf"]
    )
    kernel_wp = lambda x, y: periodic_wp_angle(x[0], y[0]) * square_wp_sf(x[1], y[1])

    empirical = jnp.cov((y_full - y_full.mean(0)[None]).reshape(k_trials * (c1 * c2), n_neurons).T)
    v_picked = empirical + wp_sample_diag * jnp.eye(n_neurons)

    wp = base.models.WishartLRDProcess(
        kernel=kernel_wp,
        P=int(hyperparams["p"]),
        V=v_picked,
        optimize_L=True,
        diag_scale=diag_scale,
    )
    lik = base.models.NormalConditionalLikelihood(n_neurons)
    joint = base.models.JointGaussianWishartProcess(gp, wp, lik)

    init = {"G": y_full.mean(0).T[:, None]}
    varfam = base.inference.VariationalNormal(joint.model, init=init)
    optimizer = optim.Adam(ADAM_OPT)
    key = jax.random.PRNGKey(inference_seed)
    varfam.infer(
        optimizer,
        x_full.squeeze(),
        y_full,
        n_iter=ITERATIONS,
        key=key,
        num_particles=NUM_PART,
    )
    joint.update_params(varfam.posterior)

    posterior = base.models.NormalGaussianWishartPosterior(joint, varfam, x_full)
    mean_orig_mode, sigma_orig_mode, _ = posterior.mode(x_full)

    mu_test_hat = []
    for angle_gap in small_angle:
        angle_pairs = jnp.column_stack(
            [angles, (angles + angle_gap / 30.0) % period]
        ).reshape(-1)
        x_full_small = jnp.column_stack(
            [
                jnp.repeat(angle_pairs, c2),
                jnp.tile(sfs, len(angle_pairs)),
            ]
        )
        mu_test_hat_gap, _, _ = posterior.mode(x_full_small)
        if mu_test_hat_gap.shape[0] != len(angle_pairs) * c2:
            mu_test_hat_gap = mu_test_hat_gap.transpose()
        mu_test_hat.append(mu_test_hat_gap)
    mu_test_hat = jnp.stack(mu_test_hat, axis=0)

    output = compute_noise_metrics_equal_neurons(
        mean_orig_mode,
        sigma_orig_mode,
        mu_test_hat,
        inference_seed,
    )
    snr_per_condition, snr_small = base.compute_snr_outputs(
        output["dot_prod"], output["top_evals"], output["dot_prod_small"]
    )

    saved_summary = {
        "animal": int(animal),
        "dataset_state": DATASET_STATE,
        "start": int(start),
        "stop": int(stop),
        "small_angle": [float(angle) for angle in small_angle],
        "small_angle_units": "degrees",
        "total_k": "all",
        "repeat_idx": int(repeat_idx),
        "n_full_neurons": int(n_full_neurons),
        "n_subsample": int(n_neurons),
        "neuron_indices": np.asarray(neuron_indices, dtype=int),
        "subsample_seed": int(subsample_seed),
        "inference_seed": int(inference_seed),
        "snr_per_condition": snr_per_condition,
        "snr_small": snr_small,
        "overlaps_per_condition": output["dot_prod"],
        "eigs_per_condition": output["top_evals"],
        "norm_per_condition": output["norm"],
        "overlaps_small": output["dot_prod_small"],
        "norm_small": output["norm_small"],
        "hyperparams": dict(hyperparams),
        "hyperparams_source": str(hyperparams_source) if hyperparams_source is not None else None,
        "losses": np.asarray(varfam.losses),
        "mean_orig_mode": np.asarray(mean_orig_mode),
        "sigma_orig_mode": np.asarray(sigma_orig_mode),
        "mu_test_hat": np.asarray(mu_test_hat),
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = (fname_prefix + "_") if fname_prefix else ""
        out_path = save_dir / f"{prefix}{animal}_overlaps_eigs.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(saved_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {out_path}", flush=True)

    return saved_summary


def validate_neuron_counts(animals, n_subsample):
    deconv, angle, sf = load_state_inputs()
    counts = {
        int(animal): int(base.resort_preprocessing(deconv, angle, sf, animal).shape[0])
        for animal in animals
    }
    below = {animal: count for animal, count in counts.items() if count < n_subsample}
    if below:
        raise ValueError(f"Animals below N_SUBSAMPLE={n_subsample}: {below}")
    return counts


def run_group(animals, group_prefix, start, stop, animal_hyperparams, fname_middle=None):
    for animal in animals:
        hyperparams, hyperparams_source = animal_hyperparams[int(animal)]
        for repeat_idx in range(REPEATS):
            prefix_parts = [group_prefix]
            if fname_middle is not None:
                prefix_parts.append(fname_middle)
            prefix_parts.append(f"r{repeat_idx:02d}")
            analysis(
                animal,
                start=start,
                stop=stop,
                repeat_idx=repeat_idx,
                n_subsample=N_SUBSAMPLE,
                save_dir=SAVE_DIR,
                fname_prefix="_".join(prefix_parts),
                animal_hyperparams=hyperparams,
                hyperparams_source=hyperparams_source,
            )


def main():
    validate_small_angles()
    selected_animals = FOOD_RESTRICTED_ANIMALS + CONTROL_ANIMALS
    validate_neuron_counts(selected_animals, N_SUBSAMPLE)
    animal_hyperparams = base.preload_search_hyperparams(selected_animals, HYPERPARAM_DIR)

    if RUN_DYNAMIC:
        assert len(TIME_WINDOWS) == N_TIME_WINDOWS
        assert all(stop - start == TIME_WINDOW_SIZE for start, stop in TIME_WINDOWS)
        for t_idx, (start, stop) in enumerate(TIME_WINDOWS):
            fname_middle = f"t{t_idx:02d}_{start}_{stop}"
            run_group(
                FOOD_RESTRICTED_ANIMALS,
                "FR",
                start,
                stop,
                animal_hyperparams,
                fname_middle=fname_middle,
            )
            run_group(
                CONTROL_ANIMALS,
                "CTR",
                start,
                stop,
                animal_hyperparams,
                fname_middle=fname_middle,
            )
    else:
        run_group(
            FOOD_RESTRICTED_ANIMALS,
            "FR",
            STATIC_START,
            STATIC_STOP,
            animal_hyperparams,
        )
        run_group(
            CONTROL_ANIMALS,
            "CTR",
            STATIC_START,
            STATIC_STOP,
            animal_hyperparams,
        )


if __name__ == "__main__":
    main()
