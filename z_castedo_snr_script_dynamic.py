import sys
import jax
from numpyro import optim
jax.config.update('jax_enable_x64', True)  # Use float64
jax.config.update("jax_default_matmul_precision", "highest")
# sys.path.append('wishart-process')
import inference
import models
import visualizations
import evaluation
import utils

import jax.numpy as jnp
import numpyro
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import pickle
import os, json


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
DATASET_STATE = "hungry"  # "sated", "hungry", or "recovery"
VALID_DATASET_STATES = {"sated", "hungry", "recovery"}


FOOD_RESTRICTED_ANIMALS = [1,2,3,6,7,8,11,12]
CONTROL_ANIMALS         = [0,4,5,9,10,13]  

STATE_DECONV = None
STATE_ANGLE = None
STATE_SF = None
LOADED_DATASET_STATE = None


def load_state_inputs():
    global STATE_DECONV, STATE_ANGLE, STATE_SF, LOADED_DATASET_STATE

    state = DATASET_STATE.lower()
    if state not in VALID_DATASET_STATES:
        raise ValueError(f"DATASET_STATE must be one of {sorted(VALID_DATASET_STATES)}, got {DATASET_STATE!r}")

    if LOADED_DATASET_STATE != state:
        deconv_filename = f"predictions_fullTrace_{state}.npy"
        deconv_candidates = [
            DATA_DIR / "full_trace" / deconv_filename,
            DATA_DIR / deconv_filename,
        ]
        deconv_path = next((path for path in deconv_candidates if path.is_file()), None)
        if deconv_path is None:
            tried = "\n".join(f"  - {path}" for path in deconv_candidates)
            raise FileNotFoundError(f"Could not find deconvolved {state!r} data. Tried:\n{tried}")
        angle_path = DATA_DIR / "metadata_deconv" / f"stimAngle_{state}.mat"
        sf_path = DATA_DIR / "metadata_deconv" / f"stimSpatFreq_{state}.mat"

        STATE_DECONV = np.load(deconv_path, allow_pickle=True)
        ang_stim_data = loadmat(angle_path, simplify_cells=True)
        STATE_ANGLE = ang_stim_data['order_of_stim_arossAnimals']
        sf_stim_data = loadmat(sf_path, simplify_cells=True)
        STATE_SF = sf_stim_data['stimSpatFreq_arossAnimals']
        LOADED_DATASET_STATE = state

    return STATE_DECONV, STATE_ANGLE, STATE_SF

def resort_preprocessing(datum,angle_arr,sf_arr,animal):
    data = np.copy(datum[animal,:])
    neurons = data[0].shape[0]
    reshape_data = np.full((60,neurons,data[0].shape[1]), np.nan)
    for i in range(60):
        reshape_data[i,:,:] = data[i]

    reshape_data = reshape_data.reshape(60,neurons,12,120)
    reshape_data = np.transpose(reshape_data,(1,2,0,3))
    #Remove first two neurons
    reshape_data = reshape_data[2:,:,:,:]

    #Remove None trials
    max_trial = np.argmax(np.isnan(reshape_data[0,1,:,0]))
    reshape_data = reshape_data[:,:,:max_trial,:]

    # Remove beginning and last bit # HMMMM should I do this?
    # reshape_data[:,0,:,:32] = np.nan
    # reshape_data[:,-1,:,88:] = np.nan
    # print(np.any(np.isnan(reshape_data)))
    # print(reshape_data.shape)
    
    # Reorder angles
    angles = np.copy(angle_arr[animal])
    for itrials in range(angles.shape[1]):
        order = angles[:,itrials]-1
        reshape_data[:,:,itrials,:] = reshape_data[:,order,itrials,:]

    # Reorder SFs
    reshaped_data = []
    sfs = np.copy(sf_arr[animal])
    for experiment in range(1,6):
        mask = sfs == experiment
        reshaped_data.append(reshape_data[:,:,mask,:])

    max_trials = max([exp.shape[2] for exp in reshaped_data])
    # Pad the data for experiments with fewer trials
    for i in range(len(reshaped_data)):
        if reshaped_data[i].shape[2] < max_trials:
            padding = max_trials - reshaped_data[i].shape[2]
            reshaped_data[i] = np.pad(reshaped_data[i], ((0, 0),(0, 0),(0, padding),(0, 0)), mode='constant', constant_values=np.nan)

    reshaped_data = np.stack(reshaped_data,axis=2)    

    return reshaped_data

def remove_neurons(datum, angles,sfs, animal, count = False):
    neurons_to_keep = []
    data = resort_preprocessing(datum,angles,sfs,animal)
    number_neurons = data.shape[0]    
    for i in range(number_neurons):
        # print(np.nanmean(data[i, :, :, :, 40:80], axis = 3))
        stim_average = np.nanmean(data[i, :, :, :, 40:80], axis = 3) # OKAY TO NOT HAVE NANMEAN?
        best_sf = np.argmax(np.nanmean(stim_average, axis = (0,2))).astype('int')
        best_angle = np.argmax(np.nanmean(stim_average[:,best_sf,:], axis = 1)).astype('int')
        averaged_calcium = np.nanmean(stim_average[best_angle,best_sf,:])
        
        grey_data = data[i, :, :, :, 0:20]
        # grey_data = np.concatenate((data[i, :, :, :, 0:40], data[i, :, :, :, 80:]), axis = 3)
        grey_average = np.nanmean(grey_data, axis = 3)
        best_sf = np.argmax(np.nanmean(grey_average, axis = (0,2))).astype(int)
        best_angle = np.argmax(np.nanmean(grey_average[:,best_sf,:], axis = 1)).astype(int)
        average_grey = np.nanmean(grey_average[best_angle,best_sf,:])
        std_grey = np.nanstd(grey_average[best_angle,best_sf,:])
        
        if np.abs(averaged_calcium - average_grey) >= 1.69*std_grey:
            neurons_to_keep.append(i)
    
    # Keep only the neurons that meet the condition
    data_filtered = data[neurons_to_keep, :, :,:,:]
    if count:
        return data_filtered.shape[0]/data.shape[0]

    return data_filtered



REQUIRED_SEARCH_HYPERPARAMS = (
    "lambda_gp_angle",
    "lambda_gp_sf",
    "lambda_wp_angle",
    "lambda_wp_sf",
    "p",
)


def load_animal_search_hyperparams(animal, hyperparam_dir):
    hyperparam_dir = Path(hyperparam_dir)
    source_path = hyperparam_dir / f"animal_{int(animal):02d}_hyperparams.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Animal {animal}: missing hyperparameter result {source_path}")

    with open(source_path, "r") as f:
        result = json.load(f)

    status = result.get("status")
    if status != "completed":
        raise ValueError(f"Animal {animal}: hyperparameter result is not completed: {status!r}")

    hyperparams = result.get("final_best_hyperparams")
    if not isinstance(hyperparams, dict):
        raise ValueError(f"Animal {animal}: missing final_best_hyperparams in {source_path}")

    missing = [key for key in REQUIRED_SEARCH_HYPERPARAMS if hyperparams.get(key) is None]
    if missing:
        raise ValueError(f"Animal {animal}: missing required hyperparameters {missing} in {source_path}")

    hyperparams = dict(hyperparams)
    for key in (
        "lambda_gp_angle",
        "lambda_gp_sf",
        "lambda_wp_angle",
        "lambda_wp_sf",
        "gamma_gp_angle",
        "gamma_gp_sf",
        "gamma_wp_angle",
        "gamma_wp_sf",
        "beta_wp_angle",
        "beta_wp_sf",
    ):
        if hyperparams.get(key) is not None:
            hyperparams[key] = float(hyperparams[key])
    hyperparams["p"] = int(hyperparams["p"])

    return hyperparams, source_path


def preload_search_hyperparams(animals, hyperparam_dir):
    return {
        int(animal): load_animal_search_hyperparams(animal, hyperparam_dir)
        for animal in animals
    }


def value_or_default(mapping, key, default):
    value = mapping.get(key, default)
    if value is None:
        value = default
    return value


def make_effective_hyperparams(search_hyperparams, beta_gp, gamma_default):
    return {
        'lambda_gp_angle': float(search_hyperparams["lambda_gp_angle"]),
        'gamma_gp_angle': float(value_or_default(search_hyperparams, "gamma_gp_angle", gamma_default)),
        'beta_gp_angle': beta_gp,

        'lambda_gp_sf': float(search_hyperparams["lambda_gp_sf"]),
        'gamma_gp_sf': float(value_or_default(search_hyperparams, "gamma_gp_sf", gamma_default)),
        'beta_gp_sf': beta_gp,

        'lambda_wp_angle': float(search_hyperparams["lambda_wp_angle"]),
        'gamma_wp_angle': float(value_or_default(search_hyperparams, "gamma_wp_angle", gamma_default)),
        'beta_wp_angle': float(value_or_default(search_hyperparams, "beta_wp_angle", 1.0)),

        'lambda_wp_sf': float(search_hyperparams["lambda_wp_sf"]),
        'gamma_wp_sf': float(value_or_default(search_hyperparams, "gamma_wp_sf", gamma_default)),
        'beta_wp_sf': float(value_or_default(search_hyperparams, "beta_wp_sf", 1.0)),
        'p': int(search_hyperparams["p"]),
    }


def compute_noise_metrics_all(
    y_response, covariance_fits, mu_test_hat,
    total_k, min_neurons,repeats, seed, small_degree):
    y_response = np.asarray(y_response)
    covariance_fits = np.asarray(covariance_fits)
    mu_test_hat = np.asarray(mu_test_hat)

    conditions, neurons = y_response.shape

    if total_k == "all":
        kmax = neurons
    elif total_k is None:
        kmax = min_neurons
    else:
        kmax = total_k

    c1 = 12
    c2 = 5
    assert conditions == c1 * c2, f"Expected conditions to be {c1*c2}, got {conditions}"

    resp4 = y_response.reshape(c1, c2, neurons)
    substep = small_degree // c1
    resp4_small = mu_test_hat.reshape(small_degree, c2, neurons)
    signal_vectors = resp4[:, np.newaxis, :, :] - resp4[np.newaxis, :, :, :]
    covariance_results = covariance_fits.reshape(c1, c2, neurons, neurons)
    

    dot_prod = np.full((c1, c1 - 1, c2, kmax, repeats), np.nan)
    norm = np.full((c1, c1 - 1, c2, repeats), np.nan)
    top_evals = np.full((c1, c2, kmax, repeats), np.nan)

    dot_prod_small = np.full((c1, c2, kmax, repeats), np.nan)
    norm_small = np.full((c1, c2, repeats), np.nan)


    for r in range(repeats):
        if total_k is None:
            rng = np.random.default_rng(seed + r)
            idx = rng.choice(neurons, kmax, replace=False)
            sig_vec = signal_vectors[:, :, :, idx]
            resp4_small_r = resp4_small[:, :, idx]
            covariance_repeats = np.take(covariance_results, idx, axis=2)
            covariance_repeats = np.take(covariance_repeats, idx, axis=3)
        else:
            sig_vec = signal_vectors
            resp4_small_r = resp4_small
            covariance_repeats = covariance_results

        for sf in range(c2):
            for ia in range(c1):
                covariance = covariance_repeats[ia, sf, :, :]
                try:
                    evals, evecs = np.linalg.eigh(covariance)
                except np.linalg.LinAlgError:
                    continue

                order = np.argsort(evals)[::-1]
                sorted_evals = evals[order]
                sorted_evecs = evecs[:, order]
                top_evals[ia, sf, :, r] = sorted_evals[:kmax]

                others = (ia + np.arange(1, c1)) % c1
                for j, ja in enumerate(others):
                    sv = sig_vec[ia, ja, sf, :]
                    norm[ia, j, sf, r] = np.linalg.norm(sv)
                    for k in range(kmax):
                        pc = sorted_evecs[:, k]
                        dot_prod[ia, j, sf, k, r] = np.dot(sv, pc) ** 2
                
                ia_small = ia*substep
                ja_small = (ia_small+1) % small_degree
                sv_small = resp4_small_r[ia_small, sf, :] - resp4_small_r[ja_small, sf, :]
                norm_small[ia, sf, r] = np.linalg.norm(sv_small)
                for k in range(kmax):
                    pc = sorted_evecs[:, k]
                    dot_prod_small[ia, sf, k, r] = np.dot(sv_small, pc) ** 2

    return {
        "dot_prod": dot_prod,
        "top_evals": top_evals,
        "norm": norm,
        "dot_prod_small": dot_prod_small,
        "norm_small": norm_small,
    }


def compute_snr_outputs(overlaps_per_condition, eigs_per_condition, overlaps_small):
    # Fisher discriminability: sum_k (signal dot eigenvector_k)^2 / eigenvalue_k.
    eigs_pos = np.where(eigs_per_condition > 0, eigs_per_condition, np.nan)
    snr_per_condition = np.nansum(
        overlaps_per_condition / eigs_pos[:, None, ...],
        axis=-2,
    )
    snr_small = np.nansum(overlaps_small / eigs_pos, axis=-2)
    return snr_per_condition, snr_small


def analysis(animal, start, stop, small_angle,repeats,total_k,min_neurons,
             save_dir=None, fname_prefix=None, animal_hyperparams=None, hyperparams_source=None):
    """
    Run the full analysis for one animal, including:
        - Preprocessing the data (using your existing code)
        - Fitting the Gaussian-Wishart process model
        - Computing the noise metrics (overlaps, eigenvalues, norms) for both the original angles and the small angle grid
        - Packaging everything into a dictionary for saving
    """
    # -------- your existing preprocessing --------

    if animal_hyperparams is None:
        raise ValueError(f"Animal {animal}: analysis requires per-animal hyperparameters.")

    deconv, angle, sf = load_state_inputs()
    TEST_DATA = resort_preprocessing(deconv, angle, sf, animal)[:, :, :, :, start:stop]
    resp = jnp.nanmean(TEST_DATA, axis=-1).transpose(3, 1, 2, 0)  # K x C1 x C2 x N
    resp = resp[~jnp.isnan(resp).any(axis=(1, 2, 3))]


    SEED = 13
    K, C1, C2, N = resp.shape
    angles = jnp.arange(C1)  # Angles from 0 to 330 degrees in 30 degree increments. Here we just use indices 0-11
    sfs = jnp.array([0.02, 0.04, 0.08, 0.16, 0.32]) 
    sfs = jnp.log2(sfs)  # Use log2 of spatial frequencies so that differences are equally spaced
    assert len(sfs) == C2, f"Expected {C2} SFs, got {len(sfs)}"

    X_FULL = jnp.stack(jnp.meshgrid(angles, sfs, indexing="ij"), axis=-1).reshape(-1, 2)
    Y_FULL = resp.reshape(K, C1 * C2, N).astype(np.float64, copy=False)
    PERIOD = 12
    GAMMA = 1e-5# GAMMA set small for stability
    ADAM_OPT = 0.001
    wp_sample_diag = GAMMA  # For numerical stability
    ITERATIONS = 80000
    NUM_PART = 1


    mu = Y_FULL.mean(axis=0)
    mu_var = jnp.var(mu, axis=0)
    v = mu_var + 1e-12
    v_geo = jnp.exp(jnp.mean(jnp.log(v)))
    BETA_GP = float(jnp.sqrt(v_geo))

    hyperparams = make_effective_hyperparams(animal_hyperparams, BETA_GP, GAMMA)
    if hyperparams["p"] > 0:
        DIAG_SCALE = 0.1
    else:
        DIAG_SCALE = 1.25 

    # Define kernels
    periodic_gp_angle = lambda a, b: hyperparams["gamma_gp_angle"] * (a == b) + hyperparams["beta_gp_angle"] * jnp.exp(
        -jnp.sin(jnp.pi * jnp.abs(a - b) / PERIOD) ** 2 / hyperparams["lambda_gp_angle"]
    )
    square_gp_sf = lambda a, b: hyperparams["gamma_gp_sf"] * (a == b) + hyperparams["beta_gp_sf"] * jnp.exp(
        -(a - b) ** 2 / hyperparams["lambda_gp_sf"]
    )
    kernel_gp = lambda x, y: periodic_gp_angle(x[0], y[0]) * square_gp_sf(x[1], y[1])

    gp = models.GaussianProcess(kernel=kernel_gp, N=N)


    periodic_wp_angle = lambda a, b: hyperparams["gamma_wp_angle"] * (a == b) + hyperparams["beta_wp_angle"] * jnp.exp(
        -jnp.sin(jnp.pi * jnp.abs(a - b) / PERIOD) ** 2 / hyperparams["lambda_wp_angle"]
    )
    square_wp_sf = lambda a, b: hyperparams["gamma_wp_sf"] * (a == b) + hyperparams["beta_wp_sf"] * jnp.exp(
        -(a - b) ** 2 / hyperparams["lambda_wp_sf"]
    )
    kernel_wp = lambda x, y: periodic_wp_angle(x[0], y[0]) * square_wp_sf(x[1], y[1])

    empirical = jnp.cov((Y_FULL-Y_FULL.mean(0)[None]).reshape(K*(C1*C2),N).T)
    V_Picked = empirical + wp_sample_diag * jnp.eye(N)
    # V_Picked = empirical + 1e-3 * np.trace(empirical)/N * jnp.eye(N)    

    wp = models.WishartLRDProcess(kernel=kernel_wp, P=int(hyperparams["p"]), V=V_Picked, optimize_L=True, diag_scale=DIAG_SCALE)
    lik = models.NormalConditionalLikelihood(N)
    joint = models.JointGaussianWishartProcess(gp, wp, lik)


    init = {'G':Y_FULL.mean(0).T[:,None]}
    varfam = inference.VariationalNormal(joint.model, init=init)
    optimizer = optim.Adam(ADAM_OPT)
    key = jax.random.PRNGKey(SEED)
    varfam.infer(optimizer, X_FULL.squeeze(), Y_FULL, n_iter=ITERATIONS, key=key, num_particles = NUM_PART)
    joint.update_params(varfam.posterior)


    posterior = models.NormalGaussianWishartPosterior(joint, varfam, X_FULL)
    mean_orig_mode, sigma_orig_mode, _= posterior.mode(X_FULL)

    angles_small = jnp.linspace(0, C1, small_angle, endpoint=False)
    X_FULL_SMALL = jnp.column_stack([
        jnp.repeat(angles_small, C2),
        jnp.tile(sfs, small_angle),
    ])
    mu_test_hat, _, _ = posterior.mode(X_FULL_SMALL)
    if mu_test_hat.shape[0] != small_angle * C2:
        mu_test_hat =mu_test_hat.transpose()

    output = compute_noise_metrics_all(mean_orig_mode, sigma_orig_mode, mu_test_hat, total_k,min_neurons, repeats, SEED,small_angle)
    snr_per_condition, snr_small = compute_snr_outputs(
        output["dot_prod"], output["top_evals"], output["dot_prod_small"]
    )

   
    # -------- package everything for saving & later reuse --------
    saved_summary = {
        "animal": animal,
        "dataset_state": DATASET_STATE,
        "start": int(start),
        "stop": int(stop),
        "small_angle": small_angle,
        "total_k": total_k,
        "snr_per_condition": snr_per_condition,
        "snr_small": snr_small,
        "overlaps_per_condition": output["dot_prod"],   # list of arrays
        "eigs_per_condition": output["top_evals"],           # list of arrays
        "norm_per_condition": output["norm"],              # list of arrays
        "overlaps_small": output["dot_prod_small"],       # list of arrays
        "norm_small": output["norm_small"],              # list of arrays
        "hyperparams": dict(hyperparams),                   # record what was used
        "hyperparams_source": str(hyperparams_source) if hyperparams_source is not None else None,
        "losses": np.asarray(varfam.losses),                        # useful for debugging/tracing
        "mean_orig_mode": np.asarray(mean_orig_mode),                # the fitted mean (C1*C2, N)
        "sigma_orig_mode": np.asarray(sigma_orig_mode),              # the fitted covariance (C1*C2, N, N)
        "mu_test_hat": np.asarray(mu_test_hat),                    # the GP mean predictions at the small angle grid (small_angle*C2, N)
    }

    # Optional: write one pickle per (animal, sf)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = (fname_prefix + "_") if fname_prefix else ""
        out_path = save_dir / f"{prefix}{animal}_overlaps_eigs.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(saved_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return saved_summary


small_angle = 48
USE_ALL_NEURONS = False
TOTAL_K = "all" if USE_ALL_NEURONS else None
REPEATS = 1 if USE_ALL_NEURONS else 100

RUN_DYNAMIC = False
STATIC_START = 40
STATIC_STOP = 80
TIME_WINDOW_SIZE = 5
N_TIME_WINDOWS = 10
TIME_WINDOWS = [(40,45), (44,49), (48,53), (52,57), (56,61), (59,64), (63,68), (67,72), (71,76), (75,80)]

HYPERPARAM_DIR = Path("outputs/animal_hyperparam_search")
SAVE_DIR = f"wishart_may27_{'dynamic' if RUN_DYNAMIC else 'static'}_{DATASET_STATE}"  # create this folder if it doesn't exist


def main():
    selected_animals = FOOD_RESTRICTED_ANIMALS + CONTROL_ANIMALS
    animal_hyperparams = preload_search_hyperparams(selected_animals, HYPERPARAM_DIR)

    deconv, angle, sf = load_state_inputs()
    number_neurons = []
    for i in range(14):
        x =resort_preprocessing(deconv, angle, sf, i)
        number_neurons.append(x.shape[0])

    min_neurons = min(number_neurons)  #OR None

    if RUN_DYNAMIC:
        assert len(TIME_WINDOWS) == N_TIME_WINDOWS
        assert all(stop - start == TIME_WINDOW_SIZE for start, stop in TIME_WINDOWS)
        for t_idx, (start, stop) in enumerate(TIME_WINDOWS):
            for i, animal in enumerate(FOOD_RESTRICTED_ANIMALS):
                hyperparams, hyperparams_source = animal_hyperparams[int(animal)]
                analysis(
                    animal,start=start, stop=stop, small_angle=small_angle, repeats=REPEATS, total_k=TOTAL_K,min_neurons=min_neurons,
                    save_dir=SAVE_DIR, fname_prefix=f"FR_t{t_idx:02d}_{start}_{stop}",
                    animal_hyperparams=hyperparams, hyperparams_source=hyperparams_source
                    )
            for i, animal in enumerate(CONTROL_ANIMALS):
                hyperparams, hyperparams_source = animal_hyperparams[int(animal)]
                analysis(
                    animal,  start=start, stop=stop, small_angle=small_angle, repeats=REPEATS, total_k=TOTAL_K,min_neurons=min_neurons,
                    save_dir=SAVE_DIR, fname_prefix=f"CTR_t{t_idx:02d}_{start}_{stop}",
                    animal_hyperparams=hyperparams, hyperparams_source=hyperparams_source
                )
    else:
        for i, animal in enumerate(FOOD_RESTRICTED_ANIMALS):
            hyperparams, hyperparams_source = animal_hyperparams[int(animal)]
            analysis(
                animal,start=STATIC_START, stop=STATIC_STOP, small_angle=small_angle, repeats=REPEATS, total_k=TOTAL_K,min_neurons=min_neurons,
                save_dir=SAVE_DIR, fname_prefix="FR",
                animal_hyperparams=hyperparams, hyperparams_source=hyperparams_source
                )
        for i, animal in enumerate(CONTROL_ANIMALS):
            hyperparams, hyperparams_source = animal_hyperparams[int(animal)]
            analysis(
                animal,  start=STATIC_START, stop=STATIC_STOP, small_angle=small_angle, repeats=REPEATS, total_k=TOTAL_K,min_neurons=min_neurons,
                save_dir=SAVE_DIR, fname_prefix="CTR",
                animal_hyperparams=hyperparams, hyperparams_source=hyperparams_source
            )


if __name__ == "__main__":
    main()
