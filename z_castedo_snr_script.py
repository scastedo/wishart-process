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


# SATED_DECONV = np.load('../../Data/predictions_fullTrace_sated.npy', allow_pickle=True)
SATED_DECONV = np.load('../Data/predictions_fullTrace_sated.npy', allow_pickle=True)


FOOD_RESTRICTED_SATED = [1,2,3,6,7,8,11,12]
CONTROL_SATED         = [0,4,5,9,10,13]  

# AngStim_data = '../../Data/metadata_deconv/stimAngle_sated.mat'
AngStim_data = '../Data/metadata_deconv/stimAngle_sated.mat'

ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
SATED_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']
# print(SATED_ANGLE[0].shape)

# SfStim_data = '../../Data/metadata_deconv/stimSpatFreq_sated.mat'
SfStim_data = '../Data/metadata_deconv/stimSpatFreq_sated.mat'

SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
SATED_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']
# print(SATED_SF[0].shape)

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



# def load_best_hp(animal, out_dir):
#     """Return (best_hp: dict, best_ll: float) for this animal."""
#     json_path = os.path.join(out_dir, f"animal_{animal:02d}.json")
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     return data["best_hp"], float(data["best_ll"])
# def hp_internal_to_user(hp_int: dict) -> dict:
#     """Convert evaluator/random_search dict -> your preferred schema."""
#     return {
#         "sigma_m": float(hp_int["l_gp_a"]),
#         "gamma_gp": float(hp_int["g_gp_a"]),
#         "beta_gp": float(hp_int["b_gp_a"]),
#         "sigma_c": float(hp_int["l_wp_a"]),
#         "gamma_wp": float(hp_int["g_wp_a"]),
#         "beta_wp": float(hp_int["b_wp_a"]),
#         "p": int(hp_int["p"]),
#     }
def compute_noise_metrics_all(
    y_response, covariance_fits, mu_test_hat,
    total_k, repeats, seed, small_degree):
    y_response = np.asarray(y_response)
    covariance_fits = np.asarray(covariance_fits)
    mu_test_hat = np.asarray(mu_test_hat)

    if total_k is None:
        kmax = y_response.shape[-1]
    else:
        kmax = total_k

    conditions, neurons = y_response.shape
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
        if total_k is not None and total_k < neurons:
            rng = np.random.default_rng(seed + r)
            idx = rng.choice(neurons, total_k, replace=False)
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


def analysis(animal, start, stop, small_angle,repeats,total_k,
             save_dir=None, fname_prefix=None):
    """
    Run the full analysis for one animal, including:
        - Preprocessing the data (using your existing code)
        - Fitting the Gaussian-Wishart process model
        - Computing the noise metrics (overlaps, eigenvalues, norms) for both the original angles and the small angle grid
        - Packaging everything into a dictionary for saving
    """
    # -------- your existing preprocessing --------

    TEST_DATA = resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, animal)[:, :, :, :, start:stop]
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
    ITERATIONS = 50000
    NUM_PART = 1


    mu = Y_FULL.mean(axis=0)
    mu_var = jnp.var(mu, axis=0)
    v = mu_var + 1e-12
    v_geo = jnp.exp(jnp.mean(jnp.log(v)))
    BETA_GP = float(jnp.sqrt(v_geo))

    hyperparams = {
        'lambda_gp_angle': 0.05,
        'gamma_gp_angle':GAMMA,
        'beta_gp_angle': BETA_GP,

        'lambda_gp_sf': 5,
        'gamma_gp_sf': GAMMA,
        'beta_gp_sf': BETA_GP,

        'lambda_wp_angle': 0.8,
        'gamma_wp_angle': GAMMA,
        'beta_wp_angle': 1.,

        'lambda_wp_sf': 1.5,
        'gamma_wp_sf': GAMMA,
        'beta_wp_sf': 1.,
        'p': 0
    }
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

    output = compute_noise_metrics_all(mean_orig_mode, sigma_orig_mode, mu_test_hat, total_k, repeats, SEED,small_angle)

   
    # -------- package everything for saving & later reuse --------
    saved_summary = {
        "animal": animal,
        "small_angle": small_angle,
        "total_k": total_k,
        "overlaps_per_condition": output["dot_prod"],   # list of arrays
        "eigs_per_condition": output["top_evals"],           # list of arrays
        "norm_per_condition": output["norm"],              # list of arrays
        "overlaps_small": output["dot_prod_small"],       # list of arrays
        "norm_small": output["norm_small"],              # list of arrays
        "hyperparams": dict(hyperparams),                   # record what was used
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
number_neurons = []
for i in range(14):
    x =resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, i)
    number_neurons.append(x.shape[0])

MIN_NEURONS = None# min(number_neurons)  #OR None
REPEATS = 100
SAVE_DIR = "wishart_april"  # create this folder if it doesn't exist

for i, animal in enumerate(FOOD_RESTRICTED_SATED):
    analysis(
        animal,start=40, stop=80, small_angle=small_angle, repeats=REPEATS, total_k=MIN_NEURONS,
        save_dir=SAVE_DIR, fname_prefix="FR"
        )
for i, animal in enumerate(CONTROL_SATED):
    analysis(
        animal,  start=40, stop=80, small_angle=small_angle, repeats=REPEATS, total_k=MIN_NEURONS,
        save_dir=SAVE_DIR, fname_prefix="CTR"
    )