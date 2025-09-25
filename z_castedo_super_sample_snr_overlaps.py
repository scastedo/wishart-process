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

import itertools
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


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
        stim_average = np.mean(data[i, :, :, :, 40:80], axis = 3) # OKAY TO NOT HAVE NANMEAN?
        best_sf = np.argmax(np.nanmean(stim_average, axis = (0,2))).astype('int')
        best_angle = np.argmax(np.nanmean(stim_average[:,best_sf,:], axis = 1)).astype('int')
        averaged_calcium = np.nanmean(stim_average[best_angle,best_sf,:])
        
        grey_data = data[i, :, :, :, 0:20]
        # grey_data = np.concatenate((data[i, :, :, :, 0:40], data[i, :, :, :, 80:]), axis = 3)
        grey_average = np.mean(grey_data, axis = 3)
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

def calculate_overlap(mu_hat, sigma_hat, cos = False):
    num_angles = mu_hat.shape[0]
    num_evec = sigma_hat.shape[1]
    overlaps = np.zeros((num_angles, num_evec))
    eig_vals = np.zeros((num_angles, num_evec))
    for i in range(num_angles):
        eval, evec = np.linalg.eigh(sigma_hat[i,:,:])
        evec = evec[:,::-1]  # Sort eigenvectors by eigenvalues in descending order
        eval = eval[::-1]  # Sort eigenvalues in descending order
        eig_vals[i,:] = eval
        d_mu1 = mu_hat[i,:] - mu_hat[(i+1)%num_angles,:]
        d_mu2 = mu_hat[i,:] - mu_hat[(i-1)%num_angles,:]
        d_mu = d_mu1 if np.linalg.norm(d_mu1) < np.linalg.norm(d_mu2) else d_mu2        # choose signal vector with the smallest norm
        for j in range(num_evec):
            if cos:
                distance = np.power((np.dot(d_mu, evec[:,j])/ (np.linalg.norm(d_mu))),2)
            else:
                distance = np.power((np.dot(d_mu, evec[:,j])),2)
            overlaps[i,j] = distance
    return overlaps, eig_vals

import os
from pathlib import Path
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro

import glob, json
import numpy as np

def _pick_best_from_npz(path):
    z = np.load(path, allow_pickle=True)
    scores = z["scores"]
    combos = z["combos"]          # dtype=object array of dicts
    i = int(np.nanargmax(scores))
    return combos[i].item(), float(scores[i])

def load_best_hp_per_animal(save_glob="randsearch_*.npz"):
    """Return dict like {'FR_1': {'hp':{...}, 'score':...}, ...}."""
    out = {}
    for f in glob.glob(save_glob):
        tag = f.split("randsearch_")[1].split(".npz")[0]  # e.g. 'FR_1'
        hp, sc = _pick_best_from_npz(f)
        out[tag] = {"hp": hp, "score": sc}
    return out

def load_best_hp_shared(tags, per_animal_hp):
    """Mean-aggregate across a list of tags (e.g. all FR) to get a single shared HP."""
    # take the *median* of each scalar hp across animals (robust to outliers)
    keys = sorted({k for t in tags for k in per_animal_hp[t]["hp"].keys()})
    med = {}
    for k in keys:
        vals = [float(per_animal_hp[t]["hp"].get(k, np.nan)) for t in tags]
        vals = np.array([v for v in vals if np.isfinite(v)])
        if vals.size:
            med[k] = float(np.median(vals))
    return med

def translate_hp_for_analysis(hp):
    """
    Your analysis() expects:
      sigma_m, gamma_gp, beta_gp  (for GP over angle)
      sigma_c, gamma_wp, beta_wp  (for WP over angle)
      p
    The search produced:
      l_gp_a, g_gp_a, b_gp_a, l_wp_a, g_wp_a, b_wp_a, p
    """
    return {
        "sigma_m": float(hp.get("l_gp_a", 1.0)),
        "gamma_gp": float(hp.get("g_gp_a", 1e-4)),
        "beta_gp": float(hp.get("b_gp_a", 1.0)),
        "sigma_c": float(hp.get("l_wp_a", 1.0)),
        "gamma_wp": float(hp.get("g_wp_a", 1e-4)),
        "beta_wp": float(hp.get("b_wp_a", 1.0)),
        "p": int(hp.get("p", 0)),
    }

def analysis(animal, sf, start, stop, array_conditions,
             save_dir=None, fname_prefix=None,
             hyperparams_source=None,   # NEW: dict or callable(tag)->dict or None
             shared_hp=None):           # NEW: fallback shared dict (optional)
    """
    Runs the pipeline and returns SNR outputs as before,
    but also (optionally) saves overlaps & eigenvalues for later reuse.

    Parameters
    ----------
    animal : str
    sf : int or None
    start, stop : int
    array_conditions : list[int]
    save_dir : str or Path or None
        If provided, a pickle file per (animal, sf) will be written here.
    fname_prefix : str or None
        If provided, used as a prefix in the output filename, e.g., 'FR' or 'CTR'.

    Returns
    -------
    SNR_OUTPUTS : np.ndarray of shape (len(array_conditions),)
    saved_summary : dict
        A dictionary containing all the saved pieces (also pickled if save_dir is set).
    """
    # -------- your existing preprocessing --------
    if sf is None:
        TEST_DATA = resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, animal)[:, :, :, :, start:stop]
        best_test = np.zeros((TEST_DATA.shape[0], TEST_DATA.shape[1], TEST_DATA.shape[3], TEST_DATA.shape[4]))
        for i in range(TEST_DATA.shape[0]):
            for j in range(TEST_DATA.shape[1]):
                best_sf = np.argmax(jnp.nanmean(TEST_DATA[i, j, :, :, :], axis=(1, 2))).astype('int')
                best_test[i, j, :, :] = TEST_DATA[i, j, best_sf, :, :]
        TEST_RESPONSE = jnp.nanmean(best_test, axis=-1)  # Shape N x C x K
    else:
        TEST_DATA = resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, animal)[:, :, sf, :, start:stop]
        TEST_RESPONSE = jnp.nanmean(TEST_DATA, axis=-1)  # Shape N x C x K

    nan_mask = jnp.isnan(TEST_RESPONSE)     # (N, C, K)
    good_k = ~nan_mask.any(axis=(0, 1))     # (K,)
    TEST_RESPONSE = TEST_RESPONSE[:, :, good_k]

    TEST_RESPONSE = jnp.transpose(TEST_RESPONSE, (2, 1, 0))  # Shape K x C x N
    N = TEST_RESPONSE.shape[2]
    C = TEST_RESPONSE.shape[1]
    PERIOD = C
    X_CONDITIONS = jnp.linspace(0, C - 1, C)
    tag = f"{fname_prefix}_{animal}" if fname_prefix else str(animal)

    if isinstance(hyperparams_source, dict):
        # per-animal dict: {'FR_1': {'hp':{...}}, ...}
        hp_raw = hyperparams_source.get(tag, {}).get("hp", shared_hp or {})
    elif callable(hyperparams_source):
        # user-provided function: hp_raw = hyperparams_source(tag)
        hp_raw = hyperparams_source(tag)
    else:
        hp_raw = shared_hp or {}  # None → empty → falls back to defaults below

    if hp_raw:
        hyperparams = translate_hp_for_analysis(hp_raw)
    else:
        print(f"No hyperparams found for {tag}.")
    
    periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x == y) + \
        hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x - y)/PERIOD)**2/(hyperparams['sigma_m']))

    periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x == y) + \
        hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x - y)/PERIOD)**2/(hyperparams['sigma_c']))

    gp = models.GaussianProcess(kernel=periodic_kernel_gp, N=N)
    # ===== NEW: anchor V to the (K*C)-pooled covariance so noise doesn't shrink =====
    # pooled across trials & conditions: TEST_RESPONSE is (K, C, N)
    Y = np.asarray(TEST_RESPONSE)          # move to CPU NumPy
    YC = Y.reshape(-1, N)                  # (K*C, N)
    YC = YC - YC.mean(axis=0, keepdims=True)
    Sigma0 = (YC.T @ YC) / max(YC.shape[0], 1)   # (N, N)
    p_val = int(max(hyperparams['p'], 1))
    V0 = Sigma0 * float(p_val)             # so E[Σ] = V0 / p ≈ Sigma0
    V0 = jnp.array(V0, dtype=jnp.float32)
    # ===== end NEW =====

    wp = models.WishartLRDProcess(kernel=periodic_kernel_wp,
                                  P=hyperparams['p'],
                                  V=V0,
                                  optimize_L=True)  # allow learning around the anchor
    likelihood = models.NormalConditionalLikelihood(N)

    joint = models.JointGaussianWishartProcess(gp, wp, likelihood)

    inference_seed = 2
    varfam = inference.VariationalNormal(joint.model)
    adam = optim.Adam(1e-1)
    key = jax.random.PRNGKey(inference_seed)

    varfam.infer(adam, X_CONDITIONS, TEST_RESPONSE, n_iter=1000, key=key)
    joint.update_params(varfam.posterior)

    posterior = models.NormalGaussianWishartPosterior(joint, varfam, X_CONDITIONS)

    # -------- NEW: storage for saving overlaps/eigs per condition --------
    overlaps_per_condition = []  # list of np.ndarray, each shape ~ (num_angles, num_evec)
    eigs_per_condition = []      # list of np.ndarray, each shape ~ (num_angles, num_evec)
    angles_per_condition = []    # list of ints, the condition_number (i.e., #angles)
    snr_per_condition = []       # list of floats

    SNR_OUTPUTS = np.zeros((len(array_conditions)))

    # Sample & compute per requested condition grid
    for idx, condition_number in enumerate(array_conditions):
        with numpyro.handlers.seed(rng_seed=inference_seed):
            X_TEST_CONDITIONS = jnp.linspace(0, C - 1, condition_number)
            mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(X_TEST_CONDITIONS)

        overlaps_super, eigs_super = calculate_overlap(mu_test_hat, sigma_test_hat, cos=False)  # (~angles, ~evec)
        overlaps_np = np.asarray(overlaps_super)
        eigs_np = np.asarray(eigs_super)

        # SNR aggregation (as you already do)
        snr_per_angle = np.nanmean(overlaps_np / eigs_np, axis=1)
        snr_mean = np.nanmean(snr_per_angle)

        # Save into our lists
        overlaps_per_condition.append(overlaps_np)
        eigs_per_condition.append(eigs_np)
        angles_per_condition.append(int(condition_number))
        snr_per_condition.append(float(snr_mean))

        SNR_OUTPUTS[idx] = snr_mean

    # -------- package everything for saving & later reuse --------
    saved_summary = {
        "animal": animal,
        "sf": None if sf is None else int(sf),
        "start": int(start),
        "stop": int(stop),
        "N": int(N),
        "C": int(C),
        "array_conditions": [int(c) for c in array_conditions],
        "overlaps_per_condition": overlaps_per_condition,   # list of arrays
        "eigs_per_condition": eigs_per_condition,           # list of arrays
        "snr_per_condition": snr_per_condition,             # list of floats
        "hyperparams": dict(hyperparams),                   # record what was used
    }
    saved_summary["hyperparams"] = dict(hyperparams)

    # Optional: write one pickle per (animal, sf)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = (fname_prefix + "_") if fname_prefix else ""
        sf_tag = "best_sf" if sf is None else f"sf{int(sf)}"
        out_path = save_dir / f"{prefix}{animal}_{sf_tag}_overlaps_eigs.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(saved_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    return SNR_OUTPUTS, saved_summary



