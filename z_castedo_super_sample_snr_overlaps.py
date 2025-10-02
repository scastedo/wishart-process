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
def load_best_hp(animal, out_dir):
    """Return (best_hp: dict, best_ll: float) for this animal."""
    json_path = os.path.join(out_dir, f"animal_{animal:02d}.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["best_hp"], float(data["best_ll"])
def hp_internal_to_user(hp_int: dict) -> dict:
    """Convert evaluator/random_search dict -> your preferred schema."""
    return {
        "sigma_m": float(hp_int["l_gp_a"]),
        "gamma_gp": float(hp_int["g_gp_a"]),
        "beta_gp": float(hp_int["b_gp_a"]),
        "sigma_c": float(hp_int["l_wp_a"]),
        "gamma_wp": float(hp_int["g_wp_a"]),
        "beta_wp": float(hp_int["b_wp_a"]),
        "p": int(hp_int["p"]),
    }
def analysis(animal, sf, start, stop, array_conditions,
             save_dir=None, fname_prefix=None):
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
    good_k = ~nan_mask.all(axis=(0, 1))     # (K,)
    TEST_RESPONSE = TEST_RESPONSE[:, :, good_k]
  
    TEST_RESPONSE = jnp.transpose(TEST_RESPONSE, (2, 1, 0))  # Shape K x C x N
    N = TEST_RESPONSE.shape[2]
    C = TEST_RESPONSE.shape[1]
    PERIOD = C
    X_CONDITIONS = jnp.linspace(0, C - 1, C)
    best_hp, best_ll = load_best_hp(animal, "hp_runs/sated")
    hyperparams = hp_internal_to_user(best_hp)

    
    periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x == y) + \
        hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x - y)/PERIOD)**2/(hyperparams['sigma_m']))

    periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x == y) + \
        hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x - y)/PERIOD)**2/(hyperparams['sigma_c']))

    gp = models.GaussianProcess(kernel=periodic_kernel_gp, N=N)

    # wp = models.WishartProcess(kernel =periodic_kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    wp = models.WishartLRDProcess(kernel=periodic_kernel_wp,P=hyperparams['p'],V=1e-1*jnp.eye(N), optimize_L=False)

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


conditions_array = [12, 18, 24, 36, 48, 72, 120, 180, 360, 720,1440]
SAVE_DIR = "saved_overlap_eigs_normal_2509_smaller"  # create this folder if it doesn't exist


# FULL FR
snr_outputs_mean_fr_full = np.zeros((len(FOOD_RESTRICTED_SATED), 5, len(conditions_array)))
for i, animal in enumerate(FOOD_RESTRICTED_SATED):
    for sf in range(5):
        snr_outputs_mean_fr_full[i, sf, :], _ = analysis(
            animal, sf=sf, start=40, stop=80, array_conditions=conditions_array,
            save_dir=SAVE_DIR, fname_prefix="FR"
        )

# FULL CTR
snr_outputs_mean_ctr_full = np.zeros((len(CONTROL_SATED), 5, len(conditions_array)))
for i, animal in enumerate(CONTROL_SATED):
    for sf in range(5):
        snr_outputs_mean_ctr_full[i, sf, :], _ = analysis(
            animal, sf=sf, start=40, stop=80, array_conditions=conditions_array,
            save_dir=SAVE_DIR, fname_prefix="CTR"
        )