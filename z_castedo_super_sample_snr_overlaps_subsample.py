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
CONTROL_SATED         = [0,4,5,9,10,13]  # 5 remove later?

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

def compute_signal_vectors(response, t0=40, t1=80):
    """
    Compute Δμ for each ROI, SF, and pair of angles.
    response: (n_rois, n_angles, n_sfs, n_trials, n_time)
    Returns:
      Δμ: (n_rois, n_angles, n_angles, n_sfs)
    """
    # average over the chosen time window
    resp = np.nanmean(response[..., t0:t1], axis=-1)  # → (n_rois, n_angles, n_sfs, n_trials)
    # average over trials
    mean_resp = np.nanmean(resp, axis=-1)             # → (n_rois, n_angles, n_sfs)
    # subtract pairwise to get Δμ[:, i, j, sf]
    dm = mean_resp[:, :, np.newaxis, :] - mean_resp[:, np.newaxis, :, :]
    
    # give a random normalised vector  of shape dm
    # dm = np.random.randn(*dm.shape)  # random normalised vector
    return dm  # shape (n_rois, n_angles, n_angles, n_sfs)
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
            # for j in range(TEST_DATA.shape[1]):
            best_sf = np.argmax(jnp.nanmean(TEST_DATA[i, :, :, :, 40:80], axis=(0, 2, 3))).astype('int')
            best_test[i, :, :, :] = TEST_DATA[i, :, best_sf, :, :]
        TEST_RESPONSE = jnp.nanmean(best_test, axis=-1)  # Shape N x C x K
    else:
        TEST_DATA = resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, animal)[:, :, sf, :, start:stop]
        TEST_RESPONSE = jnp.nanmean(TEST_DATA, axis=-1)  # Shape N x C x K


    good_trials = ~jnp.isnan(TEST_RESPONSE).any(axis=(0, 1))   # shape (K,)
    # nan_mask = jnp.isnan(TEST_RESPONSE)     # (N, C, K)
    # good_k = ~nan_mask.all(axis=(0, 1))     # (K,)
    TEST_RESPONSE_full = TEST_RESPONSE[:, :, good_trials]
    N_full = int(TEST_RESPONSE_full.shape[0])
    C = int(TEST_RESPONSE_full.shape[1])
    PERIOD = C
    array_conditions = [int(c) for c in array_conditions]
    n_cond = len(array_conditions)
    X_CONDITIONS_ALL = jnp.linspace(0, C - 1, C)
    best_hp, best_ll = load_best_hp(animal, "hp_runs/sated")
    hyperparams = hp_internal_to_user(best_hp)
    
    periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x == y) + \
        hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x - y)/PERIOD)**2/(hyperparams['sigma_m']))

    periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x == y) + \
        hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x - y)/PERIOD)**2/(hyperparams['sigma_c']))

    num_repeats = REPEATS
    overlaps_all= []
    eigs_all = []
    for r in range(num_repeats):
        # ---- random subsample of neurons ----
        if num_repeats>1:
            rng = np.random.default_rng(2+r)
            idx_random = rng.choice(N_full, MIN_NEURONS, replace=False)
            TR = TEST_RESPONSE_full[idx_random, :, :]          # (MIN x C x K)
            TR = jnp.transpose(TR, (2, 1, 0))           # K x C x N_sub
            N_sub = int(TR.shape[2])
            modes = N_sub
        else:
            TR = TEST_RESPONSE_full
            N_sub = N_full
            modes = N_sub

        gp = models.GaussianProcess(kernel=periodic_kernel_gp, N=N_sub)
        # wp = models.WishartProcess(kernel =periodic_kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
        wp = models.WishartLRDProcess(kernel=periodic_kernel_wp,P=hyperparams['p'],V=1e-1*jnp.eye(N_sub), optimize_L=False)
        likelihood = models.NormalConditionalLikelihood(N_sub)
        joint = models.JointGaussianWishartProcess(gp, wp, likelihood)

        inference_seed = 2+r
        varfam = inference.VariationalNormal(joint.model)
        adam = optim.Adam(1e-1)
        key = jax.random.PRNGKey(inference_seed)

        varfam.infer(adam, X_CONDITIONS_ALL, TR, n_iter=2000, key=key)
        joint.update_params(varfam.posterior)
        
  
        posterior = models.NormalGaussianWishartPosterior(joint, varfam, X_CONDITIONS_ALL)
        # Sample from the posterior
        
        
        # with numpyro.handlers.seed(rng_seed=inference_seed + 1000*r):
        #     mu_orig, sigma_orig, _ = posterior.sample(X_CONDITIONS_ALL)


        n_draws = 50
        mu_draws = []
        sig_draws = []

        for d in range(n_draws):
            with numpyro.handlers.seed(rng_seed=inference_seed + 1000*r + 17*d):
                mu_d, sigma_d, _ = posterior.sample(X_CONDITIONS_ALL)  # (C,N), (C,N,N)
            mu_draws.append(np.array(mu_d))
            sig_draws.append(np.array(sigma_d))

        mu_orig = np.mean(mu_draws, axis=0)       # (C, N_sub)
        sigma_orig = np.mean(sig_draws, axis=0)   # (C, N_sub, N_sub)    





        overlaps_list = []
        eigs_list = []     

        # Sample & compute per requested condition grid
        for idx, condition_number in enumerate(array_conditions):
            X_TEST_CONDITIONS = jnp.linspace(0, C - 1, condition_number)
            
            
            
            # with numpyro.handlers.seed(rng_seed=inference_seed + 1000*r + 10*idx):
            #     mu_test_hat, _, _ = posterior.sample(X_TEST_CONDITIONS)

            n_draws = 50
            mu_draws = []
            for d in range(n_draws):
                with numpyro.handlers.seed(rng_seed=inference_seed + 1000*r + 10*idx+ 17*d):
                    mu_d, _, _ = posterior.sample(X_TEST_CONDITIONS)  # (C,N), (C,N,N)
                mu_draws.append(np.array(mu_d))
            mu_test_hat = np.mean(mu_draws, axis=0)       # (C, N_sub)

            overlaps_array = np.full((12, modes), np.nan)
            eigs_array = np.full((12, modes), np.nan)
            for og_conds in range(12):
                eigvals, eigvecs = np.linalg.eigh(sigma_orig[og_conds,:,:])
                eigval_order = np.argsort(eigvals)[::-1]
                eigvals = eigvals[eigval_order] 
                eigvecs = eigvecs[:,eigval_order] 
                eigs_array[og_conds,:] = eigvals[:modes] # could also be the last modes

                # find Delta Mu for each og_cond
                number_angles,number_evals = mu_test_hat.shape
                equivalent_cond = int((og_conds * condition_number) // C)
                d_mu = mu_orig[og_conds,:] - mu_test_hat[(equivalent_cond+1)%number_angles,:]
                
                for evallss in range(modes): # could also be the last modes
                    overlap = np.power((np.dot(d_mu, eigvecs[:,evallss])),2)/np.power(np.linalg.norm(d_mu)*np.linalg.norm(eigvecs[:,evallss]),2)
                    overlaps_array[og_conds,evallss] = overlap 
                    # overlaps_array[og_conds,evallss] = np.power((np.dot(d_mu, eigvecs[:,evallss])),2)
            overlaps_list.append(overlaps_array)
            eigs_list.append(eigs_array)
        overlaps_all.append(overlaps_list)
        eigs_all.append(eigs_list)
    # -------- package everything for saving & later reuse --------
    saved_summary = {
        "animal": animal,
        "sf": None if sf is None else int(sf),
        "start": int(start),
        "stop": int(stop),
        # "N": int(N),
        "C": int(C),
        "array_conditions": [int(c) for c in array_conditions],
        "overlaps_per_condition": overlaps_all,   # list of arrays
        "eigs_per_condition": eigs_all,           # list of arrays
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


conditions_array = [12,36]
number_neurons = []
for i in range(14):
    x =resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, i)
    number_neurons.append(x.shape[0])



#DO wE DO ALL SFs OR BEST SF ONLY?
#Do we do subsampling or not (top or bottom 50 neurons)

MIN_NEURONS = min(number_neurons)
REPEATS = 50
SAVE_DIR = "figure_creation_jan_15_subsampled_all_sf_cos"  # create this folder if it doesn't exist
# FULL FR
# snr_outputs_mean_fr_full = np.zeros((len(FOOD_RESTRICTED_SATED), len(conditions_array)))
# for i, animal in enumerate(FOOD_RESTRICTED_SATED):
#     analysis(
#         animal, sf=None, start=40, stop=80, array_conditions=conditions_array,
#         save_dir=SAVE_DIR, fname_prefix="FR"
#         )

# # FULL CTR
# # snr_outputs_mean_ctr_full = np.zeros((len(CONTROL_SATED), len(conditions_array)))
# for i, animal in enumerate(CONTROL_SATED):
#     analysis(
#         animal, sf=None, start=40, stop=80, array_conditions=conditions_array,
#         save_dir=SAVE_DIR, fname_prefix="CTR"
#     )


for i, animal in enumerate(FOOD_RESTRICTED_SATED):
    for sf in range(5):
        analysis(
            animal, sf=sf, start=40, stop=80, array_conditions=conditions_array,
            save_dir=SAVE_DIR, fname_prefix="FR"
        )

# FULL CTR
for i, animal in enumerate(CONTROL_SATED):
    for sf in range(5):
        analysis(
            animal, sf=sf, start=40, stop=80, array_conditions=conditions_array,
            save_dir=SAVE_DIR, fname_prefix="CTR"
        )