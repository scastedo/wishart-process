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
def linear_response(cov_matrix):
    lam, U = np.linalg.eigh(cov_matrix)
    lam = np.clip(lam, 0, None)
    z = np.sqrt(lam)
    R0 = (U * z) @ U.T
    return R0, U, lam, z

def estimate_input(response_matrix, tuning_curves):
    # solve R0 x = dmu for x via least-squares (shared across angles)
    U, s, Vt = np.linalg.svd(response_matrix, full_matrices=False)
    s_inv = np.divide(1.0, s, out=np.zeros_like(s), where=s>1e-12)
    R0_pinv = (Vt.T * s_inv) @ U.T
    return R0_pinv @ tuning_curves
def analysis(animal, sf, start,stop, array_conditions):
    if sf == None:
        TEST_DATA = resort_preprocessing(SATED_DECONV,SATED_ANGLE,SATED_SF,animal)[:,:,:,:,start:stop]
        best_test = np.zeros((TEST_DATA.shape[0], TEST_DATA.shape[1], TEST_DATA.shape[3], TEST_DATA.shape[4]))
        for i in range(TEST_DATA.shape[0]):
            for j in range(TEST_DATA.shape[1]):
                best_sf = np.argmax(jnp.nanmean(TEST_DATA[i,j,:,:,:], axis = (1,2))).astype('int')
                best_test[i,j,:,:] = TEST_DATA[i,j,best_sf,:,:]
        TEST_RESPONSE = jnp.nanmean(best_test,axis = -1) # Shape N x C x K
    else:
        TEST_DATA = resort_preprocessing(SATED_DECONV,SATED_ANGLE,SATED_SF,animal)[:,:,sf,:,start:stop]
        TEST_RESPONSE = jnp.nanmean(TEST_DATA,axis = -1) # Shape N x C x K 

    nan_mask = jnp.isnan(TEST_RESPONSE)  # shape (N, C, K)
    good_k = ~nan_mask.any(axis=(0, 1))  # shape (K,)
    TEST_RESPONSE = TEST_RESPONSE[:, :, good_k]
    print(TEST_RESPONSE.shape)

    TEST_RESPONSE = jnp.transpose(TEST_RESPONSE, (2,1,0)) # Shape K X C X N
    N = TEST_RESPONSE.shape[2]
    C = TEST_RESPONSE.shape[1]
    SEED = 1
    PERIOD = C
    X_CONDITIONS = jnp.linspace(0,C-1,C)
    # good_trials = ~jnp.isnan(TEST_RESPONSE).all(axis=(1, 2))   # shape (K,)
    # TEST_RESPONSE = TEST_RESPONSE[good_trials]                 # (K′, C, N)

    hyperparams = {
        'sigma_m':2.4886496,
        'gamma_gp': 0.00016234,
        'beta_gp': 0.24385944,
        'sigma_c': 0.28382865,
        'gamma_wp':0.00044405,
        'beta_wp':1.0398238,
        'p': 0,
    }
        
    # Initialise Kernel and Model
    periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x==y) + hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['sigma_m']))
    periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x==y) + hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['sigma_c']))

    # Prior distribution (GP and WP)
    gp = models.GaussianProcess(kernel=periodic_kernel_gp,N=N)
    # wp = models.WishartProcess(kernel =periodic_kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    wp = models.WishartLRDProcess(kernel=periodic_kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    likelihood = models.NormalConditionalLikelihood(N)

    joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    # Mean field variational family
    inference_seed = 2
    varfam = inference.VariationalNormal(joint.model)
    adam = optim.Adam(1e-1)
    key = jax.random.PRNGKey(inference_seed)

    varfam.infer(adam,X_CONDITIONS,TEST_RESPONSE,n_iter = 1000,key=key)
    joint.update_params(varfam.posterior)

    SNR_OUTPUTS = np.zeros((len(array_conditions)))

    posterior = models.NormalGaussianWishartPosterior(joint,varfam,X_CONDITIONS)

    for i, condition_number in enumerate(array_conditions):
        with numpyro.handlers.seed(rng_seed=inference_seed):
            X_TEST_CONDITIONS = jnp.linspace(0, C-1, condition_number)
            mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(X_TEST_CONDITIONS)
        # Vectorized delta_mu computation
        next_mu = jnp.roll(mu_test_hat, -1, axis=0)
        prev_mu = jnp.roll(mu_test_hat, 1, axis=0)

        dmu_next = mu_test_hat - next_mu
        dmu_prev = mu_test_hat - prev_mu

        # Choose closest neighbor based on norm
        next_norms = jnp.linalg.norm(dmu_next, axis=1)
        prev_norms = jnp.linalg.norm(dmu_prev, axis=1)
        use_next = next_norms < prev_norms

        delta_mu = jnp.where(use_next[:, None], dmu_next, dmu_prev)

        # Compute norms as list
        norms = []
        for angle in range(12):
            R0, U, lam, z = linear_response(sigma_test_hat[angle,:,:])
            xhat = estimate_input(R0, delta_mu[angle,:])
            norms.append(jnp.linalg.norm(xhat)**2/xhat.shape[0])
        
        SNR_OUTPUTS[i] = np.nanmean(norms)
    return SNR_OUTPUTS

N_SF = 5
C_NORMAL = 12

conditions_array = np.linspace(12,140,10).astype('int')


# FULL FR
snr_outputs_mean_fr_full = np.zeros((len(FOOD_RESTRICTED_SATED), 5,len(conditions_array)))
for i, animal in enumerate(FOOD_RESTRICTED_SATED):
    for sf in range(5):
        snr_outputs_mean_fr_full[i, sf, :] = analysis(animal, sf= sf, start=40, stop=80, array_conditions=conditions_array)

# FULL CTR
snr_outputs_mean_ctr_full = np.zeros((len(CONTROL_SATED), 5, len(conditions_array)))
for i, animal in enumerate(CONTROL_SATED):
    for sf in range(5):
        snr_outputs_mean_ctr_full[i, sf, :] = analysis(animal, sf= sf, start=40, stop=80, array_conditions=conditions_array)

# # # # Early FR
# snr_outputs_mean_fr_early = np.zeros((len(FOOD_RESTRICTED_SATED), len(conditions_array)))
# for i, animal in enumerate(FOOD_RESTRICTED_SATED):
#     snr_outputs_mean_fr_early[i, :] = analysis(animal, sf= None, start=40, stop=60, array_conditions=conditions_array)

# # Early CTR
# snr_outputs_mean_ctr_early = np.zeros((len(CONTROL_SATED), len(conditions_array)))
# for i, animal in enumerate(CONTROL_SATED):
#     snr_outputs_mean_ctr_early[i, :] = analysis(animal, sf= None, start=40, stop=60, array_conditions=conditions_array)

# # Late FR
# snr_outputs_mean_fr_late = np.zeros((len(FOOD_RESTRICTED_SATED), len(conditions_array)))
# for i, animal in enumerate(FOOD_RESTRICTED_SATED):
#     snr_outputs_mean_fr_late[i, :] = analysis(animal, sf= None, start=60, stop=80, array_conditions=conditions_array)
# # Late CTR
# snr_outputs_mean_ctr_late = np.zeros((len(CONTROL_SATED), len(conditions_array)))
# for i, animal in enumerate(CONTROL_SATED):
#     snr_outputs_mean_ctr_late[i, :] = analysis(animal, sf= None, start=60, stop=80, array_conditions=conditions_array)  


# Combine all 6 arrays in a dictionary
snr_results = {
    'snr_outputs_mean_fr_full': snr_outputs_mean_fr_full,
    'snr_outputs_mean_ctr_full': snr_outputs_mean_ctr_full
    # 'snr_outputs_mean_fr_early': snr_outputs_mean_fr_early,
    # 'snr_outputs_mean_ctr_early': snr_outputs_mean_ctr_early,
    # 'snr_outputs_mean_fr_late': snr_outputs_mean_fr_late,
    # 'snr_outputs_mean_ctr_late': snr_outputs_mean_ctr_late
}
# Save the dictionary to a .npz file
np.savez('xhat_results_1.npz', **snr_results)


