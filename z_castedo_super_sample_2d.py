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


FOOD_RESTRICTED_SATED = [1,2,3,6,7,9,11,12]
CONTROL_SATED = [0,4,5,8,10,13]

# AngStim_data = '../../Data/metadata_deconv/stimAngle_sated.mat'
AngStim_data = '../Data/metadata_deconv/stimAngle_sated.mat'

ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
SATED_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']
print(SATED_ANGLE[0].shape)

# SfStim_data = '../../Data/metadata_deconv/stimSpatFreq_sated.mat'
SfStim_data = '../Data/metadata_deconv/stimSpatFreq_sated.mat'

SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
SATED_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']
print(SATED_SF[0].shape)

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


def calculate_overlap(mu_hat, sigma_hat):
    num_angles = mu_hat.shape[0]
    num_evec = sigma_hat.shape[2]
    num_sf = sigma_hat.shape[1]
    overlaps = np.zeros((num_angles, num_sf, num_evec))
    eig_vals = np.zeros((num_angles,num_sf, num_evec))
    for i_sf in range(num_sf):
        for i_angles in range(num_angles):
            eval, evec = np.linalg.eigh(sigma_hat[i_angles,i_sf,:,:])
            evec = evec[:,::-1]  # Sort eigenvectors by eigenvalues in descending order
            eval = eval[::-1]  # Sort eigenvalues in descending order
            eig_vals[i_angles,i_sf,:] = eval
            d_mu = mu_hat[i_angles,i_sf,:] - mu_hat[(i_angles+1)%num_angles,i_sf,:]
            for j in range(num_evec):
                # square_overlap = np.abs(np.dot(d_mu, evec[:,j])**2 / (np.linalg.norm(d_mu)**2 * np.linalg.norm(evec[:,j])**2))
                # cosine = np.power((np.dot(d_mu, evec[:,j]))/(np.linalg.norm(d_mu) * np.linalg.norm(evec[:,j])),2)
                overlap = np.power((np.dot(d_mu, evec[:,j])),2)
                overlaps[i_angles,i_sf,j] = overlap
    return overlaps, eig_vals


def analysis(animal,k):

    TEST_DATA = resort_preprocessing(SATED_DECONV,SATED_ANGLE,SATED_SF,animal)[:,:,:,:,40:80]
    TEST_RESPONSE = jnp.nanmean(TEST_DATA,axis = -1) # Shape N x C1 x C2 x K 
    nan_mask = jnp.isnan(TEST_RESPONSE)  # shape (N, C1, C2, K)
    good_k = ~nan_mask.any(axis=(0, 1, 2))  # shape (K,)
    TEST_RESPONSE = TEST_RESPONSE[:, :, :, good_k]
    N = TEST_RESPONSE.shape[0]
    K = TEST_RESPONSE.shape[3]
    C1 = TEST_RESPONSE.shape[1]
    C2 = TEST_RESPONSE.shape[2]

    X_CONDITIONS = jnp.stack(jnp.meshgrid(jnp.arange(C1), jnp.arange(C2), indexing='ij'), axis=-1).reshape(-1, 2)
    TEST_RESPONSE_flat = TEST_RESPONSE.reshape(N, C1*C2, K)
    TEST_RESPONSE_transposed = jnp.transpose(TEST_RESPONSE_flat, (2, 1, 0)) # Now we need to transpose to get K x (C1*C2) x N
    Y_RESPONSE = TEST_RESPONSE_transposed
    SEED = 1
    PERIOD = C1 

    hyperparams = {
        'lambda_gp_angle': 1.8103336,
        'gamma_gp_angle': 4.5379544e-05,
        'beta_gp_angle': 0.12863505,

        'lambda_gp_sf': 0.26290625,
        'gamma_gp_sf': 0.00011522,
        'beta_gp_sf': 0.1991709,

        'lambda_wp_angle': 0.22797374,
        'gamma_wp_angle': 0.00010354,
        'beta_wp_angle': 1.6773869,

        'lambda_wp_sf': 0.36553314,
        'gamma_wp_sf': 2.8193786e-05,
        'beta_wp_sf': 0.27211133,

        'p': 1
    }
        
    # Initialise Kernel and Model
    periodic_kernel_gp_angle = lambda x, y: hyperparams['gamma_gp_angle']*(x==y) + hyperparams['beta_gp_angle']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['lambda_gp_angle']))
    square_kernel_gp_sf = lambda x, y: hyperparams['gamma_gp_sf']*(x==y) + hyperparams['beta_gp_sf']*jnp.exp(-(x-y)**2/(hyperparams['lambda_gp_sf']))
    kernel_gp = lambda x, y: periodic_kernel_gp_angle(x[0], y[0]) * square_kernel_gp_sf(x[1], y[1])
    # kernel_gp = lambda x, y: periodic_kernel_gp_angle(x[1], y[1]) * square_kernel_gp_sf(x[0], y[0])

    periodic_kernel_wp_angle = lambda x, y: hyperparams['gamma_wp_angle']*(x==y) + hyperparams['beta_wp_angle']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['lambda_wp_angle']))
    square_kernel_wp_sf = lambda x, y: hyperparams['gamma_wp_sf']*(x==y) + hyperparams['beta_wp_sf']*jnp.exp(-(x-y)**2/(hyperparams['lambda_wp_sf']))
    kernel_wp = lambda x, y: periodic_kernel_wp_angle(x[0], y[0]) * square_kernel_wp_sf(x[1], y[1])
    # kernel_wp = lambda x, y: periodic_kernel_wp_angle(x[1], y[1]) * square_kernel_wp_sf(x[0], y[0])

    # Prior distribution (GP and WP)
    gp = models.GaussianProcess(kernel=kernel_gp,N=N)
    # wp = models.WishartProcess(kernel = kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    wp = models.WishartLRDProcess(kernel=kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    likelihood = models.NormalConditionalLikelihood(N)

    joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    # Mean field variational family
    inference_seed = 2
    varfam = inference.VariationalNormal(joint.model)
    adam = optim.Adam(1e-1)
    key = jax.random.PRNGKey(inference_seed)

    varfam.infer(adam,X_CONDITIONS,Y_RESPONSE,n_iter = 5000,key=key)
    joint.update_params(varfam.posterior)

    # Posterior distribution
    posterior = models.NormalGaussianWishartPosterior(joint,varfam,X_CONDITIONS)

    # Sample from the posterior
    with numpyro.handlers.seed(rng_seed=inference_seed):
        mu_hat, sigma_hat, F_hat = posterior.sample(X_CONDITIONS)
    N = mu_hat.shape[1]
    mu_hat_reshaped = mu_hat.reshape(C1, C2, N)
    sigma_hat_reshaped = sigma_hat.reshape(C1, C2, N, N)
    overlaps_normal, eigs_normal = calculate_overlap(mu_hat_reshaped, sigma_hat_reshaped)

    with numpyro.handlers.seed(rng_seed=inference_seed):
        X_TEST_CONDITIONS = jnp.stack(jnp.meshgrid(jnp.linspace(0, C1-1, NEW_C1), jnp.linspace(0, C2-1, 5), indexing='ij'), axis=-1).reshape(-1, 2)
        mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(X_TEST_CONDITIONS)
    mu_test_hat_reshaped = mu_test_hat.reshape(NEW_C1, 5, N)
    sigma_test_hat_reshaped = sigma_test_hat.reshape(NEW_C1, 5, N, N)
    overlaps_super, eigs_super = calculate_overlap(mu_test_hat_reshaped, sigma_test_hat_reshaped)
    return overlaps_normal[:,:,:k], eigs_normal[:,:,:k], overlaps_super[:,:,:k], eigs_super[:,:,:k]

NEW_C1 = 120
k = 50
overlaps_fr_normal = np.zeros((8,12,5,k))
overlaps_fr_super = np.zeros((8,NEW_C1,5,k))
overlaps_ctr_normal = np.zeros((6,12,5,k))
overlaps_ctr_super = np.zeros((6,NEW_C1,5,k))

eigs_fr_normal = np.zeros((8,12,5,k))
eigs_fr_super = np.zeros((8,NEW_C1,5,k))
eigs_ctr_normal = np.zeros((6,12,5,k))
eigs_ctr_super = np.zeros((6,NEW_C1,5,k))


for i, FR in enumerate(FOOD_RESTRICTED_SATED):
    # print('ss')
    # try:
    overlaps_fr_normal[i,:,:,:], eigs_fr_normal[i,:,:,:], overlaps_fr_super[i,:,:,:], eigs_fr_super[i,:,:,:] = analysis(FR,k)
    # except Exception as e:
        # continue
for i, CTR in enumerate(CONTROL_SATED):
    # try:
    overlaps_ctr_normal[i,:,:,:], eigs_ctr_normal[i,:,:,:], overlaps_ctr_super[i,:,:,:], eigs_ctr_super[i,:,:,:] = analysis(CTR,k)
    # except Exception as e:
        # continue

# Save the results in one file
np.savez('../Data/overlaps_sated_food_restricted_super_2d_0.npz',
         overlaps_fr_normal=overlaps_fr_normal,
         overlaps_fr_super=overlaps_fr_super,
         eigs_fr_normal=eigs_fr_normal,
         eigs_fr_super=eigs_fr_super)
np.savez('../Data/overlaps_sated_control_super_2d_0.npz',
         overlaps_ctr_normal=overlaps_ctr_normal,
         overlaps_ctr_super=overlaps_ctr_super,
         eigs_ctr_normal=eigs_ctr_normal,
         eigs_ctr_super=eigs_ctr_super)

