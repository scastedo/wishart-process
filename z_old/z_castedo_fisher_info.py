import sys

import jax
# jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update('jax_enable_x64', True)  # Use float64

from numpyro import optim
sys.path.append('wishart-process')

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
# HUNGRY_DECONV = np.load('../Data/predictions_fullTrace_hungry.npy', allow_pickle=True)
# FOOD_RESTRICTED_HUNGRY = [1,2,3,6,7,9,11,12]
# CONTROL_HUNGRY = [0,4,5,8,10,13]

# AngStim_data = '../Data/metadata_deconv/stimAngle_hungry.mat'
# ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
# HUNGRY_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']

# SfStim_data = '../Data/metadata_deconv/stimSpatFreq_hungry.mat'
# SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
# HUNGRY_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']
# TEST_DATA = resort_preprocessing(HUNGRY_DECONV,HUNGRY_ANGLE,HUNGRY_SF,0)[...,40:80]


SATED_DECONV = np.load('../Data/predictions_fullTrace_sated.npy', allow_pickle=True)
FOOD_RESTRICTED_SATED = [1,2,3,6,7,9,11,12]
CONTROL_SATED = [0,4,5,8,10,13]
AngStim_data = '../Data/metadata_deconv/stimAngle_sated.mat'
ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
SATED_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']
SfStim_data = '../Data/metadata_deconv/stimSpatFreq_sated.mat'
SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
SATED_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']



def evaluate_fi(test_data):
    sf_means = np.nanmean(test_data,axis = (0,1,3,4))
    max_sf_index = np.nanargmax(sf_means)
    test_data = test_data[:,:,max_sf_index,:,:] # Shape N x C x K x T
    test_response = jnp.nanmean(test_data,axis = -1) # Shape N x C x K 
    test_response = jnp.transpose(test_response, (2,1,0)) # Shape K X C X N
    N = test_response.shape[2]
    C = test_response.shape[1]
    K = test_response.shape[0]
    PERIOD = C
    x_conditions = jnp.linspace(0,C-1,C)
    hyperparams = {
    'sigma_m': 1,
    'gamma_gp': 1e-5,
    'beta_gp': 10.0,
    'sigma_c': 0.5,
    'gamma_wp': 1e-6,
    'beta_wp': 1.0,
    'p': 0
    }
    # Initialise Kernel and Model
    periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x==y) + hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['sigma_m']))
    periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x==y) + hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['sigma_c']))

    # Prior distribution (GP and WP)
    gp = models.GaussianProcess(kernel=periodic_kernel_gp,N=N)
    wp = models.WishartLRDProcess(kernel =periodic_kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    likelihood = models.NormalConditionalLikelihood(N)

    joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    # Mean field variational family
    inference_seed = 2
    varfam = inference.VariationalNormal(joint.model)
    adam = optim.Adam(1e-1)
    key = jax.random.PRNGKey(inference_seed)

    varfam.infer(adam,x_conditions,test_response,n_iter = 3000,key=key)
    joint.update_params(varfam.posterior)
    posterior = models.NormalGaussianWishartPosterior(joint,varfam,x_conditions)
    X_NEW_CONDITIONS = jnp.linspace(0,C-1,60)
    mu_hat, sigma_hat, F_hat = posterior.sample(X_NEW_CONDITIONS)
    mu_prime, sigma_prime = posterior.derivative(X_NEW_CONDITIONS)
    # Compute Fisher Information
    fi = evaluation.fisher_information(X_NEW_CONDITIONS, mu_prime, sigma_hat, sigma_prime)
    fi = jnp.squeeze(fi)
    return fi

with numpyro.handlers.seed(rng_seed=1):
    fr_fisher_info = np.zeros((len(FOOD_RESTRICTED_SATED), 60))
    ctr_fisher_info = np.zeros((len(CONTROL_SATED), 60))
    for i,animal in enumerate(FOOD_RESTRICTED_SATED):
        print('animal:',animal)
        test_data = remove_neurons(SATED_DECONV,SATED_ANGLE,SATED_SF,animal)[...,40:80]
        good_trials = []
        for i_trial in range(test_data.shape[3]):
            if not np.any(np.isnan(test_data[:,:,:,i_trial,:])):
                good_trials.append(i_trial)
        test_data = test_data[:,:,:,good_trials,:]    
        fi_animal = evaluate_fi(test_data)
        fr_fisher_info[i,:] =fi_animal
    for j, animal in enumerate(CONTROL_SATED):
        print('animal:',animal)
        test_data = remove_neurons(SATED_DECONV,SATED_ANGLE,SATED_SF,animal)[...,40:80]
        good_trials = []
        for i_trial in range(test_data.shape[3]):
            if not np.any(np.isnan(test_data[:,:,:,i_trial,:])):
                good_trials.append(i_trial)
        test_data = test_data[:,:,:,good_trials,:]    
        fi_animal = evaluate_fi(test_data)
        ctr_fisher_info[j,:] = fi_animal
    # Save Fisher Information results, handling different sizes
    np.save('fr_fisher_info.npy', fr_fisher_info)
    np.save('ctr_fisher_info.npy', ctr_fisher_info)