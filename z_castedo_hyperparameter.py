from jax import config
config.update("jax_enable_x64", True)
import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import sys
from numpyro import optim
import jax.numpy as jnp
import numpyro

import inference
import models
import visualizations
import evaluation
import utils

import itertools
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat

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

def evaluate_hyperparameters(hyperparams, x_train, y_train, x_test, y_test, n_iter=5000, two_d = True):
    if two_d:
        # Initialise Kernel and Model
        periodic_kernel_gp_angle = lambda x, y: hyperparams['gamma_gp_angle']*(x==y) + hyperparams['beta_gp_angle']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['lambda_gp_angle']))
        square_kernel_gp_sf = lambda x, y: hyperparams['gamma_gp_sf']*(x==y) + hyperparams['beta_gp_sf']*jnp.exp(-(x-y)**2/(hyperparams['lambda_gp_sf']))
        kernel_gp = lambda x, y: periodic_kernel_gp_angle(x[0], y[0]) * square_kernel_gp_sf(x[1], y[1])
        # kernel_gp = lambda x, y: periodic_kernel_gp_angle(x[1], y[1]) * square_kernel_gp_sf(x[0], y[0])

        periodic_kernel_wp_angle = lambda x, y: hyperparams['gamma_wp_angle']*(x==y) + hyperparams['beta_wp_angle']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['lambda_wp_angle']))
        square_kernel_wp_sf = lambda x, y: hyperparams['gamma_wp_sf']*(x==y) + hyperparams['beta_wp_sf']*jnp.exp(-(x-y)**2/(hyperparams['lambda_wp_sf']))
        kernel_wp = lambda x, y: periodic_kernel_wp_angle(x[0], y[0]) * square_kernel_wp_sf(x[1], y[1])
        # kernel_wp = lambda x, y: periodic_kernel_wp_angle(x[1], y[1]) * square_kernel_wp_sf(x[0], y[0])
    else:
        # Initialise Kernel and Model
        kernel_gp = lambda x, y: hyperparams['gamma_gp_angle']*(x==y) + hyperparams['beta_gp_angle']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['lambda_gp_angle']))
        kernel_wp = lambda x, y: hyperparams['gamma_wp_angle']*(x==y) + hyperparams['beta_wp_angle']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(hyperparams['lambda_wp_angle']))
    
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

    varfam.infer(adam,x_train,y_train,n_iter,key=key)
    joint.update_params(varfam.posterior)
    
    posterior = models.NormalGaussianWishartPosterior(joint,varfam,x_train)
    with numpyro.handlers.seed(rng_seed=inference_seed):
        mu_hat, sigma_hat, F_hat = posterior.sample(x_train)
    mu_empirical = y_train.mean(0)
    log_likelihood = likelihood.log_prob(y_test['x'], mu_empirical, sigma_hat).flatten()

    return log_likelihood.mean()

def optimize_hyperparameters(x_train, y_train, x_val, y_val, hyperparam_grid, n_iter=5000,two_d = True):
    
    # Generate all combinations of hyperparameters
    keys = list(hyperparam_grid.keys())
    values = list(hyperparam_grid.values())
    hyperparameter_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
        
    best_score = float('-inf')  # For log-likelihood, higher is better
    score_list = []
    hyperparameter_list = []
    best_hyperparams = None
    
    for i, hyperparams in enumerate(tqdm(hyperparameter_combinations)):
        try:
            print(f"\nEvaluating hyperparameters {i+1}/{len(hyperparameter_combinations)}:")
            hyperparameter_list.append(hyperparams)
            score = evaluate_hyperparameters(hyperparams, x_train, y_train, x_val, y_val, n_iter, two_d=two_d)
            score_list.append(score)
            print(f"Score: {score}")
            
            # Update best if needed
            if score > best_score:
                best_score = score
                best_hyperparams = hyperparams
                print(f"New best score: {best_score}")
                
        except Exception as e:
            print(f"Error with hyperparams {hyperparams}: {str(e)}")
            continue
    
    return best_hyperparams, best_score, score_list, hyperparameter_list



# Main execution
if __name__ == "__main__":
    # Split data
    HUNGRY_DECONV = np.load('../Data/predictions_fullTrace_hungry.npy', allow_pickle=True)
    FOOD_RESTRICTED_HUNGRY = [1,2,3,6,7,9,11,12]
    CONTROL_HUNGRY = [0,4,5,8,10,13]

    AngStim_data = '../Data/metadata_deconv/stimAngle_hungry.mat'
    ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
    HUNGRY_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']

    SfStim_data = '../Data/metadata_deconv/stimSpatFreq_hungry.mat'
    SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
    HUNGRY_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']
    ######################################################
    hyperparam_grid = {
        'lambda_gp_angle': [1.0,5,10,15,20],
        'gamma_gp_angle': [1e-5],
        'beta_gp_angle': [10.0],
        
        # 'lambda_gp_sf': [0.1, 0.5, 1.0,5,10],
        # 'gamma_gp_sf': [1e-5],
        # 'beta_gp_sf': [10.0],
        
        'lambda_wp_angle': [0.5,1.0,5,10],
        'gamma_wp_angle': [1e-6],
        'beta_wp_angle': [1.0],
        
        # 'lambda_wp_sf': [0.1, 0.5, 1.0,5,10],
        # 'gamma_wp_sf': [1e-6],
        # 'beta_wp_sf': [1.0],
        
        'p': [0]
    }
    SEED = 1
    td = False

    #########################################################

    if td:
        TEST_DATA = resort_preprocessing(HUNGRY_DECONV,HUNGRY_ANGLE,HUNGRY_SF,0)[:,:,:,:,40:80] # Animal = 0
        TEST_RESPONSE = jnp.nanmean(TEST_DATA,axis = -1) # Shape N x C1 x C2 x K 
        N = TEST_RESPONSE.shape[0]
        K = TEST_RESPONSE.shape[3]
        C1 = TEST_RESPONSE.shape[1]
        C2 = TEST_RESPONSE.shape[2]
        X_CONDITIONS = jnp.stack(jnp.meshgrid(jnp.arange(C1), jnp.arange(C2), indexing='ij'), axis=-1).reshape(-1, 2)
        TEST_RESPONSE_flat = TEST_RESPONSE.reshape(N, C1*C2, K)
        TEST_RESPONSE_transposed = jnp.transpose(TEST_RESPONSE_flat, (2, 1, 0)) # Now we need to transpose to get K x (C1*C2) x N
        Y_RESPONSE = TEST_RESPONSE_transposed
        PERIOD = C1
        data = utils.split_data(x=X_CONDITIONS, y=Y_RESPONSE,  
                           train_trial_prop=0.8, train_condition_prop=0.8, seed=SEED)
        x_train, y_train, _, _, x_test, y_test, _, _, _, _, _, _, _, _ = data

    else:
        TEST_DATA = resort_preprocessing(HUNGRY_DECONV,HUNGRY_ANGLE,HUNGRY_SF,0)[:,:,1,:,40:80]
        TEST_RESPONSE = jnp.nanmean(TEST_DATA,axis = -1) # Shape N x C x K 
        Y_RESPONSE = jnp.transpose(TEST_RESPONSE, (2,1,0)) # Shape K X C X N
        N = Y_RESPONSE.shape[2]
        C = Y_RESPONSE.shape[1]
        K = Y_RESPONSE.shape[0]
        PERIOD = C
        X_CONDITIONS = jnp.linspace(0,C-1,C)
        data = utils.split_data(x=X_CONDITIONS[:,None], y=Y_RESPONSE,  
                           train_trial_prop=0.8, train_condition_prop=0.8, seed=SEED)
        # print('x conditions train ', x_train.shape)     # x_train 80% of conditions: 9/12
        # print('y data train ', y_train.shape)     # ytrain 80% conditions and trials : 9/12, 8/11
        # print('x conditions val ', x_test.shape)     # x_test 20% of conditions: 3/12
        # print('y_val, (x) ', y_test['x'].shape)     # y_test['x'] 20% of trials and 80% conditions: 9/12, 3/11
        # print('y_val, (x_test) ', y_test['x_test'].shape)     # y_test['x_test'] 100% of trials and 20% conditions: 3/12, 11/11
        x_train, y_train, _, _, x_test, y_test, _, _, _, _, _, _, _, _ = data
        x_train = x_train.reshape(x_train.shape[0])
        x_test = x_test.reshape(x_test.shape[0])


    best_hyperparams, best_score, score_collection, hyperparam_collection = optimize_hyperparameters(
        x_train, y_train, x_test, y_test, 
        hyperparam_grid, 
        n_iter=1500,two_d=td)

    print("\n=== Optimization Results ===")
    print(f"Best hyperparameters:")
    for k, v in best_hyperparams.items():
        print(f"  {k}: {v}")
    print(f"Best score: {best_score}")
    np.savez('hyperparameter_optimization_results.npz', 
             hyperparam_grid=hyperparam_grid, 
             best_hyperparams=best_hyperparams, 
             best_score=best_score,
             score_collection=score_collection,
             hyperparam_collection=hyperparam_collection)


# Took 3h to do 625 iterations