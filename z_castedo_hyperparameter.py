from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
import jax
from numpyro import optim

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



# def resort_preprocessing(datum,angle_arr,sf_arr,animal):
#     data = np.copy(datum[animal,:])
#     neurons = data[0].shape[0]
#     reshape_data = np.full((60,neurons,data[0].shape[1]), np.nan)
#     for i in range(60):
#         reshape_data[i,:,:] = data[i]

#     reshape_data = reshape_data.reshape(60,neurons,12,120)
#     reshape_data = np.transpose(reshape_data,(1,2,0,3))
#     #Remove first two neurons
#     reshape_data = reshape_data[2:,:,:,:]

#     #Remove None trials
#     max_trial = np.argmax(np.isnan(reshape_data[0,1,:,0]))
#     reshape_data = reshape_data[:,:,:max_trial,:]

#     # Remove beginning and last bit # HMMMM should I do this?
#     # reshape_data[:,0,:,:32] = np.nan
#     # reshape_data[:,-1,:,88:] = np.nan
#     # print(np.any(np.isnan(reshape_data)))
#     # print(reshape_data.shape)
    
#     # Reorder angles
#     angles = np.copy(angle_arr[animal])
#     for itrials in range(angles.shape[1]):
#         order = angles[:,itrials]-1
#         reshape_data[:,:,itrials,:] = reshape_data[:,order,itrials,:]

#     # Reorder SFs
#     reshaped_data = []
#     sfs = np.copy(sf_arr[animal])
#     for experiment in range(1,6):
#         mask = sfs == experiment
#         reshaped_data.append(reshape_data[:,:,mask,:])

#     max_trials = max([exp.shape[2] for exp in reshaped_data])
#     # Pad the data for experiments with fewer trials
#     for i in range(len(reshaped_data)):
#         if reshaped_data[i].shape[2] < max_trials:
#             padding = max_trials - reshaped_data[i].shape[2]
#             reshaped_data[i] = np.pad(reshaped_data[i], ((0, 0),(0, 0),(0, padding),(0, 0)), mode='constant', constant_values=np.nan)

#     reshaped_data = np.stack(reshaped_data,axis=2)    

#     return reshaped_data


# def evaluate_hyperparameters(hyperparams, x_train, y_train, x_test, y_test, n_iter=5000):

#     # Initialize kernels with the hyperparameters
#     periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x==y) + hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(2*hyperparams['sigma_m']**2))
#     periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x==y) + hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(2*hyperparams['sigma_c']**2))
    
#     # Set up the model
#     gp = models.GaussianProcess(kernel=periodic_kernel_gp, N=N)
#     wp = models.WishartProcess(kernel=periodic_kernel_wp, P=hyperparams['p'], V=1e-2*jnp.eye(N), optimize_L=False)
#     likelihood = models.NormalConditionalLikelihood(N)
#     joint = models.JointGaussianWishartProcess(gp, wp, likelihood)
    
#     # Set up variational inference
#     inference_seed = 2
#     key = jax.random.PRNGKey(inference_seed)
#     varfam = inference.VariationalNormal(joint.model)
#     adam = optim.Adam(1e-1)
    
#     # Train the model
#     varfam.infer(adam, x_train, y_train, n_iter=n_iter, key=key)
#     joint.update_params(varfam.posterior)
    
#     posterior = models.NormalGaussianWishartPosterior(joint,varfam,x_train)
#     with numpyro.handlers.seed(rng_seed=inference_seed):
#         mu_hat, sigma_hat, F_hat = posterior.sample(x_train)
#     mu_empirical = y_train.mean(0)
#     log_likelihood = likelihood.log_prob(y_test['x'], mu_empirical, sigma_hat).flatten()

#     # # Create posterior and compute log likelihood on validation data
#     # posterior = models.NormalGaussianWishartPosterior(joint, varfam, x_train)
#     # Compute log likelihood (directly using the posterior's log_prob method)
#     # This uses x_val and y_val_data
#     # log_likelihood = posterior.log_prob(x_test, y_test['x'])

#     return log_likelihood.mean()

# def optimize_hyperparameters(x_train, y_train, x_val, y_val, hyperparam_grid, n_iter=5000):
    
#     # Generate all combinations of hyperparameters
#     keys = list(hyperparam_grid.keys())
#     values = list(hyperparam_grid.values())
#     hyperparameter_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
        
#     best_score = float('-inf')  # For log-likelihood, higher is better
#     best_hyperparams = None
    
#     for i, hyperparams in enumerate(tqdm(hyperparameter_combinations)):
#         try:
#             print(f"\nEvaluating hyperparameters {i+1}/{len(hyperparameter_combinations)}:")
#             score = evaluate_hyperparameters(hyperparams, x_train, y_train, x_val, y_val, n_iter)
            
#             print(f"Score: {score}")
            
#             # Update best if needed
#             if score > best_score:
#                 best_score = score
#                 best_hyperparams = hyperparams
#                 print(f"New best score: {best_score}")
                
#         except Exception as e:
#             print(f"Error with hyperparams {hyperparams}: {str(e)}")
#             continue
    
#     return best_hyperparams, best_score

# Main execution
if __name__ == "__main__":


 
    N, P = 10, 11
    V = (1e-2 * jnp.eye(N) / P).astype(jnp.float64)

    print("Devices:", jax.devices())
    print("V.dtype:", V.dtype)
    print("Eigenvalues:", jnp.linalg.eigvals(V))

    # optional tiny jitter to be extra-safe
    V = V + 1e-8 * jnp.eye(N, dtype=V.dtype)

    L = jnp.linalg.cholesky(V)
    print("Cholesky decomposition succeeded.")
    # # Split data
    # HUNGRY_DECONV = np.load('../Data/predictions_fullTrace_hungry.npy', allow_pickle=True)
    # FOOD_RESTRICTED_HUNGRY = [1,2,3,6,7,9,11,12]
    # CONTROL_HUNGRY = [0,4,5,8,10,13]

    # AngStim_data = '../Data/metadata_deconv/stimAngle_hungry.mat'
    # ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
    # HUNGRY_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']

    # SfStim_data = '../Data/metadata_deconv/stimSpatFreq_hungry.mat'
    # SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
    # HUNGRY_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']


    # TEST_DATA = resort_preprocessing(HUNGRY_DECONV,HUNGRY_ANGLE,HUNGRY_SF,0)[:,:,1,:,40:80]
    # TEST_RESPONSE = jnp.nanmean(TEST_DATA,axis = -1) # Shape N x C x K 
    # TEST_RESPONSE = jnp.transpose(TEST_RESPONSE, (2,1,0)) # Shape K X C X N

    # N = TEST_RESPONSE.shape[2]
    # C = TEST_RESPONSE.shape[1]
    # K = TEST_RESPONSE.shape[0]
    # SEED = 1
    # PERIOD = C
    # X_CONDITIONS = jnp.linspace(0,C-1,C)


    # hyperparams = {
    #     'sigma_m': 0.5,
    #     'gamma_gp': 1e-5,
    #     'beta_gp': 10.0,
    #     'sigma_c': 0.5,
    #     'gamma_wp': 1e-6,
    #     'beta_wp': 1.0,
    #     'p': N+1
    #     }

    
    # periodic_kernel_gp = lambda x, y: hyperparams['gamma_gp']*(x==y) + hyperparams['beta_gp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(2*hyperparams['sigma_m']**2))
    # periodic_kernel_wp = lambda x, y: hyperparams['gamma_wp']*(x==y) + hyperparams['beta_wp']*jnp.exp(-jnp.sin(jnp.pi*jnp.abs(x-y)/PERIOD)**2/(2*hyperparams['sigma_c']**2))

    # # Prior distribution (GP and WP)
    # gp = models.GaussianProcess(kernel=periodic_kernel_gp,N=N)
    # wp = models.WishartProcess(kernel =periodic_kernel_wp,P=hyperparams['p'],V=1e-2*jnp.eye(N), optimize_L=False)
    # likelihood = models.NormalConditionalLikelihood(N)

    # # Given
    # # -----
    # # x : ndarray, (num_conditions x num_variables), stimulus conditions.
    # # y : ndarray, (num_trials x num_conditions x num_neurons), neural firing rates across C conditions repeated for K trials.

    # # Infer a posterior over neural means and covariances per condition.

    # # Joint distribution
    # joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

    # # Mean field variational family
    # inference_seed = 2
    # varfam = inference.VariationalNormal(joint.model)
    # adam = optim.Adam(1e-1)
    # key = jax.random.PRNGKey(inference_seed)

    # varfam.infer(adam,X_CONDITIONS,TEST_RESPONSE,n_iter = 100,key=key)
    # joint.update_params(varfam.posterior)
    # posterior = models.NormalGaussianWishartPosterior(joint,varfam,X_CONDITIONS)








    
    # data = utils.split_data(x=X_CONDITIONS[:, None], y=TEST_RESPONSE, 
    #                        train_trial_prop=0.8, train_condition_prop=0.8, seed=SEED)
    # x_train, y_train, _, _, x_test, y_test, _, _, _, _, _, _, _, _ = data
    
    # print('x conditions train ', x_train.shape)     # x_train 80% of conditions: 9/12

    # print('y data train ', y_train.shape)     # ytrain 80% conditions and trials : 9/12, 8/11

    # print('x conditions val ', x_test.shape)     # x_test 20% of conditions: 3/12

    # print('y_val, (x) ', y_test['x'].shape)     # y_test['x'] 20% of trials and 80% conditions: 9/12, 3/11

    # print('y_val, (x_test) ', y_test['x_test'].shape)     # y_test['x_test'] 100% of trials and 20% conditions: 3/12, 11/11
    # x_train = x_train.reshape(x_train.shape[0])
    # x_test = x_test.reshape(x_test.shape[0])

    # # Define a more focused hyperparameter grid
    # # hyperparam_grid = {
    # #     'sigma_m': [0.1,0.2,0.3,0.4, 0.5,0.6,0.7, 1.0,5],
    # #     'gamma_gp': [1e-6,1e-5,1e-4,1e-3],
    # #     'beta_gp': [1e-1,1,1e1,1e2],
    # #     'sigma_c': [0.1,0.2,0.3,0.4, 0.5,0.6,0.7, 1.0,5],
    # #     'gamma_wp': [1e-6,1e-5,1e-4,1e-3],
    # #     'beta_wp': [1e-1,1,1e1,1e2],
    # #     'p': [N+1]
    # # }

    # hyperparam_grid = {
    # 'sigma_m': [0.5,1,2],
    # 'gamma_gp': [1e-6],
    # 'beta_gp': [1e-1],
    # 'sigma_c': [0.5,1],
    # 'gamma_wp': [1e-6],
    # 'beta_wp': [1e-1,1],
    # 'p': [N+1]
    # }

    # # Run optimization with a time budget of 12 hours
    # best_hyperparams, best_score = optimize_hyperparameters(
    #     x_train, y_train, x_test, y_test, 
    #     hyperparam_grid, 
    #     n_iter=1000)

    # print("\n=== Optimization Results ===")
    # print(f"Best hyperparameters:")
    # for k, v in best_hyperparams.items():
    #     print(f"  {k}: {v}")
    # print(f"Best score: {best_score}")

    # # Save hyperparameter grid, the scores,  and best hyperparameters
    # np.savez('hyperparameter_optimization_results.npz', 
    #             hyperparam_grid=hyperparam_grid, 
    #             best_hyperparams=best_hyperparams, 
    #             best_score=best_score)
        
