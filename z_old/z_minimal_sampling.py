import sys
import jax
from numpyro import optim
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update('jax_enable_x64', True)  # Use float64


sys.path.append('wishart-process')
import inference
import models
import visualizations
import evaluation

import jax.numpy as jnp
import numpyro

N = 2
K = 20
C = 20
seed = 1
sigma_m = 10.
sigma_c = 10.

rng = [-10,10] # range of input conditions
x = jnp.linspace(rng[0], rng[1], C)

kernel_gp = lambda x, y: 1e1*(1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_m**2)))
kernel_wp = lambda x, y: 1e-6*(x==y)+jnp.exp(-jnp.linalg.norm(x-y)**2/(2*sigma_c**2))

gp = models.GaussianProcess(kernel=kernel_gp,N=N)
wp = models.WishartProcess(kernel=kernel_wp,P=N+1,V=1e-2*jnp.eye(N), optimize_L=False)

likelihood = models.NormalConditionalLikelihood(N)

with numpyro.handlers.seed(rng_seed=seed):
    mu = gp.sample(x)
    sigma = wp.sample(x)
    data = [likelihood.sample(mu,sigma,ind=jnp.arange(len(mu))) for i in range(K)]
    y = jnp.stack(data)

joint = models.JointGaussianWishartProcess(gp,wp,likelihood) 

inference_seed = 2
varfam = inference.VariationalNormal(joint.model)
adam = optim.Adam(1e-1)
key = jax.random.PRNGKey(inference_seed)
varfam.infer(adam,x,y,n_iter = 2000,key=key)
joint.update_params(varfam.posterior)

# Posterior distribution
x_test = jnp.linspace(rng[0], rng[1], 25)

posterior = models.NormalGaussianWishartPosterior(joint,varfam,x)
 
with numpyro.handlers.seed(rng_seed=seed):
    mu_test_hat, sigma_test_hat, F_test_hat = posterior.sample(x_test)


print(mu_test_hat.shape)
print(jnp.isnan(mu_test_hat).any())
