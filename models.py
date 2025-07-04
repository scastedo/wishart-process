# -*- coding: utf-8 -*-
"""
@author: Amin
"""

import numpy as onp
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from jax import vmap

# %%
class WishartProcess:
    def __init__(self, kernel, P, V, optimize_L=False, diag_scale=1e-1):
        self.kernel = kernel
        self.P = P
        self.N = V.shape[0]
        # Wishart mean is V/P
        self.L = jnp.linalg.cholesky(V/P)
        self.optimize_L = optimize_L
        self.diag_scale=diag_scale

    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)

    def f2sigma(self, F, L=None):
        if L is None: L = self.L
        diag = self.diag_scale*jnp.eye(self.N)[:,:,None]
        fft = jnp.einsum('abn,cbn->acn',F[:,:-1],F[:,:-1]) + diag
        afft = jnp.einsum('ab,bcn->acn',L,fft) 
        sigma = jnp.einsum('abn,bc->nac',afft,L.T) 
        return sigma

    def sample(self, x):
        C = x.shape[0]
        L = numpyro.param('L', self.L) if self.optimize_L else self.L

        c_f = self.evaluate_kernel(x,x)

        F = numpyro.sample(
            'F',dist.MultivariateNormal(jnp.zeros(C),covariance_matrix=c_f),
            sample_shape=(self.N,self.P)
        )
        self.F = F

        sigma = self.f2sigma(F,L)

        return sigma
    
    
    def posterior(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        f = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
        K = K_x_x - K_X_x.T@Ki@K_X_x
        
        K = K + 1e-4 * jnp.eye(K.shape[-1]) # regularize covariance ADDED BY SEB

        F = numpyro.sample(
            'F_test',dist.MultivariateNormal(f,covariance_matrix=K),
            sample_shape=(1,1)
        ).squeeze()

        sigma = self.f2sigma(F)

        return F, sigma
    
    def posterior_mode(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        F = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
        sigma = self.f2sigma(F)

        return F, sigma
    
    def posterior_derivative(self, X, Y, x_new):
        K_X_X  = self.evaluate_kernel(X,X)
        Ki   = jnp.linalg.inv(K_X_X)

        def sigma(x):
            K_X_x  = self.evaluate_kernel(x,X)
            F = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
            sigma = self.f2sigma(F)
            return sigma

        grad = vmap(lambda x: jax.jacrev(sigma)(x[None]))(x_new).squeeze().T

        return grad
    
    
    def log_prob(self, x, F):
        # TODO: input to this fn must be sigma, not F
        C = x.shape[0]
        c_f = self.evaluate_kernel(x,x)
        LPF = dist.MultivariateNormal(jnp.zeros(C),covariance_matrix=c_f).log_prob(F)
        return LPF

# %%
class WishartLRDProcess:
    def __init__(self, kernel, P, V, optimize_L=False, diag_scale=1e-1):
        self.kernel = kernel
        self.P = P
        self.N = V.shape[0]
        # Wishart mean is V/nu
        self.L = jnp.linalg.cholesky(V/max(P,1))
        self.optimize_L = optimize_L
        self.diag_scale=diag_scale

    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)

    def f2sigma(self, F, L=None):
        if L is None: L = self.L
        diag = self.diag_scale*jnp.stack([jnp.diag(jax.nn.softplus(F[:,-1,i])) for i in range(F.shape[2])],axis=-1)
        fft = jnp.einsum('abn,cbn->acn',F[:,:-1],F[:,:-1]) + diag
        afft = jnp.einsum('ab,bcn->acn',L,fft) 
        sigma = jnp.einsum('abn,bc->nac',afft,L.T) 
        
        return sigma

    def sample(self, x):
        C = x.shape[0]
        L = numpyro.param('L', self.L) if self.optimize_L else self.L

        c_f = self.evaluate_kernel(x,x)

        F = numpyro.sample(
            'F',dist.MultivariateNormal(jnp.zeros(C),covariance_matrix=c_f),
            sample_shape=(self.N,self.P+1)
        )
        self.F = F

        sigma = self.f2sigma(F,L)

        return sigma

    
    def posterior(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        f = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
        K = K_x_x - K_X_x.T@Ki@K_X_x
        # regularize covariance ADDED BY SEB
        jitter = 1e-6  # ADDED BY SEB
        K = K + jitter * jnp.eye(K.shape[0], dtype=K.dtype) # ADDED BY SEB
        
        F = numpyro.sample(
            'F_test',dist.MultivariateNormal(f,covariance_matrix=K),
            sample_shape=(1,1)
        )[0,0]

        sigma = self.f2sigma(F)

        return F, sigma
    
    def posterior_mode(self, X, Y, sigma, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y, sigma

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)

        Ki   = jnp.linalg.inv(K_X_X)
        
        F = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)

        sigma = self.f2sigma(F)

        return F, sigma
    
    def posterior_derivative(self, X, Y, x_new):
        K_X_X  = self.evaluate_kernel(X,X)
        Ki   = jnp.linalg.inv(K_X_X)

        def sigma(x):
            K_X_x  = self.evaluate_kernel(x,X)
            F = jnp.einsum('ij,mnj->mni',(K_X_x.T@Ki),Y)
            sigma = self.f2sigma(F)
            return sigma
        

        grad = vmap(lambda x: jax.jacrev(sigma)(x[None]))(x_new).squeeze().T

        return grad
    
    def log_prob(self, x, F):
        # TODO: input to this fn must be sigma, not F
        C = x.shape[0]
        c_f = self.evaluate_kernel(x,x)
        LPF = dist.MultivariateNormal(jnp.zeros(C),covariance_matrix=c_f).log_prob(F)
        return LPF
    
# %%
class GaussianProcess:
    def __init__(self, kernel, N):
        self.kernel = kernel
        self.N = N


    def evaluate_kernel(self, xs, ys):
        return vmap(lambda x: vmap(lambda y: self.kernel(x, y))(xs))(ys)


    def sample(self, x):
        C = x.shape[0]
        c_g = self.evaluate_kernel(x,x)
        G = numpyro.sample(
            'G',dist.MultivariateNormal(jnp.zeros(C),covariance_matrix=c_g),
            sample_shape=(self.N,1)
        ).squeeze().T
        return G
    
    def posterior(self, X, Y, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)
        
        Ki   = jnp.linalg.inv(K_X_X)
        f = jnp.einsum('ij,jm->mi',(K_X_x.T@Ki),Y)
        
        K = K_x_x - K_X_x.T@Ki@K_X_x
        # regularize covariance ADDED BY SEB
        jitter = 1e-6  # ADDED BY SEB
        K = K + jitter * jnp.eye(K.shape[0], dtype=K.dtype) # ADDED BY SEB

        G_new = numpyro.sample(
            'G_test',dist.MultivariateNormal(f,covariance_matrix=K),
            sample_shape=(1,1)
        ).squeeze().T
        
        return G_new
    
    def posterior_mode(self, X, Y, x):
        # TODO: If x is a subset of X, return that subset of Y
        if jnp.array_equal(X,x): return Y

        K_X_x  = self.evaluate_kernel(x,X)
        K_x_x  = self.evaluate_kernel(x,x)
        K_X_X  = self.evaluate_kernel(X,X)
        
        Ki   = jnp.linalg.inv(K_X_X)
        G = jnp.einsum('ij,jm->mi',(K_X_x.T@Ki),Y)
        
        return G
    
    def posterior_derivative(self, X, Y, x_new):
        K_X_X  = self.evaluate_kernel(X,X)
        Ki = jnp.linalg.inv(K_X_X)
        def f(x):
            K_X_x  = self.evaluate_kernel(x,X)
            f = jnp.einsum('ij,jm->mi',(K_X_x.T@Ki),Y)
            return f

        grad = vmap(lambda x: jax.jacrev(f)(x[None]))(x_new).squeeze().T
        return grad
    
    def log_prob(self, x, G):
        C = x.shape[0]
        c_g = self.evaluate_kernel(x,x)
        LPG = dist.MultivariateNormal(jnp.zeros(C),covariance_matrix=c_g).log_prob(G)
        return LPG

# %% 
class NeuralTuningProcess:
    def __init__(self, N, spread, amp):
        self.N = N
        self.spread = spread
        self.amp = amp

    def sample(self, x):
        # generate N random phases
        # generate cosine response curves with the given spread
        p = numpyro.sample('phase', dist.Uniform(),sample_shape=(self.N,))
        a = numpyro.sample('amp', dist.Uniform(low=.5,high=1.5),sample_shape=(self.N,))
        return self.amp*a*(1+jnp.cos(((jnp.pi*x[:,None]/360.)-(p[None])*jnp.pi)/self.spread))
        


# %%
class NormalConditionalLikelihood:
    def __init__(self,N):
        self.N = N

    def sample(self,mu,sigma,ind=None,y=None):
        Y = numpyro.sample(
            'y',dist.MultivariateNormal(mu[ind,...],sigma[ind,...]),
            obs=y
        )
        return Y
    
    def log_prob(self,Y,mu,sigma,ind=None):
        LPY = dist.MultivariateNormal(mu[ind,...],sigma[ind,...]).log_prob(Y)
        return LPY
    
# %%
class PoissonConditionalLikelihood:
    def __init__(self,N):
        self.gain_fn = lambda x: jax.nn.softplus(x)
        self.gain_inverse_fn = lambda x: jnp.log(jnp.exp(x)-1)

        self.N = N
        self.rate = jnp.ones(N)
    
    def initialize_rate(self,y):
        self.rate = self.gain_inverse_fn(y.mean(0).mean(0))

    def sample(self,mu,sigma,ind=None,y=None):
        # rate = self.rate
        sample_shape = () if y is None else (len(y),)
        
        rate = numpyro.param('rate', self.rate)

        G = numpyro.sample(
            'g',dist.MultivariateNormal(mu[ind,...],sigma[ind,...]),sample_shape=sample_shape
        )

        Y = numpyro.sample(
            'y',dist.Poisson(self.gain_fn(G+rate[None])).to_event(1),obs=y
        )
        return Y
    
    def log_prob(self,Y,mu,sigma,n_samples=10):
        sample_shape = (len(Y),)
        LPY = []
        
        for i in range(n_samples):
            G = numpyro.sample(
                'g',dist.MultivariateNormal(mu,sigma),sample_shape=sample_shape
            )
            
            LPY.append(dist.Poisson(self.gain_fn(G+self.rate[None])).to_event(1).log_prob(Y)
            )
        
        lp = jax.nn.logsumexp(jnp.stack(LPY),axis=0) - jnp.log(n_samples)
        
        return lp
    

# %%
class JointGaussianWishartProcess:
    def __init__(self, gp, wp, likelihood):
        self.gp = gp
        self.wp = wp
        self.likelihood = likelihood

    def model(self, x, y):
        B,N,D = y.shape

        sigma = self.wp.sample(x)
        G = self.gp.sample(x)

        with numpyro.plate('obs', N) as ind:
            self.likelihood.sample(G,sigma,ind,y=y)
        return
    
    def log_prob(self, G, F, sigma, x, y):
        B,N,D = y.shape

        LPW = self.wp.log_prob(x,F.transpose(1,2,0))
        LPG = self.gp.log_prob(x,G.T)

        with numpyro.plate('obs', N) as ind:
            LPL = self.likelihood.log_prob(y,G,sigma,ind)

        return LPW.sum() + LPG.sum() + LPL.sum()
    
    def update_params(self, posterior):
        params = [k for k in posterior.keys() if 'auto' not in k]
        for p in params:
            if hasattr(self.wp,p): exec('self.wp.'+p+'=posterior[\''+p+'\']')
            if hasattr(self.gp,p): exec('self.gp.'+p+'=posterior[\''+p+'\']')
            if hasattr(self.likelihood,p): exec('self.likelihood.'+p+'=posterior[\''+p+'\']')

    

# %%
class NormalPrecisionConditionalLikelihood:
    def sample(self,mu,sigma,ind=None,y=None):
        Y = numpyro.sample(
            'y',dist.MultivariateNormal(mu[ind,...],precision_matrix=sigma[ind,...]),
            obs=y
        )
        return Y
    
    def log_prob(self,Y,mu,sigma,ind=None):
        LPY = dist.MultivariateNormal(mu[ind,...],precision_matrix=sigma[ind,...]).log_prob(Y)
        return LPY


# %%
class NormalGaussianWishartPosterior:
    def __init__(self, joint, posterior, x):
        self.joint = joint
        self.posterior = posterior
        self.x = x

    def derivative(self,x):
        F,G = self.posterior.sample()

        mu_ = self.joint.gp.posterior_derivative(self.x, G.squeeze().T, x)
        sigma_ = self.joint.wp.posterior_derivative(self.x, F, x) 
        return mu_, sigma_ 

    def mean_stat(self,fun,x,vi_samples=1,y_samples=100):
        '''returns monte carlo estimate of a function expectation
        '''

        ys = []

        for _ in range(vi_samples):
            F,G = self.posterior.sample()
            sigma = self.joint.wp.f2sigma(F)
            mu_ = self.joint.gp.posterior(self.x, G.squeeze().T, x)
            _, sigma_ = self.joint.wp.posterior(self.x, F, sigma, x)
            
            for _ in range(y_samples):
                y = self.joint.likelihood.sample(mu_,sigma_)
                ys.append(y[0])
        
        return jnp.array([fun(y) for y in ys]).mean(0)

    def mode(self,x):
        F,G = self.posterior.mode()
        sigma = self.joint.wp.f2sigma(F)

        mu_ = self.joint.gp.posterior_mode(self.x, G.squeeze().T, x)
        F_, sigma_ = self.joint.wp.posterior_mode(self.x, F, sigma, x) 

        return mu_, sigma_, F_

    def sample(self, x):
        F,G = self.posterior.sample()
        
        sigma = self.joint.wp.f2sigma(F)

        mu_ = self.joint.gp.posterior(self.x, G.squeeze().T, x)
        F_, sigma_ = self.joint.wp.posterior(self.x, F, sigma, x) 
        return mu_, sigma_, F_
    
    def log_prob(self,x,y,vi_samples=10,gp_samples=1):
        # TODO: we need to exponentiate log_prob before summing!
        '''returns monte carlo estimate of log posterior predictive
        '''
        LPL = []
        for i in range(vi_samples):
            F,G = self.posterior.sample()
            sigma = self.joint.wp.f2sigma(F)
            for j in range(gp_samples):
                mu_ = self.joint.gp.posterior(self.x, G.squeeze().T, x)
                _, sigma_ = self.joint.wp.posterior(self.x, F, sigma, x) 
                lpl = self.joint.likelihood.log_prob(y,mu_,sigma_)
                LPL.append(lpl)
        
        LPP = jax.nn.logsumexp(jnp.stack(LPL),axis=0) - jnp.log(vi_samples) - jnp.log(gp_samples)
        return LPP
    

### 
# This is a workaround to avoid GPU Memory Errors. This allows for better performance and compatibility
# with JAX's automatic differentiation and JIT compilation.
# ADDED BY SEB
###
from functools import partial

def make_posterior_derivative(evaluate_kernel, f2sigma):
    @partial(jax.jit, static_argnums=(3,))
    def _compute_derivative(X, Y, x_new, batch_size):
        KXX = evaluate_kernel(X, X)
        Ki  = jnp.linalg.inv(KXX)

        def single_out(x):
            KXx = evaluate_kernel(x, X)
            F   = jnp.einsum('ij,mnj->mni', (KXx.T @ Ki), Y)
            return f2sigma(F)

        single_jac = jax.jit(jax.jacrev(single_out))

        def process(chunk):
            return jax.vmap(lambda xx: single_jac(xx[None]))(chunk)

        parts = jnp.array_split(x_new, max(1, x_new.shape[0] // batch_size))
        return jnp.concatenate([process(p) for p in parts], axis=0)

    return _compute_derivative

# ——— Now the squeezed patch ———
def _wp_derivative_squeezed(self, X, Y, x_new, batch_size=1):
    raw = make_posterior_derivative(self.evaluate_kernel, self.f2sigma)(
        X, Y, x_new, batch_size
    )
    # drop *all* singleton dims so (M,1,N,N,1) → (M,N,N)
    return raw.squeeze()

WishartProcess.posterior_derivative = _wp_derivative_squeezed
WishartLRDProcess.posterior_derivative = _wp_derivative_squeezed