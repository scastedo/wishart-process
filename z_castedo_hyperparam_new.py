from __future__ import annotations
import argparse
import itertools
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import optim
from tqdm import tqdm

import inference
import models
import utils
from scipy.io import loadmat
# -----------------------------------------------------------------------------
#  A.  JIT‑compiled evaluator factory
# -----------------------------------------------------------------------------

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
def make_evaluator(N, period, two_d, x_tr, y_tr, x_te, y_te,
                   n_vi_steps=1500, n_mc=8, df_min=0, V_scale=1e-2):

    data = (x_tr, y_tr, x_te, y_te)        # captured in the closure

    # @jax.jit                              # now only (key, hyper) are args
    def _evaluate(key, hyper):
        if two_d:
            # angle kernel (periodic)
            per_gp = lambda a, b: hyper["g_gp_a"] * (a == b) + hyper["b_gp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_gp_a"]
            )
            per_wp = lambda a, b: hyper["g_wp_a"] * (a == b) + hyper["b_wp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_wp_a"]
            )
            # SF kernel (squared‑exp)
            sq_gp = lambda a, b: hyper["g_gp_s"] * (a == b) + hyper["b_gp_s"] * jnp.exp(
                -(a - b) ** 2 / hyper["l_gp_s"]
            )
            sq_wp = lambda a, b: hyper["g_wp_s"] * (a == b) + hyper["b_wp_s"] * jnp.exp(
                -(a - b) ** 2 / hyper["l_wp_s"]
            )

            K_gp = lambda x, y: per_gp(x[0], y[0]) * sq_gp(x[1], y[1])
            K_wp = lambda x, y: per_wp(x[0], y[0]) * sq_wp(x[1], y[1])
        else:
            K_gp = lambda a, b: hyper["g_gp_a"] * (a == b) + hyper["b_gp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_gp_a"]
            )
            K_wp = lambda a, b: hyper["g_wp_a"] * (a == b) + hyper["b_wp_a"] * jnp.exp(
                -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_wp_a"]
            )

        # --- model -----------------------------------------------------------
        gp = models.GaussianProcess(kernel=K_gp, N=N)
        P  = max(int(hyper["p"]), df_min)                           # any value ≥ N is valid
        wp = models.WishartLRDProcess(
            kernel      = K_wp,
            P           = P,
            V           = V_scale * jnp.eye(N),
            optimize_L  = False,               # as before
        )
        lik = models.NormalConditionalLikelihood(N)
        joint = models.JointGaussianWishartProcess(gp, wp, lik)

        # --- VI --------------------------------------------------------------
        guide = inference.VariationalNormal(joint.model)
        optimizer = optim.Adam(1e-1)
        svi_key = jax.random.split(key, 1)[0]

        try:
            guide.infer(optimizer, x_tr, y_tr, n_vi_steps, key=svi_key)
            joint.update_params(guide.posterior)
            posterior = models.NormalGaussianWishartPosterior(joint, guide, x_tr)
        except ValueError as err:          # Cholesky / scale_tril failure
            return -jnp.inf                # or skip this hyperparam

        lik = models.NormalConditionalLikelihood(N)
        y_obs = y_te["x_test"]                   # or "x", depending on your goal
        def _one_draw(rng):  # This bit scores the posterior samples
            with numpyro.handlers.seed(rng_seed=rng):
                mu, Sigma, _ = posterior.sample(x_te)
                # average log-prob over all test observations
            return lik.log_prob(y_obs, mu, Sigma).mean()
        mc_keys = jax.random.split(key, n_mc)         # n_mc independent keys
        score   = jax.vmap(_one_draw)(mc_keys).mean()  # ← Monte-Carlo average
        return score
    return _evaluate


# -----------------------------------------------------------------------------
#  B.  Random search driver
# -----------------------------------------------------------------------------

def sample_hyperparams(key: jax.Array, two_d: bool) -> Dict[str, float]:
    """Draw a random hyper‑parameter point (log‑uniform priors)."""
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    def _logu(k, low, high):
        return jnp.exp(
            jax.random.uniform(
                k,
                (),                         # shape
                minval=jnp.log(low),        # ← name it
                maxval=jnp.log(high)        # ← name it
            )
        )
    hp = {
        "l_gp_a": _logu(k1, 0.2, 3.0),
        "g_gp_a": _logu(k2, 1e-5, 1e-3),
        "b_gp_a": _logu(k3, 0.05, 2.0),

        # WP for the covariance ---------------------------------------------
        "l_wp_a": _logu(k4, 0.2, 3.0),
        "g_wp_a": _logu(k5, 1e-5, 1e-3),
        "b_wp_a": _logu(k6, 0.05, 2.0),
        "p": int(jax.random.randint(k7, (), 0, 6)),   # 0‥5 inclusive

    }
    if two_d:
        k7, k8, k9, k10, k11, k12 = jax.random.split(k6, 6)
        hp.update(
            {
                "l_gp_s": _logu(k7, 0.1, 5.0),
                "g_gp_s": _logu(k8, 1e-6, 1e-3),
                "b_gp_s": _logu(k9, 0.05, 2.0),
                "l_wp_s": _logu(k10, 0.2, 5.0),
                "g_wp_s": _logu(k11, 1e-6, 1e-3),
                "b_wp_s": _logu(k12, 0.05, 4.0),
            }
        )
    return hp


def random_search(
    eval_fn,
    data_tuple,
    two_d: bool,
    n_draws: int = 200,
    seed: int = 0,
):
    best_hp, best_score = None, -jnp.inf
    scores, combos = [], []

    key = jax.random.PRNGKey(seed)
    for i in tqdm(range(n_draws), desc="random search"):
        key, sub = jax.random.split(key)
        hp = sample_hyperparams(sub, two_d)
        score = float(eval_fn(sub, hp))  # cast to Py float for tqdm
        scores.append(score)
        combos.append(hp)
        if score > best_score:
            best_score, best_hp = score, hp
            tqdm.write(f"▲ {i:3d}: {best_score:.3f}  {best_hp}")

    return best_hp, best_score, np.array(scores), combos


# -----------------------------------------------------------------------------
#  C.  Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", type=int, default=0)
    parser.add_argument("--two_d", type=int, default=1)  # 0/1
    parser.add_argument("--n_draws", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # --------------------------------------------------- load + preprocess data
    # HUNGRY_DECONV = np.load("../Data/predictions_fullTrace_hungry.npy", allow_pickle=True)
    # ANG = np.load("../Data/angles.npy", allow_pickle=True)
    # SF = np.load("../Data/spatfreq.npy", allow_pickle=True)


    HUNGRY_DECONV = np.load('../Data/predictions_fullTrace_hungry.npy', allow_pickle=True)
    FOOD_RESTRICTED_HUNGRY = [1,2,3,6,7,9,11,12]
    CONTROL_HUNGRY = [0,4,5,8,10,13]

    AngStim_data = '../Data/metadata_deconv/stimAngle_hungry.mat'
    ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells= True)
    ANG = ANG_STIM_DATA['order_of_stim_arossAnimals']

    SfStim_data = '../Data/metadata_deconv/stimSpatFreq_hungry.mat'
    SF_STIM_DATA = loadmat(SfStim_data, simplify_cells= True)
    SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']

    td = bool(args.two_d)

    if td:
        # N × C_a × C_s × K  →  K × (C_a*C_s) × N
        TEST_DATA = resort_preprocessing(HUNGRY_DECONV,ANG,SF,args.animal)[:,:,:,:,40:80]
        TEST_RESPONSE = jnp.nanmean(TEST_DATA, axis=-1)
        good_trials = ~jnp.isnan(TEST_RESPONSE).all(axis=(1, 2, 3))   # shape (K,)
        TEST_RESPONSE = TEST_RESPONSE[good_trials]                 # (K′, C, N)
        N, Ca, Cs, K = TEST_RESPONSE.shape[0], *TEST_RESPONSE.shape[1:3], TEST_RESPONSE.shape[3]
        x_full = jnp.stack(jnp.meshgrid(jnp.arange(Ca), jnp.arange(Cs), indexing="ij"), axis=-1).reshape(-1, 2)
        y_full = jnp.transpose(TEST_RESPONSE.reshape(N, Ca * Cs, K), (2, 1, 0))
        period = Ca
    else:
        TEST_DATA = resort_preprocessing(HUNGRY_DECONV,ANG,SF,args.animal)[:,:,1,:,40:80] # Select SF 1
        # N × C × K  →  K × C × N
        TEST_RESPONSE = jnp.nanmean(TEST_DATA, axis=-1)
        good_trials = ~jnp.isnan(TEST_RESPONSE).all(axis=(1, 2))   # shape (K,)
        TEST_RESPONSE = TEST_RESPONSE[good_trials]                 # (K′, C, N)
        K = TEST_RESPONSE.shape[0]
        y_full = jnp.transpose(TEST_RESPONSE, (2, 1, 0))
        K, C, N = y_full.shape
        x_full = jnp.arange(C)[:, None]
        period = C

    # ------------------------------------------------------- split train / test
    data = utils.split_data(
        x=x_full,
        y=y_full,
        train_trial_prop=0.8,
        train_condition_prop=0.8,
        seed=args.seed,
    )
    x_tr, y_tr, _, _, x_te, y_te, *_ = data
    if not td:
        x_tr, x_te = x_tr.reshape(-1), x_te.reshape(-1)

    data_tuple = (x_tr, y_tr, x_te, y_te)

    # ------------------------------------------------------- evaluator + search
    eval_fn = make_evaluator(N, period, td, x_tr, y_tr, x_te, y_te, n_vi_steps=args.steps)
    best_hp, best_ll, scores, combos = random_search(
        eval_fn, data_tuple, td, n_draws=args.n_draws, seed=args.seed
    )

    print("\n🏆 best log‑lik:", best_ll)
    print("   best hyper‑params:\n", best_hp)

    np.savez(
        "random_search_results.npz",
        best_hp=best_hp,
        best_ll=best_ll,
        scores=scores,
        combos=np.array(combos, dtype=object),
    )


if __name__ == "__main__":
    main()
