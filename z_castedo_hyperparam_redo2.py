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



def make_evaluator(N, period, x_tr, y_tr, x_te, y_te,
                   n_vi_steps=50000, n_mc=50, V_scale=1e-1):

    # @jax.jit                              # now only (key, hyper) are args
    def _evaluate(key, hyper):
        K_gp = lambda a, b: hyper["g_gp_a"] * (a == b) + hyper["b_gp_a"] * jnp.exp(
            -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_gp_a"]
        )
        K_wp = lambda a, b: hyper["g_wp_a"] * (a == b) + hyper["b_wp_a"] * jnp.exp(
            -jnp.sin(jnp.pi * jnp.abs(a - b) / period) ** 2 / hyper["l_wp_a"]
        )

        # --- model -----------------------------------------------------------
        gp = models.GaussianProcess(kernel=K_gp, N=N)                   
        wp = models.WishartLRDProcess(
            kernel      = K_wp,
            P           =  int(hyper["p"]) ,
            V           = V_scale * jnp.eye(N),
            optimize_L  = False,               # as before
        )

        lik = models.NormalConditionalLikelihood(N)
        joint = models.JointGaussianWishartProcess(gp, wp, lik)

        # --- VI --------------------------------------------------------------
        guide = inference.VariationalNormal(joint.model)
        optimizer = optim.Adam(1e-2)
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
        k_svi, k_mc = jax.random.split(key)
        mc_keys = jax.random.split(k_mc, n_mc)
        # mc_keys = jax.random.split(key, n_mc)         # n_mc independent keys
        score   = jax.vmap(_one_draw)(mc_keys).mean()  # ← Monte-Carlo average

        return score
    return _evaluate


# -----------------------------------------------------------------------------
#  B.  Random search driver
# -----------------------------------------------------------------------------

def sample_hyperparams(key: jax.Array) -> Dict[str, float]:
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
        "l_wp_a": _logu(k4, 0.07, 3.0),
        "g_wp_a": _logu(k5, 1e-5, 1e-1),
        "b_wp_a": _logu(k6, 0.1, 5.0),
        "p": int(jax.random.randint(k7, (), 0, 6)),   # 0‥5 inclusive

    }
    return hp


def random_search(
    eval_fn,
    n_draws: int = 1000,
    seed: int = 0,
):
    best_hp, best_score = None, -jnp.inf
    scores, combos = [], []

    key = jax.random.PRNGKey(seed)
    for i in tqdm(range(n_draws), desc="random search"):
        key, sub = jax.random.split(key)
        hp = sample_hyperparams(sub)
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
import json, numbers
import os
def to_jsonable(x):
    # Already JSON-safe
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    # NumPy scalar -> Python scalar
    if isinstance(x, np.generic):
        return x.item()
    # JAX/NumPy arrays -> list or scalar
    if isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
        arr = np.asarray(x)
        return arr.item() if arr.shape == () else arr.tolist()
    # Other numeric types
    if isinstance(x, numbers.Number):
        return float(x)
    # Mappings
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    # Sequences / sets
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    # Fallback: string representation
    return str(x)

def save_best_hp(animal, best_hp, best_ll, scores, combos, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # JSON: easy to read later
    json_path = os.path.join(out_dir, f"animal_{animal:02d}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "animal": int(animal),
                "best_ll": to_jsonable(best_ll),
                "best_hp": to_jsonable(best_hp),
                "scores": to_jsonable(scores),
                "combos": to_jsonable(combos),
            },
            f,
            indent=2,
        )
    # Optional: raw arrays in NPZ (handy for debugging)
    npz_path = os.path.join(out_dir, f"animal_{animal:02d}.npz")
    np.savez(
        npz_path,
        best_ll=np.array(best_ll),
        # store best_hp as a 0-d object array (still fine in npz)
        best_hp=np.array(best_hp, dtype=object),
        scores=np.asarray(scores),
        combos=np.array(combos, dtype=object),
    )
    return json_path
# --- your analysis ------------------------------------------------------------

def analyse(animal, seed, n_draws, steps, out_dir="hp_runs/sated"):
    # ------------- load & prepare data
    # SATED_DECONV = np.load('../Data/predictions_fullTrace_sated.npy', allow_pickle=True)
    # AngStim_data = '../Data/metadata_deconv/stimAngle_sated.mat'
    # ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells=True)
    # SATED_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']
    # SfStim_data = '../Data/metadata_deconv/stimSpatFreq_sated.mat'
    # SF_STIM_DATA = loadmat(SfStim_data, simplify_cells=True)
    # SATED_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']

    # TEST_DATA = resort_preprocessing(SATED_DECONV, SATED_ANGLE, SATED_SF, animal)[:, :, :, :, 40:80]


    HUNGRY_DECONV = np.load('../Data/predictions_fullTrace_hungry.npy', allow_pickle=True)
    AngStim_data = '../Data/metadata_deconv/stimAngle_hungry.mat'
    ANG_STIM_DATA = loadmat(AngStim_data, simplify_cells=True)
    HUNGRY_ANGLE = ANG_STIM_DATA['order_of_stim_arossAnimals']
    SfStim_data = '../Data/metadata_deconv/stimSpatFreq_hungry.mat'
    SF_STIM_DATA = loadmat(SfStim_data, simplify_cells=True)
    HUNGRY_SF = SF_STIM_DATA['stimSpatFreq_arossAnimals']

    TEST_DATA = resort_preprocessing(HUNGRY_DECONV, HUNGRY_ANGLE, HUNGRY_SF, animal)[:, :, :, :, 40:80]


    best_test = np.zeros((TEST_DATA.shape[0], TEST_DATA.shape[1], TEST_DATA.shape[3], TEST_DATA.shape[4]))
    for i in range(TEST_DATA.shape[0]):
        best_sf = np.argmax(np.nanmean(TEST_DATA[i, :, :, :, :], axis=(0, 2, 3))).astype('int')
        best_test[i, :, :, :] = TEST_DATA[i, :, best_sf, :, :]

    TEST_RESPONSE = np.nanmean(best_test, axis=-1)  # Shape N x C x K
    good_trials = ~jnp.isnan(TEST_RESPONSE).any(axis=(0, 1))   # shape (K,)
    # Apply mask to keep only good trials
    TEST_RESPONSE = TEST_RESPONSE[:, :, good_trials]           # shape (N, C, K')
    # good_trials = ~np.isnan(TEST_RESPONSE).all(axis=(1, 2))   # shape (K,)
    # TEST_RESPONSE = TEST_RESPONSE[good_trials]                 # (K′, C, N)

    # K = TEST_RESPONSE.shape[0]
    y_full = jnp.transpose(TEST_RESPONSE, (2, 1, 0))  # (N, C, K′)
    K, C, N = y_full.shape
    x_full = jnp.arange(C)[:, None]
    period = C

    # -------- train/test split
    data = utils.split_data(
        x=x_full,
        y=y_full,
        train_trial_prop=0.8,
        train_condition_prop=0.8,
        seed=seed,
    )
    x_tr, y_tr, _, _, x_te, y_te, *_ = data
    x_tr, x_te = x_tr.reshape(-1), x_te.reshape(-1)

    # -------- evaluator + search
    # IMPORTANT: ensure make_evaluator uses y_obs = y_te (targets), not a misnamed key
    eval_fn = make_evaluator(N, period, x_tr, y_tr, x_te, y_te, n_vi_steps=steps)

    # per-animal deterministic seed (optional): combine base seed & animal id
    per_animal_seed = int(jax.random.key_data(jax.random.PRNGKey(seed))[0] ^ animal)

    best_hp, best_ll, scores, combos = random_search(
        eval_fn,
        n_draws=n_draws,
        seed=per_animal_seed,
    )

    print(f"\n[animal {animal}] 🏆 best log-lik: {best_ll:.3f}")
    print("best hyper-params:\n", best_hp)

    # save per-animal
    path = save_best_hp(animal, best_hp, best_ll, scores, combos, out_dir)
    print(f"[animal {animal}] saved → {path}")

    return best_hp, best_ll

# --- runner -------------------------------------------------------------------

if __name__ == "__main__":
    ALL_ANIMALS = list(range(14))
    ndraws = 500
    steps = 50000
    base_seed = 0
    out_dir = "hp_runs/hungry"  # change if you want a different directory

    for animal in tqdm(sorted(ALL_ANIMALS), desc="animals"):
        analyse(animal=animal, seed=base_seed, n_draws=ndraws, steps=steps, out_dir=out_dir)