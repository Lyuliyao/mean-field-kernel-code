# Implementation of online parameter estimation method from "Online parameter estimation for the McKean–Vlasov stochastic differential equation" by L. Sharrock, N. Kantas, P. Parpas, and G. A. Pavliotis (2023)

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def influence_and_grad(x_ref_idx, x_all, ell, block=2000):
    """
    For reference agent indices `x_ref_idx` (into `x_all`), returns
        g_i  = sum_{j!=i} w_ij (x_j - x_i) / sum_{j!=i} w_ij
        dg_i/d(ell)
    where w_ij = exp(-(x_i - x_j)^2 / ell^2). Self term (w_ii=1, diff=0)
    is removed by subtracting 1 from the denominator (dw_ii/dell = 0 already).
    """
    x_ref_idx = np.atleast_1d(x_ref_idx)
    x_ref = x_all[x_ref_idx]
    m = x_ref.shape[0]

    num = np.zeros(m); den = np.zeros(m)
    dnum = np.zeros(m); dden = np.zeros(m)

    N = x_all.shape[0]
    j0 = 0
    while j0 < N:
        j1 = min(j0 + block, N)
        xj = x_all[j0:j1][None, :]
        diff = xj - x_ref[:, None]                    # x_j - x_i
        w = np.exp(-(diff ** 2) / ell ** 2)
        dw = w * 2.0 * diff ** 2 / ell ** 3

        num += np.sum(w * diff, axis=1)
        den += np.sum(w, axis=1)
        dnum += np.sum(dw * diff, axis=1)
        dden += np.sum(dw, axis=1)
        j0 = j1

    den_ns = den - 1.0 + 1e-12
    g = num / den_ns
    dg = (dnum * den_ns - num * dden) / (den_ns ** 2)
    return g, dg


def online_est_ips_local_kernel(xtN, dt, ell0, ell_true, est_ell, gamma,
                                 seed=1, average=True, block=2000,
                                 sigma=0):
    """
    xtN      : (T+1, N) observed IPS trajectory
    dt       : Euler step used to generate xtN
    ell0     : initial parameter guess (used if est_ell)
    ell_true : true/fixed value (used if not est_ell, or for reference)
    est_ell  : whether to recursively estimate ell (else ell_t == ell_true throughout)
    gamma    : learning rate; scalar or length-(T+1) schedule
    average  : True  -> average the score*innovation update over M resampled
                        reference agents each step ("Estimator 1")
               False -> use a single fixed reference agent, index 0 ("Estimator 2")
    sigma    : None (default) -> noiseless-ODE velocity residual; numeric -> standard
               SDE-style residual/1/sigma^2 weighting

    Returns ell_t : (T+1,) recursive estimate.
    """
    rng = np.random.default_rng(seed)
    T = xtN.shape[0] - 1
    N = xtN.shape[1]

    if type(gamma) is float or type(gamma) is int:
        all_gamma = [gamma] * (T + 1)
    else:
        all_gamma = gamma

    ell_t = np.zeros(T + 1)
    if est_ell:
        ell_t[0] = ell0
    else:
        ell_t[:] = ell_true
        return ell_t

    for t in tqdm(range(T), leave=False):
        x = xtN[t]
        dx = xtN[t + 1] - x

        if average:
            idx = rng.choice(N, size=N, replace=False)
        else:
            idx = np.array([0])

        g, dg = influence_and_grad(idx, x, ell_t[t], block=block)

        if sigma==0:
            innovation = dx[idx] / dt - g
            update = np.mean(dg * innovation)
        else:
            innovation = dx[idx] - g * dt
            update = np.mean(dg * innovation) / (sigma ** 2)

        ell_t[t + 1] = max(ell_t[t] + all_gamma[t] * update, 1e-3)

    return ell_t


def find_trajectory_files(data_dir, pattern="opinion_traj_*.npy", min_seed=1, max_seed=100):
    files = glob.glob(os.path.join(data_dir, pattern))
    if not files:
        raise FileNotFoundError(f"no files matching '{pattern}' in {data_dir}")

    def seed_key(f):
        # m = re.search(r"(\d+)\.npy$", os.path.basename(f))
        m = re.search(r"(\d+)(?:_\w+)?\.npy$", os.path.basename(f)) # for threebody triplet_<id>_<measure>.npy
        return int(m.group(1)) if m else None

    filtered = [f for f in files
                if seed_key(f) is not None and min_seed <= seed_key(f) <= max_seed]

    if not filtered:
        raise FileNotFoundError(
            f"no files matching '{pattern}' with seed in "
            f"[{min_seed}, {max_seed}] in {data_dir}")

    return sorted(filtered, key=seed_key)


if __name__ == "__main__":

    # data_dir = "/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_neural_network/simple/case5/data_generation/data"
    data_dir = "/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_rebuttal/outputs/threebody/data/final"
    dt = 1e-2
    ell_true = 0.5
    ell0 = 1.5   # deliberately far from the true value, to show convergence
    SIGMA = 0

    # files = find_trajectory_files(data_dir, min_seed=1, max_seed=100)
    files = find_trajectory_files(
        data_dir, pattern="triplet_*_plus.npy", min_seed=0, max_seed=29
    )
    all_xtN = [np.load(f) for f in files]
    print(f"found {len(files)} files: {[os.path.basename(f) for f in files]}")

    T = all_xtN[0].shape[0] - 1
    print('Timesteps: ' + str(T))
    t_grid = [i * dt for i in range(T + 1)]
    gamma = [0.5 / (1 + i) ** 0.6 for i in range(T + 1)]

    n_seeds = len(all_xtN)
    all_ell_ips_t = np.zeros((T + 1, n_seeds, 2))   # [:, idx, 0]=Estimator 1 (avg), [:, idx, 1]=Estimator 2 (single)

    for idx, xtN in enumerate(all_xtN):
        print(f"seed {idx}: {files[idx]}  (T={xtN.shape[0]-1}, N={xtN.shape[1]})")

        ell_t_one = online_est_ips_local_kernel(
            xtN, dt, ell0=ell0, ell_true=ell_true, est_ell=True, gamma=gamma,
            seed=idx, average=True, sigma=SIGMA,
        )
        ell_t_two = online_est_ips_local_kernel(
            xtN, dt, ell0=ell0, ell_true=ell_true, est_ell=True, gamma=gamma,
            seed=idx, average=False, sigma=SIGMA
        )

        all_ell_ips_t[:, idx, 0] = ell_t_one
        all_ell_ips_t[:, idx, 1] = ell_t_two
        
    ell_hat_est1 = np.mean(all_ell_ips_t[-1, :, 0])   # Estimator 1 (averaged over agents)
    ell_hat_est2 = np.mean(all_ell_ips_t[-1, :, 1])   # Estimator 2 (single agent)
    print(f"learned ell_hat: Estimator 1 = {ell_hat_est1:.4f}, Estimator 2 = {ell_hat_est2:.4f}  "
          f"(untrained ell0 = {ell0}, true = {ell_true})")
    np.save("ell_hat_est1.npy", ell_hat_est1)
    np.save("ell_hat_est2.npy", ell_hat_est2)
    print("saved ell_hat_est1.npy and ell_hat_est2.npy -- pick either up in the forward-simulation script")
    
    plt.figure()
    plt.plot(t_grid, np.mean(all_ell_ips_t[:, :, 0], axis=1), label=r"$\widehat{\ell}_t$ (IPS Estimator 1, averaged)")
    plt.plot(t_grid, np.mean(all_ell_ips_t[:, :, 1], axis=1), label=r"$\widehat{\ell}_t$ (IPS Estimator 2, single agent)")
    plt.axhline(y=ell_true, linestyle="--", color="black", label=r"true $\ell$")
    plt.xlabel("t")
    plt.legend()
    plt.title(f"Online estimation of ell -- mean over {n_seeds} seeds")
    os.makedirs("result", exist_ok=True)
    plt.savefig("result/ell_est_mean_over_seeds.png", dpi=150)
    print(f"final mean estimates: avg={np.mean(all_ell_ips_t[-1,:,0]):.4f}  "
          f"single={np.mean(all_ell_ips_t[-1,:,1]):.4f}  (true={ell_true})")
    plt.show()