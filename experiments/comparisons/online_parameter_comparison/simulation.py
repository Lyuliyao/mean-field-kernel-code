import numpy as np
import sys
import os

outdir = "data"
seed = int(sys.argv[1])
SIGMA = 0.1
DT = 1e-2
STEPS = 200
N = 16000

# --- load the learned kernel bandwidth
ESTIMATOR = 1   # 1 = averaged-over-agents estimator, 2 = single-agent estimator
ELL_HAT_PATH = f"ell_hat_est{ESTIMATOR}.npy"

ell_hat = float(np.load(ELL_HAT_PATH))
print(f"loaded learned kernel bandwidth: ell_hat = {ell_hat:.5f} (from {ELL_HAT_PATH})")


def init_opinion(N, seed=0):
    rng = np.random.default_rng(seed)
    num_gaussian = rng.integers(low=9, high=10, size=1)[0]
    value = rng.uniform(low=0, high=3, size=num_gaussian)
    p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
    x = 0.5 * rng.standard_normal(N) + rng.choice(value, size=N, p=p)
    x = x - np.mean(x, axis=0)
    return x.astype(np.float64), x.copy().astype(np.float64), rng


def learned_kernel_sum(x, ell, block=1000):
    """
    Normalized (Nadaraya-Watson) Gaussian-kernel mean-field drift with the LEARNED
    bandwidth `ell`:
        g_i = [sum_j w_ij (x_j - x_i)] / [sum_j w_ij],   w_ij = exp(-(x_i-x_j)^2/ell^2)
    Self-interaction (w_ii=1, diff=0) removed by subtracting 1 from the denominator,
    same as `influence_and_grad` in online_est_local_kernel.py. Blocked the same way
    as `wsindy_kernel_sum` above, for memory at N=16000.
    """
    N = x.shape[0]
    g = np.zeros(N, dtype=x.dtype)
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        xi = x[i0:i1][:, None]      # (Bi,1)
        num = np.zeros((i1 - i0,), dtype=x.dtype)
        den = np.zeros((i1 - i0,), dtype=x.dtype)
        for j0 in range(0, N, block):
            j1 = min(j0 + block, N)
            xj = x[j0:j1][None, :]  # (1,Bj)
            diff = xj - xi           # (Bi,Bj)
            w = np.exp(-(diff ** 2) / ell ** 2)
            num += np.sum(w * diff, axis=1)
            den += np.sum(w, axis=1)
        g[i0:i1] = num / (den - 1.0 + 1e-12)
    return g


x, _, _ = init_opinion(N, seed=seed)

x_save = np.empty((STEPS + 1, N), dtype=np.float64)
x_save[0] = x

rng = np.random.default_rng(seed)

for i in range(STEPS):
    f = learned_kernel_sum(x, ell_hat)
    if SIGMA > 0:
        # Euler-Maruyama
        x = x + DT * f + np.sqrt(DT) * SIGMA * rng.standard_normal(x.shape)
    else:
        x = x + DT * f

    x_save[i + 1] = x

os.makedirs(outdir, exist_ok=True)
np.save(f"{outdir}/x_save_param_est_{N}_{seed}.npy", x_save)
print(f"Saved shape {x_save.shape}")