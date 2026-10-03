"""
E2 (Reviewer #2(a)): quantitative prediction error of the LEARNED dynamics vs the
TRUE dynamics over time, replacing purely-qualitative density overlays.

Metrics (averaged over the unseen test ICs, with std bands):
  - Wasserstein-1 distance W1(rho_pred_t, rho_true_t)  [exact in 1D]
  - L2 density error between Gaussian-KDE densities (the paper's existing metric)
for the deterministic and stochastic 1D Motsch-Tadmor systems. SIM = learned
mean-field rollout deployed in the paper; TEST = reference particle simulation,
both started from the SAME unseen initial condition.
"""
import numpy as np, os, glob, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figs")
DT = 1e-2
ROOT = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/SIMPLE"
CASES = {
    "deterministic": dict(sim=f"{ROOT}/CASE5/SIM3/data/x_save_{{}}.npy",
                          tru=f"{ROOT}/CASE5/TEST_DATA3/data/opinion_traj_{{}}.npy"),
    "stochastic ($\\sigma$=0.1)": dict(sim=f"{ROOT}/CASE5_stochastic/SIM3/data/x_save_{{}}.npy",
                          tru=f"{ROOT}/CASE5_stochastic/TEST_DATA3/data/opinion_traj_{{}}.npy"),
}


def w1_1d(a, b):
    """exact 1D Wasserstein-1 between empirical measures (resampled to equal size)."""
    a = np.sort(np.ravel(a)); b = np.sort(np.ravel(b))
    n = min(a.size, b.size)
    qa = np.quantile(a, (np.arange(n)+0.5)/n)
    qb = np.quantile(b, (np.arange(n)+0.5)/n)
    return float(np.mean(np.abs(qa-qb)))


def kde(x, grid, bw=None):
    x = np.ravel(x); n = x.size
    if bw is None:
        bw = 1.06*np.std(x)*n**(-1/5)
    u = (grid[:, None]-x[None, :])/bw
    return (np.exp(-0.5*u*u)/np.sqrt(2*np.pi)).mean(1)/bw


def l2_dens(a, b, grid):
    pa, pb = kde(a, grid), kde(b, grid)
    return float(np.sqrt(np.trapz((pa-pb)**2, grid)))


TIMES = list(range(0, 401, 20))
results = {}
for name, paths in CASES.items():
    ids = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
                 for f in glob.glob(paths["sim"].format("*")))
    W = np.full((len(ids), len(TIMES)), np.nan)
    L = np.full((len(ids), len(TIMES)), np.nan)
    for r, sid in enumerate(ids):
        sim = np.load(paths["sim"].format(sid)).reshape(401, -1)
        tru = np.load(paths["tru"].format(sid)).reshape(401, -1)
        lo = min(sim.min(), tru.min())-0.5; hi = max(sim.max(), tru.max())+0.5
        grid = np.linspace(lo, hi, 400)
        for c, t in enumerate(TIMES):
            W[r, c] = w1_1d(sim[t], tru[t])
            L[r, c] = l2_dens(sim[t], tru[t], grid)
    results[name] = dict(W=W, L=L, ids=ids)
    print(f"{name}: {len(ids)} ICs, mean W1={np.nanmean(W):.4f}, mean L2dens={np.nanmean(L):.4f}")

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
tt = np.array(TIMES)*DT
for name, res in results.items():
    for j, (key, lab) in enumerate([("W", r"Wasserstein-1 error"), ("L", r"$L^2$ density error")]):
        arr = res[key]; m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
        ax[j].plot(tt, m, marker="o", lw=2, label=name)
        ax[j].fill_between(tt, m-s, m+s, alpha=0.18)
for j, lab in enumerate([r"$W_1(\rho^{\rm pred}_t,\rho^{\rm true}_t)$",
                         r"$\|\rho^{\rm pred}_t-\rho^{\rm true}_t\|_{L^2}$"]):
    ax[j].set_xlabel("time $t$"); ax[j].set_ylabel(lab); ax[j].legend(); ax[j].set_ylim(bottom=0)
ax[0].set_title("(a) Predicted-vs-true distribution error")
ax[1].set_title("(b) Predicted-vs-true density error")
plt.tight_layout()
fig.savefig(f"{OUT}/pred_error_1d.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/pred_error_1d.png", dpi=130, bbox_inches="tight")

metrics = {name: {"mean_W1": float(np.nanmean(res["W"])),
                  "final_W1": float(np.nanmean(res["W"][:, -1])),
                  "mean_L2dens": float(np.nanmean(res["L"])),
                  "n_ICs": len(res["ids"])} for name, res in results.items()}
with open(f"{OUT}/metrics_pred_error_1d.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2)); print("saved pred_error_1d.pdf")
