"""
E1 (multi-group hierarchy): learned vs true drift recovery for the 3-group
asymmetric (upper-triangular) influence system. Shows MG-MVNN recovers the
directional cross-group coupling, not just the dynamics (Reviewer #2(b)),
on OOD test initial conditions (Reviewer #1(4)).
"""
import numpy as np, sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from mvnn_lib import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/SIMPLE/CASE7_opinion_dynamics"
OUT  = os.path.join(os.path.dirname(__file__), "figs")
DT = 1e-3
CKPT = f"{CASE}/TRAINING3/model_save_path/jax_ckpt_000100.npz"
GROUPS = {1: "Group 1 (workers)", 2: "Group 2 (managers)", 3: "Group 3 (CEOs)"}
TIMES = [0, 100, 200, 350, 500]
NQ = 2000

params = load_params(CKPT)

def load_state(seed, t):
    x1 = np.load(f"{CASE}/TEST_CASE3/data/opinion_workers_mpi_{seed}.npy")[t][:, None]
    x2 = np.load(f"{CASE}/TEST_CASE3/data/opinion_managers_mpi_{seed}.npy")[t][:, None]
    x3 = np.load(f"{CASE}/TEST_CASE3/data/opinion_ceos_mpi_{seed}.npy")[t][:, None]
    return x1, x2, x3

seeds = sorted(int(os.path.basename(f).split('_')[-1].split('.')[0])
               for f in glob.glob(f"{CASE}/TEST_CASE3/data/opinion_workers_mpi_*.npy"))[:5]
print("test seeds:", seeds)
rng = np.random.default_rng(0)

scat = {k: {"t": [], "l": []} for k in GROUPS}
err_time = {k: [] for k in GROUPS}              # list over seeds of [per-time err]

for seed in seeds:
    per_seed = {k: [] for k in GROUPS}
    for t in TIMES:
        x1, x2, x3 = load_state(seed, t)
        mu1 = embedding_mean_mg(params, x1, 1)
        mu2 = embedding_mean_mg(params, x2, 2)
        mu3 = embedding_mean_mg(params, x3, 3)
        Xs = {1: x1, 2: x2, 3: x3}
        for k in GROUPS:
            X = Xs[k]
            q = rng.choice(X.shape[0], min(NQ, X.shape[0]), replace=False)
            Xq = X[q]
            bl = mvnn_drift_mg(params, Xq, mu1, mu2, mu3, k)
            bt = true_drift_mg_group(k, x1, x2, x3, Xq=Xq)
            per_seed[k].append(rel_l2(bl, bt))
            if seed == seeds[0]:
                scat[k]["t"].append(bt.ravel()); scat[k]["l"].append(bl.ravel())
    for k in GROUPS:
        err_time[k].append(per_seed[k])

# ---- drift profiles at t index 200 on seed 0 ----
x1, x2, x3 = load_state(seeds[0], 200)
mu1 = embedding_mean_mg(params, x1, 1); mu2 = embedding_mean_mg(params, x2, 2); mu3 = embedding_mean_mg(params, x3, 3)
prof = {}
for k, X in zip(GROUPS, [x1, x2, x3]):
    order = np.argsort(X.ravel()); xs = X[order]
    bl = mvnn_drift_mg(params, xs, mu1, mu2, mu3, k).ravel()
    bt = true_drift_mg_group(k, x1, x2, x3, Xq=xs).ravel()
    prof[k] = (xs.ravel(), bt, bl)

# ============================ figure ============================
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
colors = {1: "C0", 2: "C1", 3: "C2"}
r2s = {}
for k in GROUPS:
    t = np.concatenate(scat[k]["t"]); l = np.concatenate(scat[k]["l"])
    r2s[k] = r2_score(l, t)
    ax[0].scatter(t, l, s=3, alpha=0.25, color=colors[k],
                  label=f"{GROUPS[k]} ($R^2$={r2s[k]:.3f})")
lim = [-1.1, 1.1]
allt = np.concatenate([np.concatenate(scat[k]["t"]) for k in GROUPS])
lim = [allt.min(), allt.max()]
ax[0].plot(lim, lim, 'r--', lw=1.2)
ax[0].set_xlabel(r"true drift $b^\star_k$"); ax[0].set_ylabel(r"learned drift $b_{\theta,k}$")
ax[0].set_title("(a) Multi-group drift recovery (OOD)"); ax[0].legend(fontsize=9)

for k in GROUPS:
    xs, bt, bl = prof[k]
    ax[1].plot(xs, bt, '-', color=colors[k], lw=2, label=f"{GROUPS[k]} true")
    ax[1].plot(xs, bl, '--', color=colors[k], lw=1.8)
ax[1].set_xlabel("opinion $x$"); ax[1].set_ylabel("drift")
ax[1].set_title("(b) Per-group drift profiles (solid=true, dashed=learned)")
ax[1].legend(fontsize=9)

for k in GROUPS:
    arr = np.array(err_time[k])               # (seeds, times)
    m, s = arr.mean(0), arr.std(0)
    tt = np.array(TIMES)*DT
    ax[2].plot(tt, m, marker="o", color=colors[k], lw=2, label=GROUPS[k])
    ax[2].fill_between(tt, m-s, m+s, alpha=0.15, color=colors[k])
ax[2].set_xlabel("time $t$"); ax[2].set_ylabel(r"relative $L^2$ drift error")
ax[2].set_title("(c) Drift error over time"); ax[2].legend(fontsize=9); ax[2].set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(f"{OUT}/drift_recovery_mg.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/drift_recovery_mg.png", dpi=130, bbox_inches="tight")

metrics = {"case": "CASE7_multigroup_hierarchy",
           "R2_by_group": {GROUPS[k]: float(r2s[k]) for k in GROUPS},
           "rel_l2_by_group_time": {GROUPS[k]: {str(TIMES[i]*DT): float(np.array(err_time[k]).mean(0)[i]) for i in range(len(TIMES))} for k in GROUPS}}
with open(f"{OUT}/metrics_drift_mg.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2)); print("saved drift_recovery_mg.pdf")
