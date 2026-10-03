"""
E1 (1D Motsch-Tadmor): learned-drift vs true-drift recovery.

Reviewer #2 (b): show the network learned the right DRIFT FUNCTION, not just that it
replicates the dynamics. We have analytic access to b*(x,mu), so we evaluate the
learned drift b_theta(x,mu) on the (unseen) test measures and compare directly.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from mvnn_lib import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/SIMPLE/CASE5"
OUT  = os.path.join(os.path.dirname(__file__), "figs")
DT, ELL = 1e-2, 0.5
TIMES = [0, 100, 200, 300, 400]                 # t = 0,1,2,3,4
NQ = 3000                                        # query subsample per measure

# canonical checkpoint = the one deployed by CASE5/SIM3 to produce the paper's figures
params = load_params(f"{CASE}/TRAINING2/model_save_path/jax_ckpt_000400.npz")
import glob
traj_files = sorted(glob.glob(f"{CASE}/TEST_DATA3/data/opinion_traj_*.npy"),
                    key=lambda s: int(s.split('_')[-1].split('.')[0]))
print(f"{len(traj_files)} test trajectories")

rng = np.random.default_rng(0)
# pooled scatter data and per-time error stats
scat_true, scat_learn, scat_time = [], [], []
err_by_time = {t: [] for t in TIMES}
# dense error vs time on one representative trajectory
dense_times = list(range(0, 401, 10))
dense_err = []

for tf in traj_files:
    traj = np.load(tf)                                  # (T+1, N)
    for t in TIMES:
        X = traj[t][:, None]
        mu_emb = embedding_mean(params, X)
        q = rng.choice(X.shape[0], size=min(NQ, X.shape[0]), replace=False)
        Xq = X[q]
        bl = mvnn_drift(params, Xq, mu_emb)
        bt = true_drift_motsch_tadmor(Xq, X, ell=ELL)
        err_by_time[t].append(rel_l2(bl, bt))
        if tf == traj_files[0]:
            scat_true.append(bt.ravel()); scat_learn.append(bl.ravel())
            scat_time.append(np.full(bt.shape[0], t*DT))

# dense error vs time, averaged over first 5 trajectories
for tf in traj_files[:5]:
    traj = np.load(tf)
    row = []
    for t in dense_times:
        X = traj[t][:, None]
        mu_emb = embedding_mean(params, X)
        q = rng.choice(X.shape[0], size=1500, replace=False)
        Xq = X[q]
        bl = mvnn_drift(params, Xq, mu_emb)
        bt = true_drift_motsch_tadmor(Xq, X, ell=ELL)
        row.append(rel_l2(bl, bt))
    dense_err.append(row)
dense_err = np.array(dense_err)

scat_true = np.concatenate(scat_true); scat_learn = np.concatenate(scat_learn)
scat_time = np.concatenate(scat_time)
overall_rel = rel_l2(scat_learn, scat_true)
overall_r2  = r2_score(scat_learn, scat_true)

# ---- representative drift-profile overlay at t=1 (index 100) on traj 0 ----
traj0 = np.load(traj_files[0]); t_prof = 100
Xp = traj0[t_prof][:, None]
mu_emb = embedding_mean(params, Xp)
order = np.argsort(Xp.ravel())
xs = Xp[order]
bl_p = mvnn_drift(params, xs, mu_emb).ravel()
bt_p = true_drift_motsch_tadmor(xs, Xp, ell=ELL).ravel()

# ============================ figure ============================
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# (a) scatter learned vs true
sc = ax[0].scatter(scat_true, scat_learn, c=scat_time, s=4, alpha=0.35, cmap="viridis")
lim = [min(scat_true.min(), scat_learn.min()), max(scat_true.max(), scat_learn.max())]
ax[0].plot(lim, lim, 'r--', lw=1.5, label="ideal")
ax[0].set_xlabel(r"true drift $b^\star(x,\mu)$")
ax[0].set_ylabel(r"learned drift $b_\theta(x,\mu)$")
ax[0].set_title(f"(a) Drift recovery  ($R^2$={overall_r2:.3f}, rel-$L^2$={overall_rel:.3f})")
cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("time $t$")
ax[0].legend(loc="upper left")

# (b) drift profile overlay
ax[1].plot(xs.ravel(), bt_p, 'k-', lw=2, label=r"true $b^\star(\cdot,\mu_t)$")
ax[1].plot(xs.ravel(), bl_p, 'C1--', lw=2, label=r"learned $b_\theta(\cdot,\mu_t)$")
ax[1].set_xlabel("state $x$"); ax[1].set_ylabel("drift")
ax[1].set_title(r"(b) Drift profile at $t=1$ (unseen IC)")
ax[1].legend()

# (c) error vs time
mean_e = dense_err.mean(0); std_e = dense_err.std(0)
tt = np.array(dense_times)*DT
ax[2].plot(tt, mean_e, 'C0-', lw=2)
ax[2].fill_between(tt, mean_e-std_e, mean_e+std_e, alpha=0.25)
ax[2].set_xlabel("time $t$"); ax[2].set_ylabel(r"relative $L^2$ drift error")
ax[2].set_title("(c) Drift error over time (mean$\\pm$std, 5 test ICs)")
ax[2].set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(f"{OUT}/drift_recovery_mt_1d.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/drift_recovery_mt_1d.png", dpi=130, bbox_inches="tight")
print("saved figs/drift_recovery_mt_1d.pdf")

# ============================ metrics ============================
metrics = {
    "case": "CASE5_1D_MotschTadmor",
    "overall_rel_l2": overall_rel, "overall_R2": overall_r2,
    "rel_l2_by_time": {str(t*DT): float(np.mean(err_by_time[t])) for t in TIMES},
    "rel_l2_by_time_std": {str(t*DT): float(np.std(err_by_time[t])) for t in TIMES},
    "n_test_traj": len(traj_files),
}
with open(f"{OUT}/metrics_drift_mt_1d.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
