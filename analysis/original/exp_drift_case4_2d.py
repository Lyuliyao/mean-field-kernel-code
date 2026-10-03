"""
E1 (2D attraction-repulsion aggregation): learned-drift vs true-drift recovery,
including out-of-distribution test measures (disk, density-step) unseen in training
(training uses a single ring). Reviewer #2(b) + Reviewer #1(4) (OOD).
"""
import numpy as np, sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from mvnn_lib import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/SIMPLE/CASE4"
OUT  = os.path.join(os.path.dirname(__file__), "figs")
DT = 1e-2
KER = dict(c_rep=1.0, ell_rep=0.5, c_att=0.7, ell_att=2.0)
TESTSETS = {"ring (train-like)": "TEST_DATA", "disk (OOD)": "TEST_DATA3",
            "density-step (OOD)": "TEST_DATA4"}
TIMES = [0, 50, 100, 150, 199]

# canonical checkpoint = the one deployed by CASE4/SIM* to produce the paper's figures
params = load_params(f"{CASE}/TRAINING/model_save_path/jax_ckpt_001000.npz")
rng = np.random.default_rng(0)

# ---- aggregate scatter (disk OOD) + per-set error-vs-time ----
err_curves = {}
scat_t, scat_l = [], []
for name, d in TESTSETS.items():
    files = sorted(glob.glob(f"{CASE}/{d}/data/simulation_*.npy"))[:3]
    curve = []
    for t in TIMES:
        es = []
        for fpath in files:
            traj = np.load(fpath)
            X = traj[t]; mu = embedding_mean(params, X)
            q = rng.choice(X.shape[0], 2500, replace=False); Xq = X[q]
            bl = mvnn_drift(params, Xq, mu)
            bt = true_drift_attraction_repulsion(Xq, X, **KER)
            es.append(rel_l2(bl, bt))
            if name == "disk (OOD)" and fpath == files[0]:
                scat_t.append(bt.ravel()); scat_l.append(bl.ravel())
        curve.append((np.mean(es), np.std(es)))
    err_curves[name] = np.array(curve)
scat_t = np.concatenate(scat_t); scat_l = np.concatenate(scat_l)
ov_rel, ov_r2 = rel_l2(scat_l, scat_t), r2_score(scat_l, scat_t)

# ---- quiver field on disk measure at t=0 (OOD) ----
traj = np.load(sorted(glob.glob(f"{CASE}/TEST_DATA3/data/simulation_*.npy"))[0])
Xm = traj[0]
mu = embedding_mean(params, Xm)
gx = np.linspace(Xm[:,0].min(), Xm[:,0].max(), 17)
gy = np.linspace(Xm[:,1].min(), Xm[:,1].max(), 17)
GX, GY = np.meshgrid(gx, gy)
G = np.stack([GX.ravel(), GY.ravel()], 1)
Bt = true_drift_attraction_repulsion(G, Xm, **KER)
Bl = mvnn_drift(params, G, mu)

# ============================ figure ============================
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

ax[0].scatter(scat_t, scat_l, s=3, alpha=0.25, color="C0")
lim = [min(scat_t.min(), scat_l.min()), max(scat_t.max(), scat_l.max())]
ax[0].plot(lim, lim, 'r--', lw=1.5)
ax[0].set_xlabel(r"true drift $b^\star$ (per component)")
ax[0].set_ylabel(r"learned drift $b_\theta$")
ax[0].set_title(f"(a) 2D drift recovery, disk OOD\n$R^2$={ov_r2:.3f}, rel-$L^2$={ov_rel:.3f}")

ax[1].scatter(Xm[::40,0], Xm[::40,1], s=2, color="0.8", alpha=0.5)
ax[1].quiver(GX, GY, Bt[:,0].reshape(GX.shape), Bt[:,1].reshape(GX.shape),
             color="k", angles="xy", scale_units="xy", scale=4, width=0.004, label="true")
ax[1].quiver(GX, GY, Bl[:,0].reshape(GX.shape), Bl[:,1].reshape(GX.shape),
             color="C1", angles="xy", scale_units="xy", scale=4, width=0.0028, alpha=0.85, label="learned")
ax[1].set_title("(b) Drift field on unseen disk ($t=0$)")
ax[1].set_xlabel("$x_1$"); ax[1].set_ylabel("$x_2$"); ax[1].legend(loc="upper right")
ax[1].set_aspect("equal")

for name, c in err_curves.items():
    tt = np.array(TIMES)*DT
    ax[2].plot(tt, c[:,0], marker="o", lw=2, label=name)
    ax[2].fill_between(tt, c[:,0]-c[:,1], c[:,0]+c[:,1], alpha=0.18)
ax[2].set_xlabel("time $t$"); ax[2].set_ylabel(r"relative $L^2$ drift error")
ax[2].set_title("(c) Drift error over time"); ax[2].legend(); ax[2].set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(f"{OUT}/drift_recovery_ar_2d.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/drift_recovery_ar_2d.png", dpi=130, bbox_inches="tight")

metrics = {"case": "CASE4_2D_AttractionRepulsion",
           "disk_OOD_overall_rel_l2": ov_rel, "disk_OOD_overall_R2": ov_r2,
           "rel_l2_by_set_time": {n: {str(TIMES[i]*DT): float(err_curves[n][i,0]) for i in range(len(TIMES))} for n in TESTSETS}}
with open(f"{OUT}/metrics_drift_ar_2d.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2)); print("saved drift_recovery_ar_2d.pdf")
