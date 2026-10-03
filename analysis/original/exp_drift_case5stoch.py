"""
E4 (stochastic Motsch-Tadmor, sigma=0.1): why the likelihood (Girsanov) objective
matters even though it reduces to MSE.

Under the SDE, the per-sample finite-difference velocity is
    V = (X_{t+dt}-X_t)/dt = b*(X,mu) + sigma * dB/dt ,
whose noise has standard deviation sigma/sqrt(dt) = 0.1/0.1 = 1.0, i.e. LARGER than
the drift magnitude (|b*| <~ 0.6). The MLE derived from Girsanov is exactly the MSE
regression of V onto b_theta; it is a *consistent* estimator (Sharrock et al.) because
E[V | X, mu] = b*. The MVNN denoises by pooling N*L*M samples -> recovers b* even
though each target is noise-dominated. A naive per-sample velocity fit cannot.
"""
import numpy as np, sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from mvnn_lib import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/SIMPLE/CASE5_stochastic"
OUT  = os.path.join(os.path.dirname(__file__), "figs")
DT, ELL, SIGMA = 1e-2, 0.5, 0.1
CKPT = f"{CASE}/TRAINING3/model_save_path/jax_ckpt_000400.npz"
TIMES = [20, 100, 200, 300]
NQ = 1500

params = load_params(CKPT)
files = sorted(glob.glob(f"{CASE}/TEST_DATA3/data/opinion_traj_*.npy"),
               key=lambda s: int(s.split('_')[-1].split('.')[0]))
print(len(files), "stochastic test trajectories")
rng = np.random.default_rng(0)

V_all, bt_all, bl_all = [], [], []
err_time = []                      # rel L2 of learned drift vs true, per time
for tf in files[:6]:
    traj = np.load(tf)             # (401, N)
    row = []
    for t in TIMES:
        X = traj[t][:, None]
        Xn = traj[t+1][:, None]
        V = (Xn - X)/DT                                   # noisy per-sample target
        mu = embedding_mean(params, X)
        q = rng.choice(X.shape[0], min(NQ, X.shape[0]), replace=False)
        Xq = X[q]
        bl = mvnn_drift(params, Xq, mu)
        bt = true_drift_motsch_tadmor(Xq, X, ell=ELL)
        row.append(rel_l2(bl, bt))
        if tf == files[0]:
            V_all.append(V[q].ravel()); bt_all.append(bt.ravel()); bl_all.append(bl.ravel())
    err_time.append(row)
V_all = np.concatenate(V_all); bt_all = np.concatenate(bt_all); bl_all = np.concatenate(bl_all)
err_time = np.array(err_time)

rel_V  = rel_l2(V_all, bt_all)        # raw target error (huge)
rel_bl = rel_l2(bl_all, bt_all)       # MLE estimate error (small)
r2_bl  = r2_score(bl_all, bt_all)
noise_std = SIGMA/np.sqrt(DT)

# binned average of V to show E[V|x] ~ b*(x): denoising by conditional mean
order = np.argsort(bt_all)
nb = 25
edges = np.quantile(bt_all, np.linspace(0,1,nb+1))
cen = 0.5*(edges[1:]+edges[:-1]); Vbin = np.full(nb, np.nan)
for i in range(nb):
    m = (bt_all>=edges[i]) & (bt_all<edges[i+1])
    if m.sum()>0: Vbin[i] = V_all[m].mean()

# ============================ figure ============================
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

ax[0].scatter(bt_all, V_all, s=3, alpha=0.12, color="0.5")
ax[0].plot(cen, Vbin, 'C3o-', ms=4, lw=1.5, label=r"binned mean $\mathbb{E}[V\,|\,b^\star]$")
lim = [bt_all.min(), bt_all.max()]
ax[0].plot(lim, lim, 'r--', lw=1.3, label="ideal")
ax[0].set_xlabel(r"true drift $b^\star$"); ax[0].set_ylabel(r"finite-diff velocity $V=\Delta X/\Delta t$")
ax[0].set_title(f"(a) Raw targets are noise-dominated\nnoise std $\\sigma/\\sqrt{{\\Delta t}}$={noise_std:.1f}, rel-$L^2$(V,$b^\\star$)={rel_V:.1f}")
ax[0].legend(fontsize=9)

ax[1].scatter(bt_all, bl_all, s=4, alpha=0.3, color="C0")
ax[1].plot(lim, lim, 'r--', lw=1.3)
ax[1].set_xlabel(r"true drift $b^\star$"); ax[1].set_ylabel(r"learned drift $b_\theta$")
ax[1].set_title(f"(b) MLE/MVNN recovers the drift\n$R^2$={r2_bl:.3f}, rel-$L^2$={rel_bl:.3f}")

m, s = err_time.mean(0), err_time.std(0)
tt = np.array(TIMES)*DT
ax[2].plot(tt, m, 'C0o-', lw=2)
ax[2].fill_between(tt, m-s, m+s, alpha=0.2)
ax[2].set_xlabel("time $t$"); ax[2].set_ylabel(r"relative $L^2$ drift error")
ax[2].set_title("(c) Learned-drift error over time (stochastic)")
ax[2].set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(f"{OUT}/drift_recovery_stoch.pdf", bbox_inches="tight")
fig.savefig(f"{OUT}/drift_recovery_stoch.png", dpi=130, bbox_inches="tight")

metrics = {"case": "CASE5_stochastic_MotschTadmor", "sigma": SIGMA,
           "noise_std_per_sample": float(noise_std),
           "rel_l2_rawV_vs_true": float(rel_V),
           "rel_l2_learned_vs_true": float(rel_bl), "R2_learned": float(r2_bl),
           "rel_l2_by_time": {str(TIMES[i]*DT): float(err_time.mean(0)[i]) for i in range(len(TIMES))}}
with open(f"{OUT}/metrics_drift_stoch.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2)); print("saved drift_recovery_stoch.pdf")
