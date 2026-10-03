import numpy as np
import matplotlib.pyplot as plt
import sys

seed = int(sys.argv[1])
N = int(sys.argv[2])

data_RF_train = np.load(f"../gp/opinion_traj_test_pred_{N}_{seed}.npy")
data_train_16 = np.load(f"../sim3/data_tf=1.0/x_save_{N}_{seed}.npy")
data_train_16000 = np.load(f"../sim3/data/x_save_{N}_{seed}.npy")
data_test = np.load(f"../test_data3/opinion_traj_{N}_{seed}.npy")

def kde_pdf(x, grid, bw=None):
    """Gaussian KDE (Silverman's rule if bw is None), returns density on grid."""
    x = np.asarray(x).ravel()
    n = x.size
    if n < 2 or np.std(x) == 0:
        return np.zeros_like(grid)
    if bw is None:
        bw = 1.06 * np.std(x) * n ** (-1/5)  # Silverman's rule
        if bw <= 0 or not np.isfinite(bw):
            bw = 1.0
    u = (grid[:, None] - x[None, :]) / bw       # (G, n)
    phi = np.exp(-0.5 * u*u) / np.sqrt(2*np.pi) # standard normal pdf
    return phi.mean(axis=1) / bw                 # density integrates to 1

def style_axes(ax, show_y=True, f_size=24):
    for side in ['right','left','top','bottom']:
        ax.spines[side].set_linewidth(2)
    ax.xaxis.set_tick_params(which='major', size=6, width=2.5, direction='out', top=False)
    ax.xaxis.set_tick_params(which='minor', size=3, width=2, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=6, width=2.5, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=3, width=2, direction='out', right=False)
    if not show_y:
        ax.yaxis.set_visible(False)
    ax.tick_params(labelsize=f_size)

# figure + panels
plt.figure(figsize=(6, 6))
plt.subplots_adjust(wspace=0, hspace=0)
f_size = 24

# common grid for KDEs
xmin = min(data_RF_train[0,:].min(), data_train_16[0,:].min(), data_train_16000[0,:].min(), data_test[0,:].min(), -2.5)
xmax = max(data_RF_train[-1,:].max(), data_train_16[-1,:].max(), data_train_16000[-1,:].max(), data_test[-1,:].max(),  2.5)
grid = np.linspace(xmin, xmax, 512)

# Panel 4: L2 error
ax4 = plt.subplot(1,1,1)
ax4.set_title(r"$L^2$ error", fontsize=f_size)
style_axes(ax4, show_y=True, f_size=f_size)

timesteps = data_RF_train.shape[0]
print(timesteps)
l2_error_gp = np.zeros(timesteps)
l2_error_mvnn_16 = np.zeros(timesteps)
l2_error_mvnn_16000 = np.zeros(timesteps)

for t in range(timesteps):
    pdf_RF_train = kde_pdf(data_RF_train[t,:], grid)
    pdf_train_16 = kde_pdf(data_train_16[t,:], grid)
    pdf_train_16000 = kde_pdf(data_train_16000[t,:], grid)
    pdf_test     = kde_pdf(data_test[t,:], grid)
    l2_error_gp[t] = np.linalg.norm(pdf_RF_train - pdf_test)
    l2_error_mvnn_16[t] = np.linalg.norm(pdf_train_16 - pdf_test)
    l2_error_mvnn_16000[t] = np.linalg.norm(pdf_train_16000 - pdf_test)

ax4.plot(np.linspace(0, 1.0, timesteps), l2_error_gp, label="Gaussian Process Model")
ax4.plot(np.linspace(0, 1.0, timesteps), l2_error_mvnn_16, label="MVNN 1")
ax4.plot(np.linspace(0, 1.0, timesteps), l2_error_mvnn_16000, label="MVNN 2")
ax4.legend(loc="upper center", fontsize=18)

ax4.set_xticks([0,0.5,1.0])
ax4.set_xlabel("t", fontsize=f_size)
ax4.set_yticks([0,1.0,2.0,3.0,4.0,5.0,6.0])
ax4.set_ylabel(r"$\|\rho_{learned}-\rho_{sim}\|_{L^2}$", fontsize=f_size)

print(l2_error_gp[-1])
print(l2_error_mvnn_16[-1])
print(l2_error_mvnn_16000[-1])

plt.savefig(f"result_kde_test_error_N={N}_seed={seed}.pdf", dpi=600, bbox_inches='tight')
