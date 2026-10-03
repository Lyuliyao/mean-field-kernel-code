import numpy as np
import matplotlib.pyplot as plt
import sys

seed = int(sys.argv[1])
N = int(sys.argv[2])

data_RF_train = np.load(f"../gp/opinion_traj_test_pred_{N}_{seed}.npy")
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
plt.figure(figsize=(18, 6))
plt.subplots_adjust(wspace=0, hspace=0)
f_size = 24

# common grid for KDEs
xmin = min(data_RF_train[0,:].min(), data_train_16000[0,:].min(), data_test[0,:].min(), -2.5)
xmax = max(data_RF_train[-1,:].max(), data_train_16000[-1,:].max(), data_test[-1,:].max(),  2.5)
grid = np.linspace(xmin, xmax, 512)

# Panel 1: t=0
ax1 = plt.subplot(1,3,1)
ax1.set_title(r"$t=0$", fontsize=f_size)
style_axes(ax1, show_y=True, f_size=f_size)

pdf_RF_train = kde_pdf(data_RF_train[0,:], grid)
pdf_train_16000 = kde_pdf(data_train_16000[0,:], grid)
pdf_test  = kde_pdf(data_test[0,:],  grid)
ax1.plot(grid, pdf_RF_train, label="Gaussian Process Model")
ax1.plot(grid, pdf_train_16000, label="Learned MVNN Dynamics")
ax1.plot(grid, pdf_test, label="Simulation (Reference)")

ax1.set_xticks([-2,0,2])
#ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
ax1.set_yticks([0,0.5,1.0,1.5,2.0])
ax1.set_xlabel("x", fontsize=f_size)
ax1.set_ylabel("p(x)", fontsize=f_size)
ax1.legend(loc="upper center", fontsize=18)

# Panel 2: t=2  (index 200)
ax2 = plt.subplot(1,3,2, sharey=ax1)
ax2.set_title(r"$t=0.5$", fontsize=f_size)
style_axes(ax2, show_y=False, f_size=f_size)

pdf_RF_train = kde_pdf(data_RF_train[10,:], grid)
pdf_train_16000 = kde_pdf(data_train_16000[10,:], grid)
pdf_test  = kde_pdf(data_test[10,:],  grid)
ax2.plot(grid, pdf_RF_train)
ax2.plot(grid, pdf_train_16000)
ax2.plot(grid, pdf_test)

ax2.set_xticks([-2,0,2])
ax2.set_xlabel("x", fontsize=f_size)

# Panel 3: t=4  (last)
ax3 = plt.subplot(1,3,3, sharey=ax1)
ax3.set_title(r"$t=1.0$", fontsize=f_size)
style_axes(ax3, show_y=False, f_size=f_size)

pdf_RF_train = kde_pdf(data_RF_train[-1,:], grid)
pdf_train_16000 = kde_pdf(data_train_16000[-1,:], grid)
pdf_test  = kde_pdf(data_test[-1,:],  grid)
ax3.plot(grid, pdf_RF_train)
ax3.plot(grid, pdf_train_16000)
ax3.plot(grid, pdf_test)

ax3.set_xticks([-2,0,2])
ax3.set_xlabel("x", fontsize=f_size)

plt.savefig(f"result_kde_test_N={N}_seed={seed}.pdf", dpi=600, bbox_inches='tight')
