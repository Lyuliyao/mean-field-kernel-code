import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"]})

seed = int(sys.argv[1])
N = 16000

data_GP_train = np.load(f"/Users/tracy/OneDrive/Documents/Research UCLA/GPDynamics - Copy/data_tf=1.0/opinion_traj_gp_{N}_{seed}.npy")
data_MVNN_train = np.load(f"/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_neural_network/simple/case5/sim3/data_tf=1.0/x_save_{seed}.npy")
data_MVNN_train_2 = np.load(f"/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_neural_network/simple/case5/sim3/data_big_training_set_tf=1.0/x_save_{seed}.npy")
data_test = np.load(f"/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_neural_network/simple/case5/test_data3/data_test_tf=1.0/opinion_traj_{N}_{seed}.npy")

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

def style_axes(ax, show_y=True, f_size=28):
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
f_size = 28

# common grid for KDEs
xmin = min(data_GP_train[0,:].min(), data_MVNN_train[0,:].min(), data_MVNN_train_2[0,:].min(), data_test[0,:].min(), -2.5)
xmax = max(data_GP_train[-1,:].max(), data_MVNN_train[-1,:].max(), data_MVNN_train_2[-1,:].max(), data_test[-1,:].max(),  2.5)
grid = np.linspace(xmin, xmax, 512)

# Panel 1: t=0
ax1 = plt.subplot(1,3,1)
ax1.set_title(r"$t=0$", fontsize=f_size)
style_axes(ax1, show_y=True, f_size=f_size)

pdf_GP_train = kde_pdf(data_GP_train[0,:], grid)
pdf_MVNN_train = kde_pdf(data_MVNN_train[0,:], grid)
pdf_MVNN_train_2 = kde_pdf(data_MVNN_train_2[0,:], grid)
pdf_test  = kde_pdf(data_test[0,:],  grid)
ax1.plot(grid, pdf_GP_train, label="Gaussian Process Model", linewidth=2.5)
ax1.plot(grid, pdf_MVNN_train, label="MVNN Trained on 16", linewidth=2.5)
ax1.plot(grid, pdf_MVNN_train_2, label="MVNN Trained on 16000", linewidth=2.5)
ax1.plot(grid, pdf_test, label="Simulation (Reference)", linewidth=2.5)

ax1.set_xticks([-2,0,2])
# ax1.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
ax1.set_yticks([0,0.5,1.0,1.5,2.0])
ax1.set_xlabel(r"$x$", fontsize=f_size)
ax1.set_ylabel(r"$\rho(x)$", fontsize=f_size)
ax1.legend(loc="upper center", fontsize=22)

# Panel 2: t=2  (index 200)
ax2 = plt.subplot(1,3,2, sharey=ax1)
ax2.set_title(r"$t=0.5$", fontsize=f_size)
style_axes(ax2, show_y=False, f_size=f_size)

pdf_GP_train = kde_pdf(data_GP_train[10,:], grid)
pdf_MVNN_train = kde_pdf(data_MVNN_train[10,:], grid)
pdf_MVNN_train_2 = kde_pdf(data_MVNN_train_2[10,:], grid)
pdf_test  = kde_pdf(data_test[10,:],  grid)
ax2.plot(grid, pdf_GP_train, linewidth=2.5)
ax2.plot(grid, pdf_MVNN_train, linewidth=2.5)
ax2.plot(grid, pdf_MVNN_train_2, linewidth=2.5)
ax2.plot(grid, pdf_test, linewidth=2.5)

ax2.set_xticks([-2,0,2])
ax2.set_xlabel(r"$x$", fontsize=f_size)

# Panel 3: t=4  (last)
ax3 = plt.subplot(1,3,3, sharey=ax1)
ax3.set_title(r"$t=1.0$", fontsize=f_size)
style_axes(ax3, show_y=False, f_size=f_size)

pdf_GP_train = kde_pdf(data_GP_train[-1,:], grid)
pdf_MVNN_train = kde_pdf(data_MVNN_train[-1,:], grid)
pdf_MVNN_train_2 = kde_pdf(data_MVNN_train_2[-1,:], grid)
pdf_test  = kde_pdf(data_test[-1,:],  grid)
ax3.plot(grid, pdf_GP_train, linewidth=2.5)
ax3.plot(grid, pdf_MVNN_train, linewidth=2.5)
ax3.plot(grid, pdf_MVNN_train_2, linewidth=2.5)
ax3.plot(grid, pdf_test, linewidth=2.5)

ax3.set_xticks([-2,0,2])
ax3.set_xlabel(r"$x$", fontsize=f_size)

plt.savefig(f"result_gp_mvnn_N={N}_seed={seed}.pdf", dpi=600, bbox_inches='tight')