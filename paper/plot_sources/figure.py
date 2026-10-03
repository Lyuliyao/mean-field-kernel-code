#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MaxNLocator
import os
os.environ["PATH"] = "/apps/spack/anvil/apps/texlive/20200406-gcc-8.4.1-gjynmo4/bin/x86_64-linux:" + os.environ.get("PATH","")


# In[8]:


matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"]})


# In[9]:


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

def style_axes(ax, show_y=True, f_size=32):
    for side in ['right','left','top','bottom']:
        ax.spines[side].set_linewidth(2)
    ax.xaxis.set_tick_params(which='major', size=6, width=2.5, direction='in', top=False)
    ax.xaxis.set_tick_params(which='minor', size=3, width=2, direction='in', top=False)
    ax.yaxis.set_tick_params(which='major', size=6, width=2.5, direction='in', right=False)
    ax.yaxis.set_tick_params(which='minor', size=3, width=2, direction='in', right=False)
    if not show_y:
        ax.yaxis.set_visible(False)
    ax.tick_params(labelsize=f_size)


# In[14]:


def plot_simple_case_with_l2(
    case_id,
    y_locator_step,
    panel3_model_label="Learned MVNN dynamics",
    test_path_template="../CASE5/TEST_DATA3/data/opinion_traj_{case_id}.npy",
    train_path_template="../CASE5/SIM3/data/x_save_{case_id}.npy",
    output_prefix="simple_case",
):
    data_test = np.load(test_path_template.format(case_id=case_id))
    data_train = np.load(train_path_template.format(case_id=case_id))

    plt.figure(figsize=(24, 6))
    plt.subplots_adjust(wspace=0, hspace=0)
    f_size = 32

    xmin = min(data_train[0, :].min(), data_test[0, :].min(), -2.5)
    xmax = max(data_train[-1, :].max(), data_test[-1, :].max(), 2.5)
    grid = np.linspace(xmin, xmax, 512)

    n_steps = min(data_train.shape[0], data_test.shape[0])
    time_axis = np.linspace(0, 2, n_steps)
    l2_error = np.zeros(n_steps)
    for i in range(n_steps):
        pdf_train_i = kde_pdf(data_train[i, :], grid)
        pdf_test_i = kde_pdf(data_test[i, :], grid)
        l2_error[i] = np.sqrt(np.trapz((pdf_train_i - pdf_test_i) ** 2, grid))

    # Panel 1: t=0
    ax1 = plt.subplot(1, 4, 1)
    ax1.set_title(r"$t=0$", fontsize=f_size)
    style_axes(ax1, show_y=True, f_size=f_size)
    pdf_train = kde_pdf(data_train[0, :], grid)
    pdf_test = kde_pdf(data_test[0, :], grid)
    ax1.plot(grid, pdf_train, label="Learned MVNN dynamics", linewidth=2.5)
    ax1.plot(grid, pdf_test, label="Simulation", linewidth=2.5)
    ax1.set_xticks([-2, 0, 2])
    plt.xlabel(r"$x$", fontsize=f_size)
    plt.ylabel(r"$\rho(x)$", fontsize=f_size)
    ax1.yaxis.set_major_locator(MultipleLocator(y_locator_step))

    # Panel 2: t=1
    ax2 = plt.subplot(1, 4, 2, sharey=ax1)
    ax2.set_title(r"$t=1$", fontsize=f_size)
    style_axes(ax2, show_y=False, f_size=f_size)
    pdf_train = kde_pdf(data_train[100, :], grid)
    pdf_test = kde_pdf(data_test[100, :], grid)
    ax2.plot(grid, pdf_train, label="Learned MVNN dynamics", linewidth=2.5)
    ax2.plot(grid, pdf_test, label="Simulation (Reference)", linewidth=2.5)
    plt.xlabel(r"$x$", fontsize=f_size)
    ax2.legend(fontsize=23)
    ax2.set_xticks([-2, 0, 2])

    # Panel 3: t=2
    ax3 = plt.subplot(1, 4, 3, sharey=ax1)
    ax3.set_title(r"$t=2$", fontsize=f_size)
    style_axes(ax3, show_y=False, f_size=f_size)
    pdf_train = kde_pdf(data_train[200, :], grid)
    pdf_test = kde_pdf(data_test[200, :], grid)
    ax3.plot(grid, pdf_train, label=panel3_model_label, linewidth=2.5)
    ax3.plot(grid, pdf_test, label="Simulation (Reference)", linewidth=2.5)
    ax3.set_xticks([-2, 0, 2])
    plt.xlabel(r"$x$", fontsize=f_size)

    # Panel 4: L2 error over time
    ax4 = plt.subplot(1, 4, 4)
    ax4.set_title(r"$L^2$ error", fontsize=f_size)
    style_axes(ax4, show_y=True, f_size=f_size)
    ax4.plot(time_axis, l2_error, color="tab:red", linewidth=2.5)
    ax4.set_xlabel(r"$t$", fontsize=f_size)
    ax4.set_ylabel(r"$\|\rho_{\mathrm{NN}}-\rho_{\mathrm{sim}}\|_{L^2}$", fontsize=f_size)
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.tick_params(axis="y", labelleft=False, left=False, right=True)

    plt.savefig(f"{output_prefix}_{case_id}.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f"{output_prefix}_{case_id}.png", dpi=200, bbox_inches="tight")


# %%
plot_simple_case_with_l2(case_id=1, y_locator_step=0.25, panel3_model_label="Learned MVNN dynamics")

# In[15]:
plot_simple_case_with_l2(case_id=2, y_locator_step=0.3, panel3_model_label="Learned MVNN dynamics")

# In[16]:
plot_simple_case_with_l2(case_id=3, y_locator_step=0.25, panel3_model_label="Learned MVNN dynamics")

# In[16]:
plot_simple_case_with_l2(
    case_id=1,
    y_locator_step=0.25,
    panel3_model_label="Learned MVNN dynamics",
    test_path_template="../CASE5_stochastic/TEST_DATA3/data/opinion_traj_{case_id}.npy",
    train_path_template="../CASE5_stochastic/SIM3/data/x_save_{case_id}.npy",
    output_prefix="stochastic_case",
)



def plot_smooth_density(
    ax,
    x,
    y,
    xmax,
    vmax,
    cmap="viridis",
    bin_width=0.1,
    sigma_bins=1.2,
    method="kde",
):
    edges = np.arange(-xmax, xmax + bin_width, bin_width)
    xedges, yedges = edges, edges

    if method == "kde":
        try:
            from scipy.stats import gaussian_kde
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            xx, yy = np.meshgrid(xc, yc, indexing="xy")
            sample = np.vstack([x, y])
            kde = gaussian_kde(sample)
            hist = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(len(yc), len(xc)).T
        except Exception:
            hist, _, _ = np.histogram2d(x, y, bins=[edges, edges], density=True)
            radius = max(1, int(3 * sigma_bins))
            grid = np.arange(-radius, radius + 1)
            kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
            kernel = kernel / kernel.sum()
            hist = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=hist)
            hist = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=hist)
    else:
        hist, _, _ = np.histogram2d(x, y, bins=[edges, edges], density=True)
        radius = max(1, int(3 * sigma_bins))
        grid = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()
        hist = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=hist)
        hist = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=hist)

    return ax.imshow(
        hist.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="bicubic",
        aspect="equal",
    )


def plot_simple_2d_case(
    result_train,
    result_test,
    save_stem,
    vmax,
    xmax,
    bin_size,
    ticks,
    fig_size=(21.43, 12),
    f_size=36,
    tick_font_size=36,
    cmap="viridis",
    dpi=200,
):
    plt.figure(figsize=fig_size)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    panel_data = [
        (result_test[0], r"$t=0$", True, True),
        (result_test[100], r"$t=1$", False, True),
        (result_test[-1], r"$t=2$", False, True),
        (result_train[0], None, True, False),
        (result_train[100], None, False, False),
        (result_train[-1], None, False, False),
    ]
    axes = []
    im_last = None
    for i, (data, title, show_y, hide_x) in enumerate(panel_data):
        ax = plt.subplot(2, 3, i + 1, sharex=axes[0] if axes else None)
        axes.append(ax)
        if title is not None:
            ax.set_title(title, fontsize=f_size)
        for side in ["right", "left", "top", "bottom"]:
            ax.spines[side].set_linewidth(0)
        ax.xaxis.set_tick_params(which="major", size=6, width=2.5, direction="in", top=False)
        ax.xaxis.set_tick_params(which="minor", size=3, width=2, direction="in", top=False)
        ax.yaxis.set_tick_params(which="major", size=6, width=2.5, direction="in", right=False)
        ax.yaxis.set_tick_params(which="minor", size=3, width=2, direction="in", right=False)
        im_last = plot_smooth_density(ax, data[:, 0], data[:, 1], xmax=xmax, vmax=vmax, cmap=cmap)
        ax.set_xlim([-xmax, xmax])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=tick_font_size)
        if not show_y:
            ax.yaxis.set_visible(False)
        if hide_x:
            ax.xaxis.set_visible(False)

    cbar = plt.colorbar(im_last, ax=axes, ticks=np.arange(0, vmax + 0.1, bin_size), pad=0.01)
    cbar.ax.tick_params(labelsize=tick_font_size)
    plt.savefig(f"{save_stem}.pdf", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.savefig(f"{save_stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0)

# %%

result_train = np.load("../CASE4/SIM/result.npy")
result_test = np.load("../CASE4/TEST_DATA/result.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="simple_2d_ring",
    vmax=0.6,
    xmax=1.8,
    bin_size=0.2,
    ticks=[-1, 0, 1],
)

# In[18]:
result_train = np.load("../CASE4/SIM2/result.npy")#result.npy")
result_test = np.load("../CASE4/TEST_DATA2/result.npy")#result.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="simple_2d_2ring",
    vmax=0.4,
    xmax=2.4,
    bin_size=0.2,
    ticks=[-2, 0, 2],
)


# In[ ]:
result_train = np.load("../CASE4/SIM3/result.npy")#result.npy")
result_test = np.load("../CASE4/TEST_DATA3/result.npy")#result.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="simple_2d_disk",
    vmax=0.4,
    xmax=2.4,
    bin_size=0.1,
    ticks=[-2, 0, 2],
)

# In[ ]:


result_train = np.load("../CASE4/SIM4/result.npy")#result.npy")
result_test = np.load("../CASE4/TEST_DATA4/result.npy")#result.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="simple_2d_hl",
    vmax=0.1,
    xmax=2.4,
    bin_size=0.05,
    ticks=[-2, 0, 2],
)



# %%
result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_1/result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/test_data_second_order_1/result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="ring_x",
    vmax=1,
    xmax=3.0,
    bin_size=0.25,
    ticks=[-2, 0, 2],
    dpi=200,
)


result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_1/v_result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/test_data_second_order_1/v_result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="ring_v",
    vmax=1.5,
    xmax=3.0,
    bin_size=0.5,
    ticks=[-2, 0, 2],
    dpi=200,
)


# %%
result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_2/result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/test_data_second_order_2/result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="2ring_x",
    vmax=0.8,
    xmax=3.0,
    bin_size=0.2,
    ticks=[-2, 0, 2],
    dpi=200,
)


result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_2/v_result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/test_data_second_order_2/v_result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="2ring_v",
    vmax=1.5,
    xmax=3.0,
    bin_size=0.5,
    ticks=[-2, 0, 2],
    dpi=200,
)

# %%
result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_3/result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/test_data_second_order_3/result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="disk_x",
    vmax=0.4,
    xmax=3.0,
    bin_size=0.1,
    ticks=[-2, 0, 2],
    dpi=200,
)


result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_3/v_result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/test_data_second_order_3/v_result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="disk_v",
    vmax=0.4,
    xmax=3.0,
    bin_size=0.1,
    ticks=[-2, 0, 2],
    dpi=200,
)




# %%
# Converted from /anvil/projects/x-mth260007/Tracy/case4/result/2ring_kde.py
result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_4/result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_4/result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="binary_x",
    vmax=0.1,
    xmax=3.0,
    bin_size=0.05,
    ticks=[-2, 0, 2],
    dpi=200,
)

# %%
result_train = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_4/v_result_second_order.npy")
result_test = np.load("/anvil/projects/x-mth260007/Tracy/case4/sim_second_order_4/v_result_second_order.npy")
plot_simple_2d_case(
    result_train=result_train,
    result_test=result_test,
    save_stem="binary_v",
    vmax=1.6,
    xmax=3.0,
    bin_size=0.5,
    ticks=[-2, 0, 2],
    dpi=200,
)

# %%


# In[ ]:


def plot_simple_case_multi_species(case_id, y_steps):
    f_size = 32
    sim_root = "../CASE7_opinion_dynamics/SIM3/data"
    test_root = "../CASE7_opinion_dynamics/TEST_CASE3/data"
    species = ["workers", "managers", "ceos"]
    rho_labels = [r"$\rho(x_1)$", r"$\rho(x_2)$", r"$\rho(x_3)$"]
    time_idx = [0, 250, -1]
    time_titles = [r"$t=0$", r"$t=0.25$", r"$t=0.5$"]

    # Build a common x-grid from workers data for this case.
    data_train_ref = np.load(f"{sim_root}/opinion_workers_mpi_{case_id}.npy")
    data_test_ref = np.load(f"{test_root}/opinion_workers_mpi_{case_id}.npy")
    xmin = min(data_train_ref[0, :].min(), data_test_ref[0, :].min(), -2.5)-2
    xmax = max(data_train_ref[-1, :].max(), data_test_ref[-1, :].max(), 2.5)+2
    grid = np.linspace(xmin, xmax, 512)

    plt.figure(figsize=(24, 18))
    plt.subplots_adjust(wspace=0, hspace=0)

    ax_row0_col0 = None
    ax_row0_col2 = None
    ax_row0_col3 = None
    for r, sp in enumerate(species):
        data_train = np.load(f"{sim_root}/opinion_{sp}_mpi_{case_id}.npy")
        data_test = np.load(f"{test_root}/opinion_{sp}_mpi_{case_id}.npy")
        n_steps = min(data_train.shape[0], data_test.shape[0])
        time_axis = np.linspace(0, 0.5, n_steps)
        l2_error = np.zeros(n_steps)
        for i in range(n_steps):
            pdf_train_i = kde_pdf(data_train[i, :], grid)
            pdf_test_i = kde_pdf(data_test[i, :], grid)
            l2_error[i] = np.sqrt(np.trapz((pdf_train_i - pdf_test_i) ** 2, grid))

        row_first_ax = None
        for c, tidx in enumerate(time_idx):
            idx = r * 4 + c + 1
            sharex = ax_row0_col0 if (r > 0 and c == 0) else (ax_row0_col2 if (r > 0 and c > 0) else None)
            sharey = row_first_ax if c > 0 else None
            ax = plt.subplot(3, 4, idx, sharex=sharex, sharey=sharey)
            if c == 0:
                row_first_ax = ax
            if r == 0 and c == 0:
                ax_row0_col0 = ax
            if r == 0 and c == 2:
                ax_row0_col2 = ax

            if r == 0:
                ax.set_title(time_titles[c], fontsize=f_size)
            style_axes(ax, show_y=(c == 0), f_size=f_size)

            pdf_train = kde_pdf(data_train[tidx, :], grid)
            pdf_test = kde_pdf(data_test[tidx, :], grid)
            label_train = "Mean-field Neural network" if c == 0 else "Learned MVNN dynamics"
            ax.plot(grid, pdf_train, label=label_train, linewidth=2.5)
            ax.plot(grid, pdf_test, label="Simulation (Reference)", linewidth=2.5)

            if c == 0:
                ax.set_ylabel(rho_labels[r], fontsize=f_size)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            if r == 0:
                ax.set_xlabel(r"$x_1$", fontsize=f_size)
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.set_xticks([-10, -5, 0, 5])
            else:
                if c == 1:
                    ax.set_xlabel(r"$x$", fontsize=f_size)
                ax.xaxis.set_major_locator(MultipleLocator(5))

            if r == 0 and c == 1:
                ax.legend(fontsize=f_size)

        ax_err = plt.subplot(3, 4, r * 4 + 4, sharex=ax_row0_col3 if r > 0 else None)
        if r == 0:
            ax_row0_col3 = ax_err
            ax_err.set_title(r"$L^2$ error", fontsize=f_size)
        style_axes(ax_err, show_y=True, f_size=f_size)
        ax_err.plot(time_axis, l2_error, color="tab:red", linewidth=2.5)
        ax_err.set_xlabel(r"$t$", fontsize=f_size)
        ax_err.set_ylabel(r"$\|\rho_{\mathrm{NN}}-\rho_{\mathrm{sim}}\|_{L^2}$", fontsize=f_size)
        ax_err.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax_err.yaxis.tick_right()
        ax_err.yaxis.set_label_position("right")
        ax_err.tick_params(axis="y", labelleft=False, left=False, right=True)

    plt.savefig(f"simple_case_multi_sepcies_{case_id}.pdf", dpi=200, bbox_inches='tight')
    plt.savefig(f"simple_case_multi_sepcies_{case_id}.png", dpi=200, bbox_inches='tight')



# %%
plot_simple_case_multi_species(case_id=1, y_steps=(0.08, 0.2, 0.05))


# In[ ]:


plot_simple_case_multi_species(case_id=2, y_steps=(0.08, 0.1, 0.025))

# In[ ]:

plot_simple_case_multi_species(case_id=3, y_steps=(0.08, 0.1, 0.025))

# In[ ]:

plot_simple_case_multi_species(case_id=4, y_steps=(0.08, 0.1, 0.025))


# In[ ]:
