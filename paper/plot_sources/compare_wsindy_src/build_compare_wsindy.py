#!/usr/bin/env python3
"""Rebuild figure/compare_wsindy.pdf (manuscript Figure 9) in the paper's figure style.

Data sources
- MVNN / mean-field (kernel) WSINDy / local PDE-WSINDy / reference curves:
  the frozen confirmation cache rebuttal_figure_data.npz of
  SIMPLE/revision/experiments/r1c1_threebody_global_mean (display triplet 100,
  mixed measure; L2 curves averaged over the 36 confirmation trajectories;
  MVNN line = mean over 3 seeds, band = min/max).
- Online parameter estimation: purple_curves.npz, the exact vector paths of the
  previously published figure (built from the uploaded param-est trajectories,
  whose raw files are no longer on disk), mapped back to data coordinates.

Style follows figure/figure.py (plot_simple_case_with_l2): usetex with
Computer Modern sans-serif text, f_size 32, legend 23, linewidth 2.5.
Run:  TEXINPUTS=$(dirname $0)//: /mnt/home/lyuliyao/.conda/envs/jax-baseline/bin/python build_compare_wsindy.py
(type1ec.sty here is a stub standing in for the absent cm-super package).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.lines import Line2D

import os
S = os.path.dirname(os.path.abspath(__file__)) + "/"   # purple_curves.npz and type1ec.sty live here
EXP = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/SIMPLE/revision/experiments/r1c1_threebody_global_mean"
OUT = "/mnt/ufs18/rs/MultiscaleML_group/Liyao/MEAN_FIELD_KERNEL/Mean-field-kernel/figure/compare_wsindy"

cache = np.load(EXP + "/outputs/threebody/final/summary/rebuttal_figure_data.npz")
P = np.load(S + "purple_curves.npz")

plt.rcParams.update({"text.usetex": True, "font.family": "sans-serif",
                     "font.sans-serif": ["Computer Modern Roman"]})

def style_axes(ax, show_y=True, f_size=32):
    for side in ["right", "left", "top", "bottom"]:
        ax.spines[side].set_linewidth(2)
    ax.xaxis.set_tick_params(which="major", size=6, width=2.5, direction="in", top=False)
    ax.yaxis.set_tick_params(which="major", size=6, width=2.5, direction="in", right=False)
    if not show_y:
        ax.yaxis.set_visible(False)
    ax.tick_params(labelsize=f_size)

C = {"mvnn": "#1f77b4", "ref": "#ff7f0e", "mf": "#2ca02c", "local": "#d62728", "pe": "#9467bd"}
LW = 2.5
f_size = 32
x, t, mv = cache["x"], cache["times"], cache["l2_mvnn"]
purple_d = [P["d0"], P["d1"], P["d2"]]

plt.figure(figsize=(24, 6))
plt.subplots_adjust(wspace=0, hspace=0)
ax1 = None
for k, (t_snap, fidx) in enumerate(zip((0, 0.5, 1), (0, 10, 20))):
    ax = plt.subplot(1, 4, k + 1, sharey=ax1)
    ax1 = ax1 or ax
    ax.set_title(rf"$t={t_snap:g}$", fontsize=f_size)
    style_axes(ax, show_y=(k == 0), f_size=f_size)
    ax.plot(x, cache["snap_local"][fidx], color=C["local"], linewidth=LW)
    ax.plot(x, cache["snap_kernel"][fidx], color=C["mf"], linewidth=LW)
    ax.plot(purple_d[k][:, 0], purple_d[k][:, 1], color=C["pe"], linewidth=LW)
    ax.plot(x, cache["snap_reference"][fidx], color=C["ref"], linewidth=LW)
    ax.plot(x, cache["snap_mvnn"][fidx], color=C["mvnn"], linewidth=LW)
    ax.set_xlim(-3.5, 3.0)
    ax.set_xticks([-2, 0, 2])
    ax.set_xlabel(r"$x$", fontsize=f_size)
    if k == 0:
        ax.set_ylabel(r"$\rho(x)$", fontsize=f_size)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        handles = [Line2D([], [], color=C["mvnn"], lw=LW, label="MVNN"),
                   Line2D([], [], color=C["mf"], lw=LW, label="Mean-field WSINDy"),
                   Line2D([], [], color=C["local"], lw=LW, label="Local PDE-WSINDy"),
                   Line2D([], [], color=C["pe"], lw=LW, label="Online parameter estimation"),
                   Line2D([], [], color=C["ref"], lw=LW, label="Simulation (Reference)")]
        ax.legend(handles=handles, fontsize=20, loc="upper left", handlelength=1.0,
                  handletextpad=0.5, borderaxespad=0.4, labelspacing=0.35)

ax4 = plt.subplot(1, 4, 4)
ax4.set_title(r"$L^2$ error", fontsize=f_size)
style_axes(ax4, show_y=True, f_size=f_size)
ax4.fill_between(t, mv.min(0), mv.max(0), color=C["mvnn"], alpha=0.3, linewidth=0)
ax4.plot(t, cache["l2_local"], color=C["local"], linewidth=LW)
ax4.plot(t, cache["l2_kernel"], color=C["mf"], linewidth=LW)
ax4.plot(P["err"][:, 0], P["err"][:, 1], color=C["pe"], linewidth=LW)
ax4.plot(t, mv.mean(0), color=C["mvnn"], linewidth=LW)
ax4.set_xlabel(r"$t$", fontsize=f_size)
ax4.set_ylabel(r"$\|\rho_{\mathrm{model}}-\rho_{\mathrm{ref}}\|_{L^2}$", fontsize=f_size)
ax4.set_xticks([0, 0.5, 1])
ax4.yaxis.set_major_locator(MultipleLocator(0.5))
ymax = max(cache["l2_local"].max(), cache["l2_kernel"].max(), P["err"][:, 1].max(), mv.max())
ax4.set_ylim(0, 1.06 * ymax)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.tick_params(axis="y", labelleft=False, left=False, right=True)

# Figure.savefig (not pyplot.savefig) avoids an Agg redraw that would need dvipng.
plt.gcf().savefig(OUT + ".pdf", dpi=200, bbox_inches="tight")
print("wrote", OUT + ".pdf", " L2 ymax =", round(float(ymax), 3))
