#!/usr/bin/env python3
"""Redraw manuscript Figures 1, 5 and 9 from their archived numerical results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT=Path(__file__).resolve().parents[1]

def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(out/(stem+'.'+ext),bbox_inches='tight',dpi=180)
    plt.close(fig)

def learning_curve(out):
    data=pd.read_csv(ROOT/'outputs/mtcurve/summary/mvnn_learning_curve_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,metric,label in zip(axes,['relative_density_L2','drift_NRMSE'],['relative density $L^2$ error','drift NRMSE']):
        for role,color,name in [('training','C0','training'),('test','red','independent test')]:
            d=data[(data.metric==metric)&(data.trajectory_type==role)].sort_values('training_M')
            x=d.training_M.to_numpy();y=d['mean'].to_numpy()
            ax.plot(x,y,'o-',color=color,label=name,ms=4)
            ax.fill_between(x,d.ci95_low.to_numpy(),d.ci95_high.to_numpy(),color=color,alpha=.2,lw=0)
        ax.set_xscale('log',base=2);ax.set_xticks([5,10,20,40,100]);ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('number of training trajectories $M$');ax.set_ylabel(label)
    axes[0].legend(frameon=False);fig.tight_layout()
    save(fig,out,'figure_01_learning_curve')

def timing(out):
    d=pd.read_csv(ROOT/'data/timing/rollout_times.csv')
    fig,ax=plt.subplots(figsize=(6.5,4.3));right=ax.twinx()
    a=ax.plot(d.agents,d.mvnn_seconds,'o-',color='C0',label='MVNN')[0]
    b=right.plot(d.agents,d.gp_seconds,'s-',color='C1',label='Gaussian process')[0]
    ax.set_xscale('log',base=2);ax.set_xlabel('number of agents $N$')
    ax.set_ylabel('MVNN simulation time (s)',color='C0');right.set_ylabel('GP simulation time (s)',color='C1')
    ax.legend(handles=[a,b],frameon=False);fig.tight_layout()
    save(fig,out,'figure_05_recorded_timing')

def threebody(out):
    with np.load(ROOT/'outputs/threebody/final/summary/rebuttal_figure_data.npz') as f:cache=dict(f)
    with np.load(ROOT/'paper/plot_sources/compare_wsindy_src/purple_curves.npz') as f:purple=dict(f)
    fig,axes=plt.subplots(1,4,figsize=(16,4.1));x=cache['x'];t=cache['times']
    styles=[('snap_mvnn','C0','MVNN'),('snap_kernel','C2','Mean-field WSINDy'),('snap_local','C3','Local PDE-WSINDy'),('snap_reference','C1','Reference')]
    for k,target in enumerate([0,.5,1]):
        ax=axes[k];i=int(np.argmin(abs(t-target)))
        for key,color,label in styles:ax.plot(x,cache[key][i],color=color,label=label)
        d=purple['d'+str(k)];ax.plot(d[:,0],d[:,1],color='C4',label='Online parameter estimation')
        ax.set_title('$t='+str(target)+'$');ax.set_xlabel('$x$');ax.set_xlim(-3.5,3)
    axes[0].set_ylabel(r'$\rho(x)$');axes[0].legend(frameon=False,fontsize=8)
    upper=max(ax.get_ylim()[1] for ax in axes[:3])
    for ax in axes[:3]:ax.set_ylim(0,upper)
    ax=axes[3];mv=cache['l2_mvnn']
    ax.plot(t,mv.mean(0),color='C0');ax.fill_between(t,mv.min(0),mv.max(0),color='C0',alpha=.25,lw=0)
    ax.plot(t,cache['l2_kernel'],color='C2');ax.plot(t,cache['l2_local'],color='C3')
    ax.plot(purple['err'][:,0],purple['err'][:,1],color='C4')
    ax.set_title('$L^2$ error');ax.set_xlabel('$t$');ax.set_ylabel('density $L^2$ error');ax.set_ylim(bottom=0)
    fig.tight_layout();save(fig,out,'figure_09_frozen_comparison')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',type=int,choices=[1,5,9],default=[1,5,9])
    parser.add_argument('--output-dir',type=Path,default=ROOT/'runs/figures')
    args=parser.parse_args();plt.rcParams.update({'font.family':'DejaVu Serif','font.size':11,'pdf.fonttype':42,'text.usetex':False})
    for number in args.figures:
        {1:learning_curve,5:timing,9:threebody}[number](args.output_dir)
        print('Redrew Figure',number,'from archived data in',args.output_dir)

if __name__=='__main__':main()
