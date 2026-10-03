#!/usr/bin/env python3
"""Plot explicitly indexed representative extracts, not full pooled paper curves."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter,gaussian_filter1d

ROOT=Path(__file__).resolve().parents[1]

def select_pairs(case,scenario,quantity):
    if case=='aggregation_2d':
        n={'ring':1,'double_ring':2,'disk':3,'binary':4}[scenario]
        base=ROOT/'data/representative/amd20/CASE4'
        return 'amd20',[(base/f'case{n}_reference.npy',base/f'case{n}_mvnn.npy')]
    if case in ['motsch_tadmor','stochastic_motsch_tadmor']:
        n=int(scenario);folder='CASE5' if case=='motsch_tadmor' else 'CASE5_stochastic'
        base=ROOT/'data/representative/amd20'/folder
        return 'amd20',[(base/f'seed{n}_reference.npy',base/f'seed{n}_mvnn.npy')]
    if case=='hierarchical':
        n=int(scenario);base=ROOT/'data/representative/amd20/CASE7'
        return 'amd20',[(base/f'seed{n}_reference_{sp}.npy',base/f'seed{n}_mvnn_{sp}.npy') for sp in ['workers','managers','ceos']]
    folder='case4' if case=='second_order_ar' else 'case4_cs'
    lookup={'ring':'','double_ring':'_2','disk':'_3','binary':'_4'} if folder=='case4' else {'gaussian':'','double_gaussian':'_2'}
    suffix=lookup[scenario];filename=('v_' if quantity=='velocity' else '')+'result_second_order.npy'
    base=ROOT/'data/representative/anvil'/folder
    return 'anvil',[(base/f'test_data_second_order{suffix}'/filename,base/f'sim_second_order{suffix}'/filename)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=['aggregation_2d','motsch_tadmor','stochastic_motsch_tadmor','hierarchical','second_order_ar','cucker_smale'])
    p.add_argument('--scenario',required=True,help='ring/double_ring/disk/binary; gaussian/double_gaussian; or seed 1/2/3')
    p.add_argument('--quantity',choices=['position','velocity'],default='position')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/snapshots')
    a=p.parse_args()
    try:host,pairs=select_pairs(a.case,a.scenario,a.quantity)
    except (KeyError,ValueError):p.error('scenario does not apply to this case')
    base=ROOT/'data/representative'/host
    records={r['dest']:r for r in json.loads((base/'manifest.json').read_text())}
    if len(pairs)>1:
        fig,axes=plt.subplots(len(pairs),3,figsize=(12,3.2*len(pairs)),squeeze=False)
        for row,(ref,pred) in enumerate(pairs):
            data=[np.load(ref),np.load(pred)];limits=(min(d.min() for d in data)-.5,max(d.max() for d in data)+.5)
            for k,ax in enumerate(axes[row]):
                for d,color,label in zip(data,['C1','C0'],['Reference','MVNN']):
                    y,e=np.histogram(d[k].ravel(),bins=128,range=limits,density=True)
                    ax.plot((e[1:]+e[:-1])/2,gaussian_filter1d(y,1),color=color,label=label)
                ax.set_title('source frame '+str(records[str(ref.relative_to(base))]['selected_frames'][k]));ax.set_ylabel(ref.stem.split('_')[-1]);ax.set_xlabel('state')
        axes[0,0].legend(frameon=False)
    else:
        ref,pred=pairs[0];arrays=[np.load(ref),np.load(pred)]
        dim=arrays[0].shape[-1] if arrays[0].ndim==3 else 1
        if dim==1:
            fig,axes=plt.subplots(1,3,figsize=(12,3.7))
            limits=(min(d.min() for d in arrays)-.3,max(d.max() for d in arrays)+.3)
            for k,ax in enumerate(axes):
                for d,color,label in zip(arrays,['C1','C0'],['Reference','MVNN']):
                    y,e=np.histogram(d[k].ravel(),bins=128,range=limits,density=True)
                    ax.plot((e[1:]+e[:-1])/2,gaussian_filter1d(y,1),color=color,label=label)
                ax.set_title('source frame '+str(records[str(ref.relative_to(base))]['selected_frames'][k]));ax.set_xlabel('state')
            axes[0].set_ylabel('smoothed histogram density');axes[0].legend(frameon=False)
        else:
            fig,axes=plt.subplots(2,3,figsize=(12,7))
            radius=max(np.abs(d).max() for d in arrays)*1.02
            densities=[]
            for d in arrays:
                densities.append([gaussian_filter(np.histogram2d(frame[:,0],frame[:,1],bins=80,range=[[-radius,radius]]*2,density=True)[0],1.2) for frame in d])
            top=max(h.max() for row in densities for h in row)
            for row,(d,path) in enumerate(zip(densities,[ref,pred])):
                for k,ax in enumerate(axes[row]):
                    im=ax.imshow(d[k].T,origin='lower',extent=[-radius,radius]*2,cmap='viridis',vmin=0,vmax=top,aspect='equal')
                    if row==0:ax.set_title('source frame '+str(records[str(path.relative_to(base))]['selected_frames'][k]))
                    if k==0:ax.set_ylabel('Reference' if row==0 else 'MVNN')
            fig.subplots_adjust(right=.9,wspace=.08,hspace=.08);cax=fig.add_axes([.92,.14,.015,.7]);fig.colorbar(im,cax=cax)
    fig.suptitle(a.case.replace('_',' ')+' / '+a.scenario+' — representative extract',fontsize=13)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    stem=a.case+'_'+a.scenario+'_'+a.quantity
    for ext in ['png','pdf']:fig.savefig(a.output_dir/(stem+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig);print('Saved representative snapshot plot:',a.output_dir/stem)

if __name__=='__main__':main()
