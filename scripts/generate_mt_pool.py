#!/usr/bin/env python3
"""Regenerate deterministic training slices using the recovered original simulator.

Full N=16000 generation is expensive; run under a compute allocation.
The resulting 201-frame slices do not have the hash of longer original files.
"""
import argparse,sys,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from mtcurve.simulate import simulate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',default='1:100',help='inclusive start:end')
    p.add_argument('--particles',type=int,default=16000)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/raw/mt_training_pool')
    a=p.parse_args();start,end=map(int,a.seeds.split(':'))
    if start<1 or end<start or a.particles<2 or a.steps<1:p.error('invalid seed or size range')
    a.output_dir.mkdir(parents=True,exist_ok=True);records=[]
    for seed in range(start,end+1):
        path=a.output_dir/f'opinion_traj_{seed}.npy'
        if path.exists():raise FileExistsError('Refusing to overwrite '+str(path))
        data,ic=simulate(seed,a.steps,n=a.particles);np.save(path,data)
        records.append({'seed':seed,'path':path.name,'shape':list(data.shape),'initial_condition':ic,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        print('Generated seed',seed,flush=True)
    (a.output_dir/'regeneration_manifest.json').write_text(json.dumps({'source':'src/mtcurve/simulate.py','slice_only':True,'records':records},indent=2)+'\n')

if __name__=='__main__':main()
