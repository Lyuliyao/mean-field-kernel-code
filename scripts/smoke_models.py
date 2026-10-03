#!/usr/bin/env python3
"""Check recovered checkpoint loading, finite outputs and particle equivariance."""
import importlib.util
from pathlib import Path
import numpy as np
import jax
from jax import config
config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    rng=np.random.default_rng(42)
    cases=[('aggregation_2d','TRAINING',2,'001000'),('motsch_tadmor','TRAINING2',1,'000400'),('stochastic_motsch_tadmor','TRAINING4',1,'009099')]
    for name,folder,dim,ck in cases:
        base=ROOT/'experiments'/name/folder;model=load_module(base/'model.py',name)
        with np.load(base/'model_save_path'/f'jax_ckpt_{ck}.npz',allow_pickle=True) as f:params=f['params'].item()
        _,apply=model.create_model([12,24,48],[128,128,128,dim],dim)
        x=rng.normal(size=(32,dim));perm=rng.permutation(len(x));a=np.asarray(apply(params,x));b=np.asarray(apply(params,x[perm]))
        assert a.shape==x.shape and np.isfinite(a).all(),name
        np.testing.assert_allclose(b,a[perm],atol=1e-10,rtol=1e-10)
        print(name,'checkpoint and equivariance passed')
    base=ROOT/'experiments/hierarchical';model=load_module(base/'SIM3_4/model.py','hierarchical')
    with np.load(base/'TRAINING3/model_save_path/jax_ckpt_000100.npz',allow_pickle=True) as f:params=f['params'].item()
    _,apply=model.create_model([12,24,48],[128,128,128,1],1)
    states=[rng.normal(size=(n,1)) for n in [32,12,3]];perms=[rng.permutation(len(x)) for x in states]
    a=apply(params,*states);b=apply(params,*[x[p] for x,p in zip(states,perms)])
    for original,permuted,p in zip(a,b,perms):
        assert np.isfinite(np.asarray(original)).all()
        np.testing.assert_allclose(np.asarray(permuted),np.asarray(original)[p],atol=1e-10,rtol=1e-10)
    print('hierarchical checkpoint and within-group equivariance passed')

if __name__=='__main__':main()
