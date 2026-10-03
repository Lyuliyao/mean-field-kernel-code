#!/usr/bin/env python3
"""Create an independent source-only run directory, preserving archived results."""
import argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);a=p.parse_args()
    dest=a.destination.resolve()
    if dest.exists():raise FileExistsError('Destination must be new: '+str(dest))
    dest.mkdir(parents=True)
    for name in ['src','scripts','tests','experiments']:
        shutil.copytree(ROOT/name,dest/name,ignore=shutil.ignore_patterns('__pycache__','model_save_path','.pytest_cache'))
    for name in ['protocol.yaml','protocol_mtcurve.yaml','pyproject.toml','environment.yml','requirements-reference-cpu.txt','requirements-macos-intel.txt','.gitignore']:
        shutil.copy2(ROOT/name,dest/name)
    print('Created independent run sources:',dest)
    print('Archived model/results directories were not copied. Initialize and commit a new Git repository here before the final freeze workflow.')

if __name__=='__main__':main()
