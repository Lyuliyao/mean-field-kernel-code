#!/usr/bin/env python3
"""Validate tracked file hashes, representative extracts and frozen models."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    errors=[];checked=0
    file=ROOT/'provenance/repository_checksums.json'
    if not file.exists():raise SystemExit('Missing repository checksum manifest')
    data=json.loads(file.read_text())
    for rel,expected in data['files'].items():
        p=ROOT/rel
        if not p.is_file() or sha(p)!=expected:errors.append(rel)
        checked+=1
    for host in ['amd20','anvil']:
        base=ROOT/'data/representative'/host
        for record in json.loads((base/'manifest.json').read_text()):
            p=base/record['dest']
            if not p.is_file() or sha(p)!=record['sha256']:errors.append(str(p.relative_to(ROOT)))
            checked+=1
    marker=json.loads((ROOT/'outputs/threebody/FREEZE.json').read_text())
    for rel,expected in marker['selection_summary']['model_artifacts'].items():
        p=ROOT/'outputs/threebody'/rel
        if not p.is_file() or sha(p)!=expected:errors.append('frozen artifact '+rel)
        checked+=1
    base=ROOT/'outputs/threebody/data/final/confirmation'
    manifest=json.loads((base/'manifest.json').read_text())
    for row in manifest['triplets']:
        for m in row['measures'].values():
            p=base/m['path']
            if not p.is_file() or sha(p)!=m['sha256']:errors.append('confirmation '+m['path'])
            checked+=1
    if errors:
        print('Integrity failures:',*errors,sep='\n');return 1
    print('Verified',checked,'file/manifest checks, including frozen models and confirmation data.');return 0

if __name__=='__main__':sys.exit(main())
