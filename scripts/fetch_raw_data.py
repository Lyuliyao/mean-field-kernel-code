#!/usr/bin/env python3
"""List or retrieve one cataloged original file, checking size and SHA-256."""
import argparse,hashlib,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list',action='store_true');p.add_argument('--contains',default='')
    p.add_argument('--index',type=int);p.add_argument('--fetch',action='store_true')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/raw/retrieved')
    a=p.parse_args();rows=json.loads((ROOT/'provenance/raw_data_catalog.json').read_text())
    if a.list:
        for i,r in enumerate(rows):
            if a.contains in r['path']:print(i,r['host'],r['bytes'],r['path'])
        return
    if not a.fetch or a.index is None:p.error('Use --list, or --index INDEX --fetch')
    if not 0<=a.index<len(rows):p.error('index out of range')
    r=rows[a.index]
    if r['host'] not in ['amd20','anvil']:p.error('unknown host')
    path=r['path'];quoted=shlex.quote(path);ssh=['ssh','-o','BatchMode=yes',r['host']]
    expected=subprocess.check_output(ssh+['sha256sum -- '+quoted],text=True).split()[0]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    dest=a.output_dir/(str(a.index)+'_'+Path(path).name);part=dest.with_name(dest.name+'.part')
    if dest.exists() or part.exists():raise FileExistsError(dest)
    h=hashlib.sha256();size=0
    with open(part,'xb') as out:
        process=subprocess.Popen(ssh+['cat -- '+quoted],stdout=subprocess.PIPE)
        for block in iter(lambda:process.stdout.read(1024*1024),b''):
            out.write(block);h.update(block);size+=len(block)
        if process.wait()!=0:raise RuntimeError('SSH transfer failed; partial file retained')
    if size!=r['bytes'] or h.hexdigest()!=expected:raise RuntimeError('Size or SHA-256 mismatch; partial file retained')
    part.replace(dest)
    receipt=dict(r,sha256=expected,downloaded_bytes=size,local_path=str(dest))
    dest.with_name(dest.name+'.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Verified and saved',dest)

if __name__=='__main__':main()
