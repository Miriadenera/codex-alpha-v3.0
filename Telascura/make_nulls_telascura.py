#!/usr/bin/env python3
import argparse, os, yaml, numpy as np

def read_cfg(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def rotate_z(pos, L, deg):
    th = np.deg2rad(deg)
    x,y,z = pos[:,0]-L/2, pos[:,1]-L/2, pos[:,2]
    xr =  x*np.cos(th) - y*np.sin(th)
    yr =  x*np.sin(th) + y*np.cos(th)
    return np.stack([xr+L/2, yr+L/2, z], axis=1) % L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', default=None)
    args = ap.parse_args()

    cfg = read_cfg(args.cfg)
    os.makedirs(args.out, exist_ok=True)
    base = np.load(args.base)
    L = float(base['L']); pos = base['pos']

    for nul in cfg.get('nulls',[]):
        name = nul['name']; op = nul['op']
        p = pos.copy()
        if op == 'rotate_box':
            p = rotate_z(p, L, float(nul.get('deg',90)))
        elif op == 'sign_flip':
            p = (2*(p<L/2).astype(float)-1.0) * p  % L  # giocattolo
        # altri operatori concreti su snapshot (phase scramble/rescale richiedono campo; qui snapshot-level)
        out = os.path.join(args.out, f"{name}.npz")
        np.savez(out, L=L, pos=p.astype(np.float32))
    if args.log:
        with open(args.log,'w') as f:
            f.write(f"[WRITE] nulls in {args.out}\n")

if __name__=='__main__':
    main()
