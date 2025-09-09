#!/usr/bin/env python3
import argparse, os, yaml, numpy as np

def read_cfg(path):
    import yaml
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--override', nargs='*', default=[])
    ap.add_argument('--log', default=None)
    args = ap.parse_args()

    cfg = read_cfg(args.cfg)
    for ov in args.override:
        k,v = ov.split('=')
        # cast int/float se possibile
        if v.isdigit(): v = int(v)
        else:
            try: v = float(v)
            except: pass
        cfg[k]=v

    os.makedirs(args.out, exist_ok=True)
    ic = np.load('ics/ic_main.npz')
    L = float(ic['L']); pos = ic['pos'].astype(np.float64)
    gradK = ic['gradK'].astype(np.float64)
    z = float(ic['z_ini']); zfin = float(cfg['z_fin'])
    cfl = float(cfg.get('adaptive_cfl',0.3))
    model = cfg.get('force_model','gradK')

    vel = np.zeros_like(pos)
    # normalizza gradK per time-step stabile (toy)
    gnorm = np.percentile(np.linalg.norm(gradK,axis=1), 95)
    gnorm = max(gnorm, 1e-6)
    dt0 = cfl * (L / (gnorm*100))  # scala grossolana

    steps = 200  # toy steps → in produzione scegli in base a Δz
    for s in range(steps):
        # forza
        F = gradK.copy()
        if model == 'grav':
            pass
        elif model == 'gradK_plus_grav':
            pass  # hook per usare una Poisson semplice

        # passo adattivo sulla 95a
        g95 = np.percentile(np.linalg.norm(F,axis=1),95)
        dt = dt0 * (gnorm / max(g95,1e-6))

        vel += F * dt
        pos += vel * dt
        pos %= L  # periodicità
        z = max(zfin, z - (ic['z_ini']-zfin)/steps)

    out = os.path.join(args.out,'snap_z0.npz')
    np.savez(out, L=L, pos=pos.astype(np.float32))
    if args.log:
        with open(args.log,'w') as f:
            f.write(f"[WRITE] {out}\n")

if __name__=='__main__':
    main()
