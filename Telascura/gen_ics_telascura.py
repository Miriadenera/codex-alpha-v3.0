#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_ics_telascura.py — genera ICs gaussiane per ∇K su griglia N, e campiona su npart_side^3 particelle.

Input:  YAML cfg (telascura.yaml) con: box_size_hMpc, np_side, npart_side, seed, z_ini, pk_file
Output: .npz con posizioni e ∇K per particella + metadati di riproducibilità
"""

import argparse, os, sys, yaml
import numpy as np
import pandas as pd
from numpy.fft import rfftn, irfftn, fftfreq

def log(msg): print(msg, flush=True)

def read_cfg(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_pk(csv_path):
    df = pd.read_csv(csv_path)
    k  = df.iloc[:,0].to_numpy(dtype=np.float64)
    pk = df.iloc[:,1].to_numpy(dtype=np.float64)
    # assicurati crescenti e >0
    m = np.isfinite(k) & np.isfinite(pk) & (k>0) & (pk>0)
    k, pk = k[m], pk[m]
    s = np.argsort(k)
    return k[s], pk[s]

def pk_interp(kgrid, k_tab, pk_tab):
    # log-log robusto
    kmin, kmax = k_tab.min(), k_tab.max()
    kk = np.clip(kgrid, kmin, kmax)
    return np.exp(np.interp(np.log(kk), np.log(k_tab), np.log(pk_tab)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', default=None)
    args = ap.parse_args()

    cfg = read_cfg(args.cfg)
    L      = float(cfg['box_size_hMpc'])
    N      = int(cfg['np_side'])
    Np     = int(cfg['npart_side'])
    z_ini  = float(cfg.get('z_ini', 1000.0))
    seed   = int(cfg.get('seed', 424242))
    pkfile = cfg['pk_file']

    # memo: con N=512 la rfft occupa parecchia RAM. Inizia con N=256 se serve.
    if N >= 512:
        log(f"[WARN] np_side={N} può richiedere molta RAM. Se vedi OOM, prova N=256 in cfg.")

    k_tab, pk_tab = load_pk(pkfile)
    log(f"[INFO] Box L={L:.1f} h^-1 Mpc, Ngrid={N}, Npart_side={Np}, seed={seed}")
    log(f"[INFO] Pk tab: {len(k_tab)} punti  k∈[{k_tab.min():.4f},{k_tab.max():.4f}] h Mpc^-1  (file={pkfile})")

    rng = np.random.default_rng(seed)

    # --- spazio-k con rfft lungo z per dimezzare memoria
    dx = L / N
    kx = 2*np.pi*fftfreq(N, d=dx)
    ky = 2*np.pi*fftfreq(N, d=dx)
    kz = 2*np.pi*np.fft.rfftfreq(N, d=dx)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    Kmag = np.sqrt(KX*KX + KY*KY + KZ*KZ).astype(np.float32)

    # spettro su griglia k
    Pk = pk_interp(Kmag, k_tab, pk_tab).astype(np.float32)
    Pk[0,0,0] = 0.0

    # campo gaussiano φ(k) con var ~ P(k)/2 (rfft: reale+immaginaria)
    amp = np.sqrt(Pk * 0.5, dtype=np.float32)
    noise_real = rng.normal(size=Kmag.shape).astype(np.float32)
    noise_imag = rng.normal(size=Kmag.shape).astype(np.float32)
    phi_k = (noise_real + 1j*noise_imag).astype(np.complex64) * amp.astype(np.complex64)

    # ∇K(k) = i k φ(k)
    iKX = (1j * KX).astype(np.complex64)
    iKY = (1j * KY).astype(np.complex64)
    iKZ = (1j * KZ).astype(np.complex64)
    gx_k = iKX * phi_k
    gy_k = iKY * phi_k
    gz_k = iKZ * phi_k

    # ritorno in real-space (float32)
    gx = irfftn(gx_k, s=(N,N,N)).astype(np.float32)
    gy = irfftn(gy_k, s=(N,N,N)).astype(np.float32)
    gz = irfftn(gz_k, s=(N,N,N)).astype(np.float32)

    # --- posizioni particelle su griglia regolare
    xs = (np.arange(Np, dtype=np.float32) + 0.5) * (L / Np)  # centro-cell
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # float32

    # nearest-neighbor sampling della ∇K sulla griglia N
    # (indice wrap periodico)
    scale = (N / L)
    ix = np.mod((pos[:,0] * scale).astype(np.int64), N)
    iy = np.mod((pos[:,1] * scale).astype(np.int64), N)
    iz = np.mod((pos[:,2] * scale).astype(np.int64), N)
    gpart = np.stack([gx[ix,iy,iz], gy[ix,iy,iz], gz[ix,iy,iz]], axis=1).astype(np.float32)

    meta = dict(
        L_hMpc=L, np_side=N, npart_side=Np, z_ini=z_ini, seed=seed,
        pk_file=str(pkfile),
        kmin=float(k_tab.min()), kmax=float(k_tab.max()),
        code="gen_ics_telascura.py"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, pos=pos, gradK=gpart, meta=meta)
    log(f"[WRITE] {args.out}  (particles={pos.shape[0]:,})")

    if args.log:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        with open(args.log, "w", encoding="utf-8") as f:
            for k,v in meta.items():
                f.write(f"{k}: {v}\n")
            f.write(f"particles: {pos.shape[0]}\n")

if __name__ == "__main__":
    main()
