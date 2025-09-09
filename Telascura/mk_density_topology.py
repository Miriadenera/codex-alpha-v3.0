#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Crea telascura\metrics\...\topology_series.csv a partire dallo snapshot
# Requisiti: numpy, scipy (ndimage)
import argparse, os, csv, math
import numpy as np
from numpy.fft import rfftn, irfftn
from scipy.ndimage import gaussian_filter, label

def ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def parse_sigmas(s):
    # es: "0.5,1,2,3,4,5"
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        out.append(float(tok))
    return out

def cic_deposit(pos, L, ngrid):
    """Deposita particelle su griglia 3D con CIC periodico."""
    N = pos.shape[0]
    g = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    x = (pos / L * ngrid) % ngrid
    i0 = np.floor(x).astype(np.int64)
    d  = x - i0
    wx = np.stack([1.0 - d[:,0], d[:,0]], axis=1)
    wy = np.stack([1.0 - d[:,1], d[:,1]], axis=1)
    wz = np.stack([1.0 - d[:,2], d[:,2]], axis=1)
    for dx in (0,1):
        ix = (i0[:,0] + dx) % ngrid
        wxv = wx[:,dx]
        for dy in (0,1):
            iy = (i0[:,1] + dy) % ngrid
            wyv = wy[:,dy]
            wxy = wxv * wyv
            for dz in (0,1):
                iz = (i0[:,2] + dz) % ngrid
                w = (wxy * wz[:,dz]).astype(np.float32)
                g[ix, iy, iz] += w
    return g

def largest_component_filamentarity(mask, vox2phys=1.0):
    """Ritorna la filamentarity del max cluster in 'mask' (True=interno)."""
    if not np.any(mask):
        return np.nan
    lab, nlab = label(mask)
    if nlab == 0:
        return np.nan
    # più grande per voxel count
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    lid = np.argmax(counts)
    idx = np.where(lab == lid)
    # coordinate (in unità di voxel o fisiche)
    X = np.stack(idx, axis=1).astype(np.float64) * vox2phys
    # centroide e covarianza
    c = X.mean(axis=0, keepdims=True)
    Y = X - c
    C = (Y.T @ Y) / max(1, Y.shape[0]-1)
    # assi principali
    w = np.sort(np.sqrt(np.maximum(np.linalg.eigvalsh(C), 0.0)))[::-1]  # a>=b>=c
    a, b = w[0], w[1] if w.size>1 else 0.0
    if a <= 0: 
        return 0.0
    # semplice definizione: F = (a-b)/a in [0,1]
    return float((a - b) / a)

def phase_shuffle_null(delta, rng):
    """Randomizza le fasi (preserva ampiezza) e torna un campo reale."""
    # FFT reale: griglia -> half-complex (kx,ky,kz >= 0)
    F = rfftn(delta)
    amp = np.abs(F)
    # fasi casuali uniformi
    phi = rng.uniform(0.0, 2*np.pi, size=F.shape)
    Fnull = amp * np.exp(1j*phi)
    # preserva la media (k=0)
    Fnull.flat[0] = F.flat[0]
    out = np.real(irfftn(Fnull, s=delta.shape))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)                 # es: telascura\runs\hi256_seedA\snap_final.npz
    ap.add_argument("--ngrid", type=int, default=256)
    ap.add_argument("--sigmas", required=True)               # es: "0.5,1,2,3,4,5" (in voxel)
    ap.add_argument("--nnull", type=int, default=64)         # null surrogates
    ap.add_argument("--seed",  type=int, default=0)
    ap.add_argument("--out",   required=True)                # es: telascura\metrics\hi256_z000\topology_series.csv
    args = ap.parse_args()

    sigmas = parse_sigmas(args.sigmas)
    if len(sigmas)==0:
        raise SystemExit("Nessuna sigma valida in --sigmas")

    # ---- carica snapshot ----
    d = np.load(args.snap)
    if "pos" not in d:
        raise SystemExit("Snapshot NPZ senza 'pos'. Chiavi viste: %s" % list(d.keys()))
    pos = np.asarray(d["pos"], dtype=np.float64)
    # lato box
    if "L" in d:
        L = float(d["L"])
    else:
        # fallback: bounding box
        L = float(pos.max() - pos.min())
        if not np.isfinite(L) or L<=0:
            L = 1.0
    # deposita densità
    rho = cic_deposit(pos, L, args.ngrid)
    rho /= rho.mean()
    delta = rho - 1.0

    rng = np.random.default_rng(args.seed)
    vox2phys = L / args.ngrid  # per informazione (non usato nella tabella)

    # prealloca null
    null_lo = []
    null_hi = []
    meas_F  = []

    for s in sigmas:
        # smoothing Gauss su delta
        d_s = gaussian_filter(delta, sigma=s, mode="wrap")
        # threshold a >0 (nu=0)
        mask = d_s > 0.0
        Fm = largest_component_filamentarity(mask, vox2phys)
        meas_F.append(Fm)

        # null surrogates (phase shuffle preservando ampiezza)
        Fnull = []
        for _ in range(args.nnull):
            dn = phase_shuffle_null(delta, rng)
            dn_s = gaussian_filter(dn, sigma=s, mode="wrap")
            Fnull.append(largest_component_filamentarity(dn_s > 0.0, vox2phys))
        Fnull = np.asarray(Fnull, dtype=np.float64)
        null_lo.append(np.nanpercentile(Fnull, 2.5))
        null_hi.append(np.nanpercentile(Fnull, 97.5))

    # scrivi CSV
    ensure_dir(args.out)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "filamentarity", "fil_lo", "fil_hi", "used"])
        for s, fm, lo, hi in zip(sigmas, meas_F, null_lo, null_hi):
            used = 1  # per ora tutti i bin sono marcati indipendenti/validi
            w.writerow([f"{s:.6g}", f"{fm:.6g}", f"{lo:.6g}", f"{hi:.6g}", used])

    print("[WRITE]", args.out)
    print("Esempio prime righe:")
    with open(args.out, newline="") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i>5: break

if __name__ == "__main__":
    main()
