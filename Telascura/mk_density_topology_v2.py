#!/usr/bin/env python3
import argparse, os, sys, math, csv, time
import numpy as np
from numpy.fft import rfftn, irfftn, fftn, ifftn, fftfreq

try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None

def read_snap(path):
    d = np.load(path, allow_pickle=True)
    pos = d["pos"]
    L = float(d["L"]) if "L" in d else float(pos.max() - pos.min())
    return L, pos

def cic_histogram(pos, L, ngrid):
    # NGP histogram robusto
    edges = np.linspace(0.0, L, ngrid+1, endpoint=True)
    H, _ = np.histogramdd(pos, bins=(edges, edges, edges))
    return H

def gaussian_smooth_fft(field, sigma_vox):
    # Smooth in Fourier space: exp(-0.5 k^2 sigma^2)
    kx = fftfreq(field.shape[0]) * 2*np.pi
    ky = fftfreq(field.shape[1]) * 2*np.pi  
    kz = fftfreq(field.shape[2]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX*KX + KY*KY + KZ*KZ
    F = fftn(field)
    F *= np.exp(-0.5 * (sigma_vox**2) * K2)
    return np.real(ifftn(F))

def largest_filamentarity(mask):
    # Calcola filamentarity del componente più grande
    if cc_label is None:
        return 0.0
    
    labeled, ncomp = cc_label(mask.astype(np.uint8))
    if ncomp == 0:
        return 0.0
    
    # Trova componente più grande
    sizes = [(labeled == i).sum() for i in range(1, ncomp+1)]
    largest_label = np.argmax(sizes) + 1
    largest_mask = (labeled == largest_label)
    
    # Calcola matrice covarianza
    pts = np.argwhere(largest_mask)
    if pts.shape[0] < 10:
        return 0.0
    
    C = np.cov(pts.T.astype(float))
    w = np.linalg.eigvals(C)
    w = np.sort(np.maximum(w, 1e-20))
    
    # Filamentarity = (λ1 - λ2) / (λ1 + λ2)
    if w[2] + w[1] < 1e-12:
        return 0.0
    return float((w[2] - w[1]) / (w[2] + w[1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--ngrid", type=int, default=256)
    ap.add_argument("--sigmas", type=str, default="0.5,1,2,3,4,5")
    ap.add_argument("--nnull", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    
    # Leggi snapshot
    L, pos = read_snap(args.snap)
    counts = cic_histogram(pos, L, args.ngrid)
    mean_counts = counts.mean()
    delta = (counts - mean_counts) / mean_counts
    
    # FFT per smoothing efficiente
    F = fftn(delta)
    amp = np.abs(F)
    dc0 = F.flat[0]
    
    # Prepara filtri Gaussiani
    kx = fftfreq(delta.shape[0]) * 2*np.pi
    ky = fftfreq(delta.shape[1]) * 2*np.pi
    kz = fftfreq(delta.shape[2]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX*KX + KY*KY + KZ*KZ
    
    filters = [np.exp(-0.5 * (s**2) * K2) for s in sigmas]
    
    # Controlla append
    done = set()
    if args.append and os.path.exists(args.out):
        with open(args.out, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 1:
                    try:
                        done.add(float(parts[0]))
                    except:
                        pass
    
    # Apri file output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not args.append or not os.path.exists(args.out):
        w = open(args.out, "w", newline="")
        cw = csv.writer(w)
        cw.writerow(["sigma", "filamentarity", "null_lo", "null_hi", "used"])
    else:
        w = open(args.out, "a", newline="")
        cw = csv.writer(w)

    rng = np.random.default_rng(args.seed)

    # Loop sui sigma
    for s, G in zip(sigmas, filters):
        if args.append and (s in done):
            print(f"[skip] sigma={s} già presente nel CSV.")
            continue

        # Misurata: F * G -> inv FFT -> mask>0
        Fm = F * G
        dm = np.real(ifftn(Fm))
        measF = largest_filamentarity(dm > 0.0)

        # Null: amp * e^{i phi} * G
        q = []
        for t in range(args.nnull):
            if t % 4 == 0:
                print(f"[{s}] null {t}/{args.nnull} ...", flush=True)
            phi = rng.uniform(0.0, 2*np.pi, size=F.shape)
            Fn = amp * np.exp(1j*phi)
            Fn.flat[0] = dc0
            dn = np.real(ifftn(Fn*G))
            q.append(largest_filamentarity(dn > 0.0))
        
        q = np.asarray(q, dtype=np.float64)
        lo = np.nanpercentile(q, 2.5)
        hi = np.nanpercentile(q, 97.5)
        
        cw.writerow([f"{s:.6g}", f"{measF:.6g}", f"{lo:.6g}", f"{hi:.6g}", 1])
        w.flush()
        print(f"[write] sigma={s}  F={measF:.3f}  null95%=[{lo:.3f},{hi:.3f}]")

    w.close()
    print("[DONE]", args.out, f"| elapsed {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()