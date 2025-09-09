# telascura/bin/mk_topology_final.py
import numpy as np
import csv, os, argparse
from numpy.fft import fftn, ifftn, fftfreq

def read_snap(path):
    d = np.load(path, allow_pickle=True)
    pos = d["pos"]
    L = float(d["L"]) if "L" in d else float(pos.max() - pos.min())
    return L, pos

def cic_histogram(pos, L, ngrid):
    edges = np.linspace(0.0, L, ngrid+1, endpoint=True)
    H, _ = np.histogramdd(pos, bins=(edges, edges, edges))
    return H

def calculate_anisotropy(field, percentile=80):
    """Calcola anisotropia usando percentili - ROBUSTO"""
    thresh = np.percentile(field, percentile)
    mask = field > thresh
    
    coords = np.argwhere(mask).astype(float)
    if coords.shape[0] < 50:
        return 0.0
    
    # Centro di massa
    cm = coords.mean(axis=0)
    coords_centered = coords - cm
    
    # Matrice covarianza
    try:
        cov_matrix = np.cov(coords_centered.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # Decrescente
        
        # Anisotropia = (λ1 - λ3) / (λ1 + λ2 + λ3)
        total = eigenvals.sum()
        if total < 1e-12:
            return 0.0
        
        anisotropy = (eigenvals[0] - eigenvals[2]) / total
        return float(np.clip(anisotropy, 0, 1))
    except:
        return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--ngrid", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    
    # Leggi dati
    L, pos = read_snap(args.snap)
    counts = cic_histogram(pos, L, args.ngrid)
    delta = (counts - counts.mean()) / counts.mean()
    
    print(f"Campo caricato: {delta.shape}, std={delta.std():.3f}")
    
    # Prepara FFT
    kx = fftfreq(delta.shape[0]) * 2*np.pi
    ky = fftfreq(delta.shape[1]) * 2*np.pi
    kz = fftfreq(delta.shape[2]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX*KX + KY*KY + KZ*KZ
    F = fftn(delta)
    
    results = []
    sigmas = [0.5, 1, 2, 3, 4, 5]
    
    for sigma in sigmas:
        # Smooth
        F_smooth = F * np.exp(-0.5 * (sigma**2) * K2)
        delta_smooth = np.real(ifftn(F_smooth))
        
        # Calcola anisotropia
        aniso = calculate_anisotropy(delta_smooth, percentile=80)
        
        # Null semplificato (shuffle)
        nulls = []
        for _ in range(16):  # Meno null per velocità
            delta_shuffled = np.random.permutation(delta_smooth.ravel()).reshape(delta_smooth.shape)
            null_aniso = calculate_anisotropy(delta_shuffled, percentile=80)
            nulls.append(null_aniso)
        
        null_lo = np.percentile(nulls, 2.5)
        null_hi = np.percentile(nulls, 97.5)
        
        results.append([sigma, aniso, null_lo, null_hi, 1])
        print(f"Sigma {sigma}: aniso={aniso:.4f}, null=[{null_lo:.4f}, {null_hi:.4f}]")
    
    # Salva
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma", "anisotropy", "null_lo", "null_hi", "used"])
        writer.writerows(results)
    
    print(f"Salvato: {args.out}")

if __name__ == "__main__":
    main()
