# telascura/bin/mk_information_coherence.py
import numpy as np
import csv, os
from scipy.stats import entropy
from numpy.fft import fftn, ifftn, fftfreq

def information_coherence(field, scale):
    """Misura coerenza informazionale invece di filamentarity"""
    # Smooth field
    kx = fftfreq(field.shape[0]) * 2*np.pi
    ky = fftfreq(field.shape[1]) * 2*np.pi
    kz = fftfreq(field.shape[2]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX*KX + KY*KY + KZ*KZ
    
    F = fftn(field)
    F_smooth = F * np.exp(-0.5 * (scale**2) * K2)
    field_smooth = np.real(ifftn(F_smooth))
    
    # Quantizza in bins per calcolare entropia
    bins = 50
    hist, _ = np.histogram(field_smooth, bins=bins, density=True)
    hist = hist + 1e-12  # Evita log(0)
    
    # Entropia di Shannon
    H = entropy(hist)
    
    # Coerenza = deviazione dall'entropia massima
    H_max = np.log(bins)  # Entropia massima (distribuzione uniforme)
    coherence = 1.0 - (H / H_max)
    
    return float(coherence)

def gradient_alignment(field, scale):
    """Misura allineamento dei gradienti - signature Telascura"""
    # Smooth
    kx = fftfreq(field.shape[0]) * 2*np.pi
    ky = fftfreq(field.shape[1]) * 2*np.pi
    kz = fftfreq(field.shape[2]) * 2*np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX*KX + KY*KY + KZ*KZ
    
    F = fftn(field)
    F_smooth = F * np.exp(-0.5 * (scale**2) * K2)
    field_smooth = np.real(ifftn(F_smooth))
    
    # Calcola gradienti
    grad_x = np.gradient(field_smooth, axis=0)
    grad_y = np.gradient(field_smooth, axis=1)
    grad_z = np.gradient(field_smooth, axis=2)
    
    # Magnitudine gradiente
    grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    # Allineamento = varianza delle direzioni del gradiente
    # Se i gradienti sono allineati (Telascura), varianza bassa
    mask = grad_mag > np.percentile(grad_mag, 75)
    if mask.sum() < 100:
        return 0.0
    
    # Direzioni normalizzate
    gx_norm = grad_x[mask] / (grad_mag[mask] + 1e-12)
    gy_norm = grad_y[mask] / (grad_mag[mask] + 1e-12)
    gz_norm = grad_z[mask] / (grad_mag[mask] + 1e-12)
    
    # Varianza delle direzioni (bassa = allineamento alto)
    var_x = np.var(gx_norm)
    var_y = np.var(gy_norm)
    var_z = np.var(gz_norm)
    
    alignment = 1.0 / (1.0 + var_x + var_y + var_z)
    return float(alignment)

# Test rapido
def read_snap(path):
    d = np.load(path, allow_pickle=True)
    pos = d["pos"]
    L = float(d["L"]) if "L" in d else float(pos.max() - pos.min())
    return L, pos

def cic_histogram(pos, L, ngrid):
    edges = np.linspace(0.0, L, ngrid+1, endpoint=True)
    H, _ = np.histogramdd(pos, bins=(edges, edges, edges))
    return H

# Main
L, pos = read_snap("telascura/runs/hi256_seedA/snap_final.npz")
counts = cic_histogram(pos, L, 128)
delta = (counts - counts.mean()) / counts.mean()

print("METRICHE INFORMAZIONALI TELASCURA:")
results = []

for sigma in [0.5, 1, 2, 3, 4, 5]:
    coherence = information_coherence(delta, sigma)
    alignment = gradient_alignment(delta, sigma)
    
    print(f"Sigma {sigma}: coherence={coherence:.4f}, alignment={alignment:.4f}")
    results.append([sigma, coherence, alignment])

# Salva
os.makedirs("telascura/metrics/hi256_z000", exist_ok=True)
with open("telascura/metrics/hi256_z000/information_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sigma", "coherence", "alignment"])
    writer.writerows(results)

print("Salvato: information_metrics.csv")
