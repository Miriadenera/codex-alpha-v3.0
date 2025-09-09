# telascura/bin/debug_density_field.py
import numpy as np

def read_snap(path):
    d = np.load(path, allow_pickle=True)
    pos = d["pos"]
    L = float(d["L"]) if "L" in d else float(pos.max() - pos.min())
    return L, pos

def cic_histogram(pos, L, ngrid):
    edges = np.linspace(0.0, L, ngrid+1, endpoint=True)
    H, _ = np.histogramdd(pos, bins=(edges, edges, edges))
    return H

# Test rapido
L, pos = read_snap("telascura/runs/hi256_seedA/snap_final.npz")
counts = cic_histogram(pos, L, 128)
delta = (counts - counts.mean()) / counts.mean()

print(f"L = {L}")
print(f"N particles = {pos.shape[0]}")
print(f"Delta stats: min={delta.min():.3f}, max={delta.max():.3f}, std={delta.std():.3f}")
print(f"Positive fraction: {(delta > 0).mean():.3f}")
print(f"Above +1σ: {(delta > delta.std()).mean():.3f}")
print(f"Above +2σ: {(delta > 2*delta.std()).mean():.3f}")

# Test soglie diverse
for thresh in [-2, -1, -0.5, 0, 0.5, 1]:
    frac = (delta > thresh).mean()
    print(f"Fraction above {thresh}: {frac:.3f}")
