#!/usr/bin/env python3
# quick_pk_particles.py  —  P(k) da snapshot con particelle (posizioni)
import numpy as np, argparse, csv, os

ap = argparse.ArgumentParser()
ap.add_argument('--snap', required=True, help='snapshot .npz con posizioni')
ap.add_argument('--out',  required=True, help='CSV di output (k,Pk,Nmodes)')
ap.add_argument('--ngrid', type=int, default=256, help='griglia per binning 3D')
ap.add_argument('--pos-key', default='pos')
ap.add_argument('--box-key', default='L')
args = ap.parse_args()

d = np.load(args.snap)
if args.pos_key not in d.files:
    print(f"ERROR: key '{args.pos_key}' not in {args.snap}; keys={list(d.files)}")
    raise SystemExit(2)

pos = np.array(d[args.pos_key], float)  # (N,3)
if pos.ndim != 2 or pos.shape[1] != 3:
    print(f"ERROR: {args.pos_key} has shape {pos.shape}, expected (N,3).")
    raise SystemExit(2)

# box-length
L = d[args.box_key] if args.box_key in d.files else np.ptp(pos, axis=0).max()
L = float(np.max(L)) if np.ndim(L) else float(L)

# porta in [0,L)
pos = np.mod(pos, L)

# binning 3D rapido (NGP) -> densità
N = int(args.ngrid)
H, _ = np.histogramdd(pos, bins=N, range=[[0, L], [0, L], [0, L]])
nbar = H.mean()
delta = H / nbar - 1.0

# FFT & potenza
F = np.fft.fftn(delta)
P3 = np.abs(F)**2
V = L**3
P3 *= V / (N**6)  # normalizzazione standard per numpy FFT

# griglia k
kx = 2*np.pi*np.fft.fftfreq(N, d=L/N)
ky = 2*np.pi*np.fft.fftfreq(N, d=L/N)
kz = 2*np.pi*np.fft.fftfreq(N, d=L/N)
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
kr = np.sqrt(KX**2 + KY**2 + KZ**2)

# media radiale
kmax = kx.max()*np.sqrt(3.0)
nbin = N//2
edges = np.linspace(0.0, kmax, nbin+1)
which = np.digitize(kr.ravel(), edges) - 1

Pk = np.zeros(nbin)
Nm = np.zeros(nbin, dtype=int)
flatP = P3.ravel()
for i in range(nbin):
    m = (which == i)
    Nm[i] = int(m.sum())
    if Nm[i] > 0:
        Pk[i] = flatP[m].mean()

k = 0.5*(edges[:-1] + edges[1:])

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
with open(args.out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['k', 'Pk', 'Nmodes'])
    for ki, pi, ni in zip(k, Pk, Nm):
        if ni > 0:
            w.writerow([f'{ki:.9e}', f'{pi:.9e}', int(ni)])

print(f"[WRITE] {args.out}  bins={(Nm>0).sum()}  keys={list(d.files)}")
