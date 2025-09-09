# quick_pk.py
import numpy as np, argparse, csv
ap=argparse.ArgumentParser()
ap.add_argument('--snap', required=True)
ap.add_argument('--out',  required=True)
args=ap.parse_args()

d = np.load(args.snap)
# prendi il primo array 3D, o 'K' se presente
arr = None
for k in d.files:
    v = d[k]
    if isinstance(v, np.ndarray) and v.ndim == 3:
        arr = v; break
if arr is None:
    raise SystemExit("No 3D array found; keys: "+str(list(d.files)))

N = arr.shape[0]
F = np.fft.fftn(arr)
P = np.abs(F)**2

kx = np.fft.fftfreq(N); ky = np.fft.fftfreq(N); kz = np.fft.fftfreq(N)
KX,KY,KZ = np.meshgrid(kx,ky,kz, indexing='ij')
kr = np.sqrt(KX**2 + KY**2 + KZ**2).ravel()
Pr = P.ravel()

edges = np.linspace(0, 0.5, 201)  # 200 bin in unità di fftfreq
inds = np.digitize(kr, edges) - 1

k_cent, pk_mean = [], []
for i in range(len(edges)-1):
    sel = inds == i
    if np.any(sel):
        k_cent.append(0.5*(edges[i]+edges[i+1]))
        pk_mean.append(float(Pr[sel].mean()))

with open(args.out,'w', newline='') as f:
    w=csv.writer(f); w.writerow(['k','pk'])
    for k,p in zip(k_cent, pk_mean):
        w.writerow([k,p])
print("[WRITE]", args.out, "rows:", len(pk_mean))
