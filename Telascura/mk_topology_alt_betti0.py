#!/usr/bin/env python3
import argparse, os, sys, csv, math
import numpy as np

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
except Exception as e:
    print("ERROR: this script needs SciPy (ndimage). Install with:  py -m pip install scipy")
    sys.exit(1)

def parse_float_list(s):
    if ":" in s:  # start:step:stop
        a, b, c = [float(x) for x in s.split(":")]
        # include stop if exact multiple
        n = int(math.floor((c - a)/b + 1e-9)) + 1
        return [a + i*b for i in range(n)]
    return [float(x) for x in s.split(",") if x.strip()]

def phase_scramble_preserve_power(field, rng):
    # build random phases with Hermitian symmetry using FFT of a real white-noise seed
    white = rng.standard_normal(field.shape, dtype=np.float64)
    phi = np.angle(np.fft.fftn(white))
    amp = np.abs(np.fft.fftn(field))
    scrambled = np.fft.ifftn(amp * np.exp(1j*phi)).real
    # z-score to comparable units
    scrambled = (scrambled - scrambled.mean()) / (scrambled.std() + 1e-12)
    return scrambled

def grid_from_particles(pos, ngrid):
    # normalize to unit cube
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    L = (maxs - mins).max()
    x = (pos - mins) / L
    H, _ = np.histogramdd(x, bins=ngrid, range=[[0,1],[0,1],[0,1]])
    # fractional overdensity, then normalize to unit variance (for thresholding in nu)
    delta = H / (H.mean() + 1e-12) - 1.0
    delta = (delta - delta.mean()) / (delta.std() + 1e-12)
    return delta

def main():
    ap = argparse.ArgumentParser(description="Betti-0 topology (robust) with null 95% band")
    ap.add_argument("--snap", required=True, help="NPZ snapshot with 'pos' (N,3)")
    ap.add_argument("--ngrid", type=int, default=256)
    ap.add_argument("--sigmas", type=str, default="0.5,1,2,3,4,5,6")
    ap.add_argument("--nu", type=str, default="-2.5:0.5:2.5", help="threshold list or start:step:stop")
    ap.add_argument("--nnull", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="CSV to write (series)")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    sigmas = parse_float_list(args.sigmas)
    nus = parse_float_list(args.nu)
    rng = np.random.default_rng(args.seed)

    snap = np.load(args.snap, allow_pickle=True)
    if 'pos' not in snap:
        print("ERROR: snapshot must contain 'pos' array (N,3). Keys:", list(snap.keys()))
        sys.exit(2)
    pos = snap['pos']
    delta0 = grid_from_particles(pos, args.ngrid)

    # prepare CSV
    write_header = not (args.append and os.path.exists(args.out))
    f = open(args.out, "a" if args.append else "w", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow(["sigma","nu","beta0","null_lo","null_hi","used"])

    # precompute null surrogates once at base resolution, reuse across sigmas
    null_fields = [phase_scramble_preserve_power(delta0, rng) for _ in range(args.nnull)]

    # struct for 26-connectivity in 3D
    conn = np.ones((3,3,3), dtype=np.uint8)

    for sigma in sigmas:
        # smooth the measured field, z-score
        d = gaussian_filter(delta0, sigma, mode="wrap")
        d = (d - d.mean()) / (d.std() + 1e-12)

        # smooth nulls similarly
        d_null = []
    for nf in null_fields:
        g = gaussian_filter(nf, sigma, mode="wrap")
        g = (g - g.mean())/(g.std()+1e-12)
        d_null.append(g)


        for nu in nus:
            mask = d > nu
            _, beta0 = ndimage.label(mask, structure=conn)

            # null distribution
            beta0_null = []
            for nf in d_null:
                m = nf > nu
                _, b0 = ndimage.label(m, structure=conn)
                beta0_null.append(b0)

            lo = float(np.quantile(beta0_null, 0.025))
            hi = float(np.quantile(beta0_null, 0.975))

            used = int(beta0 < lo or beta0 > hi)
            w.writerow([f"{sigma:.3g}", f"{nu:.3g}", int(beta0), int(math.floor(lo)), int(math.ceil(hi)), used])

    f.close()
    print(f"[WRITE] {args.out}")

if __name__ == "__main__":
    main()
