#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_monitor(path):
    # CSV con virgola, eventuali righe commentate con '#'
    df = pd.read_csv(path, sep=",", comment="#", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    # prendo 'a' (o ricavo da z)
    if "a" in df.columns:
        a = df["a"].astype(float).to_numpy()
    elif "z" in df.columns:
        a = (1.0 / (1.0 + df["z"].astype(float))).to_numpy()
    else:
        a = df.iloc[:, 1].astype(float).to_numpy()  # fallback
    # prendo D_K (dk_rms)
    if "dk_rms" in df.columns:
        dk = df["dk_rms"].astype(float).to_numpy()
    else:
        dk = df.iloc[:, 5].astype(float).to_numpy()  # fallback
    return a, dk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--A", required=True, help="monitor_DK.csv seed A")
    ap.add_argument("-b", "--B", required=True, help="monitor_DK.csv seed B")
    ap.add_argument("-o", "--out", default="telascura/metrics/dk_hi256")
    ap.add_argument("--figdir", default="telascura/figs/dk_hi256")
    ap.add_argument("--log", default=None)
    ap.add_argument("--floor", type=float, default=1e-12)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)

    aA, dkA = read_monitor(args.A)
    aB, dkB = read_monitor(args.B)

    # allineo su 'a' (interpolazione, lunghezza comune)
    a_lo = max(aA.min(), aB.min())
    a_hi = min(aA.max(), aB.max())
    n = min(len(aA), len(aB))
    a_common = np.linspace(a_lo, a_hi, num=n)

    dkA_i = np.interp(a_common, aA, dkA)
    dkB_i = np.interp(a_common, aB, dkB)

    # evito log(0) con un floor numerico
    f = args.floor
    dlog = np.log(np.clip(dkA_i, f, None)) - np.log(np.clip(dkB_i, f, None))
    max_abs = float(np.max(np.abs(dlog)))
    status = "PASS" if max_abs < 0.02 else "FAIL"

    # riassunto
    summ = os.path.join(args.out, "dk_hi256_summary.txt")
    with open(summ, "w", encoding="utf-8") as g:
        g.write(f"A={os.path.basename(args.A)}\n")
        g.write(f"B={os.path.basename(args.B)}\n")
        g.write(f"points={len(a_common)}\n")
        g.write("criterion=|Δlog D_K|<0.02\n")
        g.write(f"max_abs_dlogDK={max_abs:.6f}\n")
        g.write(f"result={status}\n")

    # figura
    plt.figure(figsize=(6,4))
    plt.loglog(aA, np.clip(dkA, f, None), label="seed A")
    plt.loglog(aB, np.clip(dkB, f, None), "--", label="seed B")
    plt.xlabel("a")
    plt.ylabel("D_K (rms)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    outpng = os.path.join(args.figdir, "fig_dk_hi256.png")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)

    if args.log:
        with open(args.log, "w", encoding="utf-8") as L:
            L.write(f"[INFO] wrote {outpng}\n")
            L.write(f"[INFO] summary {summ}\n")
            L.write(f"[INFO] max_abs_dlogDK={max_abs:.6f} status={status}\n")

if __name__ == "__main__":
    main()