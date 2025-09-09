#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, sys, os
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    print("[FATAL] Serve PyYAML: pip install pyyaml", file=sys.stderr); sys.exit(1)

def lnPk_power_running(lnk, lnA, nK, alphaK, k_p):
    """ ln P(k) = ln A + (nK-1) ln(k/kp) + 0.5 alphaK [ln(k/kp)]^2 """
    return lnA + (nK - 1.0)*(lnk - np.log(k_p)) + 0.5*alphaK*(lnk - np.log(k_p))**2

def cutoff_factor(k, kc, shape):
    if kc is None or kc <= 0: 
        return 1.0
    if shape == "gaussian":
        return np.exp(- (k/kc)**2)
    elif shape == "exponential":
        return np.exp(- (k/kc))
    else:
        return 1.0

def window_k(k, R, kind):
    if kind == "gaussian":
        return np.exp(-0.5*(k*R)**2)
    elif kind == "tophat":
        x = k*R + 1e-30
        W = 3.0*(np.sin(x) - x*np.cos(x))/x**3
        return W
    else:
        return np.ones_like(k)

def sigma_from_Pk(k, Pk, R, win):
    """ sigma^2 = ∫ d^3k/(2π)^3 P(k)|W(kR)|^2  (integrazione log-spaced) """
    W = window_k(k, R, win)
    integrand = (k**2)*Pk*(W**2) / (2.0*np.pi**2)
    # integrazione in dlnk: ∫ k^3/(2π^2) P(k) |W|^2 dlnk
    dlnk = np.log(k[1]/k[0])
    return float(np.sum(integrand * dlnk))

def main():
    ap = argparse.ArgumentParser(description="Genera pk_telascura.csv dal YAML con normalizzazione su sigma_K(R).")
    ap.add_argument("--cfg", required=True, help="telascura.yaml")
    ap.add_argument("--out", required=True, help="output CSV (pk_telascura.csv)")
    args = ap.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- leggi parametri ---
    pk_cfg   = cfg.get("pk", {})
    norm_cfg = cfg.get("norm", {})
    box_cfg  = cfg.get("box", {})

    kmin  = float(pk_cfg.get("kmin_hMpc", 0.003))
    kmax  = float(pk_cfg.get("kmax_hMpc", 5.0))
    nk    = int(pk_cfg.get("nk", 256))
    kp    = float(pk_cfg.get("pivot_k_hMpc", 0.05))
    nK    = float(pk_cfg.get("nK", 1.0))
    aK    = float(pk_cfg.get("alphaK", 0.0))
    kc    = float(pk_cfg.get("cutoff_k_hMpc", 0.0))
    csh   = str(pk_cfg.get("cutoff_shape", "gaussian")).lower()

    R     = float(norm_cfg.get("R_hMpc", 5.0))
    wkind = str(norm_cfg.get("window", "gaussian")).lower()
    sigT  = float(norm_cfg.get("target_sigmaK", 1.0))

    Lbox  = float(box_cfg.get("Lbox_hMpc", 1000.0))

    # --- sanity sulla griglia k vs box ---
    kfund = 2.0*np.pi/max(1e-12, Lbox)
    if kmin < 0.8*kfund:
        print(f"[WARN] kmin={kmin:.4g} < 0.8 * k_fund={kfund:.4g}. Correggo a kmin={kfund:.4g}.")
        kmin = kfund

    k = np.logspace(np.log10(kmin), np.log10(kmax), nk)
    lnk = np.log(k)

    # --- forma spettrale a normalizzazione arbitraria lnA=0 ---
    lnA0 = 0.0
    lnPk0 = lnPk_power_running(lnk, lnA0, nK, aK, kp)
    Pshape = np.exp(lnPk0) * cutoff_factor(k, kc, csh)

    # --- normalizzazione su sigma_K(R) ---
    # sigma(A)^2 = A * sigma(Pshape)^2  -> A = (sigT^2) / sigma0^2
    sigma0 = sigma_from_Pk(k, Pshape, R, wkind)**0.5
    if sigma0 <= 0:
        print("[FATAL] sigma0=0 dalla forma spettrale. Controlla k-range/cutoff.", file=sys.stderr)
        sys.exit(2)
    A = (sigT / sigma0)**2
    Pk = A * Pshape

    # --- scrivi CSV ---
    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame({"k_hMpc": k, "Pk_K2": Pk})
    df.to_csv(args.out, index=False)
    print(f"[WRITE] {args.out}  (nk={len(df)})")
    print(f"[INFO]  sigma_K(R={R} h^-1 Mpc) target={sigT:.3g}  achieved={sigma_from_Pk(k, Pk, R, wkind)**0.5:.3g}")
    print(f"[INFO]  k-range = [{k.min():.4g}, {k.max():.4g}] h/Mpc; cutoff={csh}@{kc} h/Mpc; nK={nK}, alphaK={aK}, kp={kp}")

if __name__ == "__main__":
    main()
