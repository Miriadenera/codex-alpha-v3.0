#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def read_yaml_L(yaml_path, fallback=750.0):
    try:
        import yaml
        with open(yaml_path,'r',encoding='utf-8') as f:
            y=yaml.safe_load(f)
        for k in ["box","project","io"]:
            pass
        for key in ["box.Lbox_hMpc","Lbox_hMpc","box_size_hMpc"]:
            cur=y
            ok=True
            for p in key.split("."):
                if isinstance(cur,dict) and p in cur: cur=cur[p]
                else: ok=False; break
            if ok: return float(cur)
    except Exception:
        pass
    return float(fallback)

def detect_cols(df):
    cols=[c.lower() for c in df.columns]
    # k
    kcol=None
    for cand in ["k","k_hmpc","khmpc","k_hmpc,pk_k2","kh^-1mpc","k[h mpc^-1]"]:
        for i,c in enumerate(cols):
            if cand in c or c==cand:
                kcol=df.columns[i]; break
        if kcol: break
    if kcol is None: kcol=df.columns[0]
    # Pk
    pcol=None
    for cand in ["pk","pk_k2","p_k","p(k)","power"]:
        for i,c in enumerate(cols):
            if cand in c or c==cand:
                pcol=df.columns[i]; break
        if pcol: break
    if pcol is None: pcol=df.columns[1]
    return kcol,pcol

def chi2_95_factors(nu):
    # Approx 95% CI for chi2_nu/nu (no scipy): 1 ± 1.96*sqrt(2/nu)
    eps = 1.0 - 1.96*np.sqrt(np.maximum(2.0/np.maximum(nu,1.0), 1e-9))
    ups = 1.0 + 1.96*np.sqrt(np.maximum(2.0/np.maximum(nu,1.0), 1e-9))
    return np.clip(eps, 0.05, None), ups

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pk", required=True, help="telascura/cfg/pk_telascura.csv")
    ap.add_argument("--yaml", required=False, default=None, help="telascura/cfg/telascura_hi.yaml (per Lbox)")
    ap.add_argument("--out", required=True, help="fig77_pk_primordial.png")
    ap.add_argument("--csv", required=True, help="pk_primordial.csv")
    args=ap.parse_args()

    L = read_yaml_L(args.yaml, fallback=750.0) if args.yaml else 750.0  # h^-1 Mpc
    df = pd.read_csv(args.pk)
    kcol,pcol = detect_cols(df)
    k = np.asarray(df[kcol], float)
    P = np.asarray(df[pcol], float)

    # Delta k per bin (geometric spacing robust)
    ksort = np.argsort(k); k = k[ksort]; P = P[ksort]
    dk = np.zeros_like(k)
    if len(k)>2:
        dk[1:-1] = 0.5*(k[2:]-k[:-2])
    if len(k)>1:
        dk[0] = k[1]-k[0]
        dk[-1] = k[-1]-k[-2]
    dk = np.clip(dk, 0.5*np.min(np.diff(np.unique(k))), None)

    # N_modes ~ (L^3/(2π^2)) k^2 Δk ; ν = 2 N_modes
    L3 = L**3
    Nm = (L3/(2.0*np.pi**2))*k**2*dk
    nu = 2.0*np.maximum(Nm, 1.0)
    lo_fac, hi_fac = chi2_95_factors(nu)
    null_med = P.copy()
    null_lo  = P*lo_fac
    null_hi  = P*hi_fac

    out = pd.DataFrame({"k":k,"Pk":P,"null_med":null_med,"null_lo":null_lo,"null_hi":null_hi})
    out.to_csv(args.csv, index=False)

    plt.figure(figsize=(6.6,4.6), dpi=140)
    plt.fill_between(k, null_lo, null_hi, alpha=0.2, label="null 95% (cosmic variance)")
    plt.loglog(k, P, lw=2.2, label="Telascura $P_K(k)$")
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_K(k)$")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[WRITE] {args.out}, {args.csv}")

if __name__=="__main__":
    main()
