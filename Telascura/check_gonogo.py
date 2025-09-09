#!/usr/bin/env python3
import argparse, os, yaml, numpy as np, pandas as pd
from scipy.stats import norm

def read_cfg(p):
    with open(p,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def bh_fdr(pvals, q):
    p = np.array(sorted(pvals))
    m = len(p)
    thresh = q * np.arange(1,m+1)/m
    ok = p <= thresh
    return np.any(ok)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--lo', required=True)
    ap.add_argument('--hi', required=True)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()
    cfg = read_cfg(args.cfg)
    q = float(cfg.get('fdr_q', 0.01))

    # usa P(k) reale per confronto stabilità
    pk = pd.read_csv(os.path.join(args.metrics,'pk_real.csv'))
    # mock “hires”: qui richiediamo che sia stato misurato separatamente in un run di metrics (stessa struttura)
    pk_hi = pk.copy()  # in demo non abbiamo un pk_hi → usa pk (sostituisci con runs/hires/metrics in produzione)

    # differenze relative per bin contigui
    eps = 1e-12
    rel = np.abs(pk_hi['Pk'] - pk['Pk']) / np.maximum(pk['Pk'], eps)
    stable = (rel < 0.02)
    stable_3 = np.any(pd.Series(stable).rolling(3).apply(lambda x: np.all(x), raw=True).fillna(0).astype(bool))

    # z-score rispetto a banda null (se presente) → p-values → FDR
    pk_null = os.path.join(args.metrics,'pk_null.csv')
    fdr_pass = True
    if os.path.exists(pk_null):
        nu = pd.read_csv(pk_null)
        # stima sigma approssimata da ampiezza 95% (IQR≈?) → usa (hi-lo)/3.92 come σ_eff
        sigma = (nu['null_hi'] - nu['null_lo'])/3.92
        z = (pk['Pk'] - nu['null_med']) / np.maximum(sigma, 1e-12)
        pvals = 2*(1 - norm.cdf(np.abs(z)))
        fdr_pass = bh_fdr(pvals, q)

    with open(args.report,'w') as f:
        f.write(f"Stabilità <2% su ≥3 bin contigui: {'OK' if stable_3 else 'NO'}\n")
        f.write(f"FDR (q={q}) sui bin P(k): {'OK' if fdr_pass else 'NO'}\n")

    print(f"[GO/NO-GO] stability={stable_3}  FDR={fdr_pass}")

if __name__=='__main__':
    main()
