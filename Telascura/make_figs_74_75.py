#!/usr/bin/env python3
import argparse, os, yaml, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--log', default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pk = pd.read_csv(os.path.join(args.metrics,'pk_real.csv'))
    xi = pd.read_csv(os.path.join(args.metrics,'xi_real.csv'))
    pk_null = os.path.join(args.metrics,'pk_null.csv')
    xi_null = os.path.join(args.metrics,'xi_null.csv')
    pkN = pd.read_csv(pk_null) if os.path.exists(pk_null) else None
    xiN = pd.read_csv(xi_null) if os.path.exists(xi_null) else None

    # Fig. 74: P(k)
    plt.figure(figsize=(6,4))
    if pkN is not None:
        plt.fill_between(pkN['k'], pkN['null_lo'], pkN['null_hi'], alpha=0.2, label='null 95%')
        plt.plot(pkN['k'], pkN['null_med'], ls='--', label='null med')
    plt.plot(pk['k'], pk['Pk'], marker='o', label='real')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('k [h Mpc$^{-1}$]'); plt.ylabel('P(k)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out,'fig74_pk.pdf')); plt.close()

    # Fig. 75: xi(r)
    plt.figure(figsize=(6,4))
    if xiN is not None:
        plt.fill_between(xiN['r'], xiN['null_lo'], xiN['null_hi'], alpha=0.2, label='null 95%')
        plt.plot(xiN['r'], xiN['null_med'], ls='--', label='null med')
    plt.plot(xi['r'], xi['xi'], marker='o', label='real')
    plt.xscale('log')
    plt.xlabel('r [h$^{-1}$ Mpc]'); plt.ylabel(r'$\xi(r)$')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out,'fig75_xi.pdf')); plt.close()

    if args.log:
        with open(args.log,'w') as f: f.write(f"[WRITE] figs in {args.out}\n")

if __name__=='__main__':
    main()
