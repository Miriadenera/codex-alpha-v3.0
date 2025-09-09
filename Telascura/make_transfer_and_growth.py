#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def read_pk(path):
    df = pd.read_csv(path)
    # rileva colonne k e Pk
    kcol = [c for c in df.columns if c.strip().lower().startswith("k")][0]
    pcol = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("p") or "pk" in cl or "p(k)" in cl:
            pcol = c; break
    if pcol is None: pcol = df.columns[1]
    return np.asarray(df[kcol], float), np.asarray(df[pcol], float)

def read_monitor(path, prefer=None):
    df = pd.read_csv(path)
    colsL = {c.lower(): c for c in df.columns}

    # scala fattore a
    if "a" in colsL:
        a = np.asarray(df[colsL["a"]], float)
    elif "z" in colsL:
        z = np.asarray(df[colsL["z"]], float); a = 1.0/(1.0+z)
    else:
        a = np.linspace(0.1, 1.0, len(df))

    # scegli la colonna D_K
    cand = []
    if prefer: cand.append(prefer.lower())
    cand += ["d_k","dk","dk_rms","logdk","log_dk","growth","log_growth"]
    dcol = None
    for k in cand:
        if k in colsL: dcol = colsL[k]; break
        # match parziale (es. "dk_rms (avg)")
        for c in df.columns:
            if k in c.lower():
                dcol = c; break
        if dcol: break
    if dcol is None:
        raise RuntimeError("Cannot find D_K column in monitor CSV (tried: %s)" % cand)

    D = np.asarray(df[dcol], float)
    # restituisci anche log D_K
    D_lin = np.clip(D, 1e-30, None)
    logD = np.log(D_lin)
    return a, logD, D_lin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pk-ini", required=True)
    ap.add_argument("--pk-fin", required=True)
    ap.add_argument("--monitor", required=True)
    ap.add_argument("--monitor-col", default=None, help="nome colonna (es. dk_rms)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--transfer-csv", default="transfer_curves.csv")
    ap.add_argument("--growth-csv",   default="growth_series.csv")
    args = ap.parse_args()

    # TRANSFER
    k0,P0 = read_pk(args.pk_ini)
    k1,P1 = read_pk(args.pk_fin)
    kc = np.intersect1d(np.round(k0,6), np.round(k1,6))
    m0 = np.isin(np.round(k0,6), kc)
    m1 = np.isin(np.round(k1,6), kc)
    T  = P1[m1]/np.maximum(P0[m0], 1e-30)
    pd.DataFrame({"k":k1[m1], "T_K":T}).to_csv(args.transfer_csv, index=False)

    # GROWTH
    a, logDk, Dk = read_monitor(args.monitor, prefer=args.monitor_col)
    pd.DataFrame({"a":a, "D_K":Dk, "logD_K":logDk}).to_csv(args.growth_csv, index=False)

    # PLOT
    fig,ax = plt.subplots(1,2, figsize=(10,4.2), dpi=140)
    ax[0].loglog(k1[m1], T, lw=2.2)
    ax[0].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax[0].set_ylabel(r"$T_K(k)\equiv P_k(z{=}0)/P_k(z_{\rm ini})$")
    ax[0].grid(True, which="both", ls=":", alpha=0.4)

    ax[1].plot(a, logDk, lw=2.2)
    ax[1].set_xlabel(r"$a$")
    ax[1].set_ylabel(r"$\log D_K(a)$")
    ax[1].grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[WRITE] {args.out} | CSV: {args.transfer_csv}, {args.growth_csv}")

if __name__ == "__main__":
    main()
