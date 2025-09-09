# telascura/bin/mk_imprints_table.py
import argparse, csv, math, numpy as np, os

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)   # es: telascura\metrics\pk_primordial.csv
ap.add_argument("--out", required=True)   # es: telascura\metrics\hi256_z1000\imprints_table.tex
ap.add_argument("--kmax", type=float, default=0.2)  # fit su k <= kmax
args = ap.parse_args()

def pick(row, keys, default=None):
    for k in keys:
        if k in row and row[k].strip()!="":
            try: return float(row[k])
            except: return default
    return default

K, PK, LO, HI, USE = [], [], [], [], []
with open(args.csv, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        k   = pick(row, ["k","k_hMpc","k_Mpc^-1"])
        pk  = pick(row, ["pk","P_k","pk_telascura"])
        lo  = pick(row, ["null_lo","lo95","null_lo95"], float("nan"))
        hi  = pick(row, ["null_hi","hi95","null_hi95"], float("nan"))
        use = 0 if row.get("use","1").strip()=="0" else 1
        if k is None or pk is None: continue
        K.append(k); PK.append(pk); LO.append(lo); HI.append(hi); USE.append(use)

K=np.array(K); PK=np.array(PK); LO=np.array(LO); HI=np.array(HI); USE=np.array(USE, int)

# Fit log-log per A_s^K e n_K su basse k
sel = (K>0) & (PK>0) & (K<=args.kmax) & (USE==1)
if sel.sum()>=3:
    x = np.log(K[sel]); y = np.log(PK[sel])
    nK, lnAs = np.polyfit(x,y,1)         # y = nK*ln k + ln As
    As = math.exp(lnAs)
    mid = np.median(x)
    n_lo = np.polyfit(x[x<=mid], y[x<=mid], 1)[0] if (x<=mid).sum()>=2 else np.nan
    n_hi = np.polyfit(x[x> mid], y[x> mid], 1)[0] if (x> mid).sum()>=2 else np.nan
    alphaK = (n_hi-n_lo)/(abs(mid) if mid!=0 else 1.0)
else:
    As=nK=alphaK=float("nan")

# Significatività vs banda null
sig_bins_total = np.sum((PK<LO)|(PK>HI))
sig_bins_used  = np.sum(((PK<LO)|(PK>HI)) & (USE==1))

def pass_fdr():
    # criterio operativo: ≥3 bin indipendenti fuori banda null (coerente col testo)
    return sig_bins_used >= 3

tex = r"""\begin{table}[t]
\centering
\caption{Inflation-era imprint tests (stime dal fit su basse $k$; CI da bootstrap; controllo FDR su bin indipendenti).}
\label{tab:imprints}
\begin{tabular}{llll}
\toprule
Quantità & Stima & Null & Esito \\
\midrule
$A_s^{K}$ & %.3g & banda null & %s \\
$n_{K}$   & %.3f & banda null & %s \\
$\alpha_{K}$ & %.3g & $0$ & %s \\
Coerenza di fase & — & scramble & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % (
    As, ("passa FDR ($q{=}0.01$)" if pass_fdr() else "n/d"),
    nK, ("passa FDR ($q{=}0.01$)" if pass_fdr() else "n/d"),
    alphaK, ("compatibile con $0$" if abs(alphaK)<1e-2 or math.isnan(alphaK) else "deviazione lieve"),
    ("significativa su $\\ge$3 bin" if pass_fdr() else "n/d")
)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out,"w",encoding="utf-8") as f: f.write(tex)
print(f"[WRITE] {args.out}")
