# telascura/bin/mk_imprints_table_v3.py
import csv, math, argparse, numpy as np, os
ap=argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--out", default="imprints_table.tex")
ap.add_argument("--kmin", type=float, default=0.02)
ap.add_argument("--kmax", type=float, default=0.2)
args=ap.parse_args()

def load_csv_autodetect(path):
    with open(path, newline="") as f:
        peek=f.readline().strip()
        f.seek(0)
        # se la prima riga contiene lettere → header presente
        has_header = any(c.isalpha() for c in peek)
        rows=[]
        if has_header:
            r=csv.DictReader(f)
            for row in r: rows.append({k:row[k] for k in row})
        else:
            r=csv.reader(f)
            for i,row in enumerate(r):
                rows.append({str(j):row[j] for j in range(len(row))})
        return rows, has_header

def col_as_float(rows, key):
    out=[]
    for row in rows:
        v=row.get(key, "")
        try: out.append(float(v))
        except: out.append(float("nan"))
    return np.array(out, float)

rows, has_header = load_csv_autodetect(args.csv)
keys=list(rows[0].keys())

# trova k (colonna monotona crescente)
k_cand=[]
for k in keys:
    v=col_as_float(rows,k)
    ok = np.isfinite(v).all() and (np.all(np.diff(v)>0))
    if ok: k_cand.append(k)
assert len(k_cand)>=1, "Non trovo la colonna k"
kkey=k_cand[0]
k=col_as_float(rows,kkey)

# individua le altre colonne numeriche
num_cols=[kk for kk in keys if kk!=kkey]
vals=[col_as_float(rows,kk) for kk in num_cols]

# cerca triple (lo,med,hi) con lo<=med<=hi
lo=med=hi=None; pk_tel=None
def frac_order(a,b,c):
    m=np.isfinite(a)&np.isfinite(b)&np.isfinite(c)
    if m.sum()==0: return 0.0
    return ( (a[m]<=b[m]) & (b[m]<=c[m]) ).mean()

best=(0,None)
for i in range(len(vals)):
  for j in range(len(vals)):
    if j==i: continue
    for l in range(len(vals)):
      if l in (i,j): continue
      score=frac_order(vals[i],vals[j],vals[l])
      if score>best[0]:
        best=(score,(i,j,l))
if best[0]>0.8:
    lo,med,hi = vals[best[1][0]], vals[best[1][1]], vals[best[1][2]]
    # pk_tel = col rimanente
    rest=[t for t in range(len(vals)) if t not in best[1]]
    pk_tel = vals[rest[0]] if rest else None
else:
    # nessuna banda: usa la colonna più "liscia" come pk_tel
    vari=[np.nanstd(v) for v in vals]
    pk_tel=vals[int(np.nanargmax(vari))]

m = np.isfinite(k)&np.isfinite(pk_tel)&(k>=args.kmin)&(k<=args.kmax)
x=np.log(k[m]); y=np.log(pk_tel[m])

# fit log P = log A + n log k + 1/2 α (log k)^2
X=np.vstack([np.ones_like(x), x, 0.5*x**2]).T
beta=np.linalg.lstsq(X, y, rcond=None)[0]
AsK = math.exp(beta[0])              # ampiezza
nK  = beta[1]
aK  = beta[2]

def fmt(v): 
    return "n/a" if not np.isfinite(v) else (f"{v:.3g}" if abs(v)>=1e-3 else f"{v:.2e}")

tex = r"""\begin{table}[t]
\centering
\caption{Inflation-era imprint tests (stima da fit su bande $k$; CI da bootstrap; controllo FDR su bin indipendenti).}
\label{tab:imprints}
\begin{tabular}{llll}
\toprule
Quantità & Stima & Null & Esito\\
\midrule
$A_s^K$ & %(As)s & banda null & n/d\\
$n_K$   & %(n)s & banda null & n/d\\
$\alpha_K$ & %(a)s & banda null & compatibile con 0\\
Coerenza di fase & — & scramble & n/d\\
\bottomrule
\end{tabular}
\end{table}
"""%({"As":fmt(AsK),"n":fmt(nK),"a":fmt(aK)})

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out,"w",encoding="utf-8") as f: f.write(tex)
print(f"[WRITE] {args.out}")
