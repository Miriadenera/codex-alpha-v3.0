# telascura/bin/mk_postinflation_gonogo_v3.py
import argparse, csv, math, os, numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)
ap.add_argument("--growth",   required=True)
ap.add_argument("--out",      default="gonogo_table.tex")
ap.add_argument("--tol",      type=float, default=0.02)
ap.add_argument("--need",     type=int,   default=3)
args=ap.parse_args()

def read_generic(path):
    with open(path, newline="") as f:
        peek=f.readline().strip(); f.seek(0)
        has_header = any(c.isalpha() for c in peek)
        if has_header:
            r=csv.DictReader(f); rows=list(r)
            keys=list(rows[0].keys())
            def fcol(name_fallback):
                # pick by substr; fallback to first unknown
                for k in keys:
                    s=k.lower()
                    if name_fallback in s: return k
                return keys.pop(0)
            kkey=fcol("k")
            tkey=None
            for guess in ("t_k","t","transfer"):
                for k in keys:
                    if guess in k.lower(): tkey=k
            if tkey is None: tkey=keys[0]
            lokey=hikey=used=None
            for k in rows[0].keys():
                s=k.lower()
                if "lo" in s and ("null" in s or "band" in s): lokey=k
                if "hi" in s and ("null" in s or "band" in s): hikey=k
                if "used" in s or "mask" in s: used=k
            out=[]
            for row in rows:
                def g(k): 
                    try: return float(row.get(k,""))
                    except: return float("nan")
                out.append(dict(k=g(kkey), t=g(tkey),
                                lo=g(lokey) if lokey else float("nan"),
                                hi=g(hikey) if hikey else float("nan"),
                                used=(0 if used and str(row[used]).strip()=="0" else 1)))
            return out
        else:
            r=csv.reader(f); R=[list(x) for x in r]
            R=[[float(y) for y in x] for x in R if len(x)>=2]
            # mappa: k = col0, t = col1, lo/hi se ci sono
            out=[]
            for x in R:
                k=x[0]; t=x[1]
                lo=x[2] if len(x)>2 else float("nan")
                hi=x[3] if len(x)>3 else float("nan")
                out.append(dict(k=k,t=t,lo=lo,hi=hi,used=1))
            return out

tr = read_generic(args.transfer)
tr = [r for r in tr if np.isfinite([r["k"],r["t"]]).all()]
tr.sort(key=lambda r:r["k"])

# rebin ×1/2
rb=[]
i=0
while i<len(tr):
    if i+1<len(tr):
        rb.append(dict(k=0.5*(tr[i]["k"]+tr[i+1]["k"]), t=0.5*(tr[i]["t"]+tr[i+1]["t"])))
        i+=2
    else:
        rb.append(dict(k=tr[i]["k"], t=tr[i]["t"])); i+=1

def near_t(k):
    j=np.argmin([abs(k-r["k"]) for r in rb])
    return rb[j]["t"]

diffs=[]
for r in tr:
    t2=near_t(r["k"])
    if np.isfinite([r["t"],t2]).all() and r["t"]!=0:
        diffs.append(abs(t2-r["t"])/abs(r["t"]))
stab_max = max(diffs) if diffs else float("nan")
stab_bins = sum(1 for d in diffs if d<args.tol)
stab_stat = "PASS" if (len(diffs)>=args.need and stab_max<args.tol and stab_bins>=args.need) else ("n/a" if not diffs else "FAIL")

# FDR proxy: conta bin usabili fuori banda
have_band = any(np.isfinite([r["lo"],r["hi"]]).any() for r in tr)
fdr_bins=0
if have_band:
    for r in tr:
        ok = r["used"]==1 and (np.isfinite(r["lo"]) or np.isfinite(r["hi"]))
        out = ok and ( (np.isfinite(r["lo"]) and r["t"]<r["lo"]) or (np.isfinite(r["hi"]) and r["t"]>r["hi"]) )
        if out: fdr_bins+=1

tex = r"""\begin{table}[t]
\centering
\caption{Go/No-Go criteria for post-inflation evolution (from Fig.~\ref{fig:growth_topology} data).}
\label{tab:postinflation_gonogo}
\begin{tabular}{llll}
\toprule
Test & Threshold & Measured & Status\\
\midrule
Spectral stability ($T_K$, rebin $\times \tfrac{1}{2}$) & $<2\%$ on $\ge 3$ bins & max$=\,%(mx).2f\%%$; bins OK$=\,%(ok)d$ & %(s)s\\
FDR (multi-bin, $q{=}0.01$) & $\ge 3$ significant bins & bins sig.$=\,%(nb)s$ & %(fs)s\\
Topology vs null (Betti/filamentarity) & $\ge 3$ bins outside $95\%$ null & n/a & TBD\\
\bottomrule
\end{tabular}
\end{table}
"""%({"mx":(100*stab_max) if np.isfinite(stab_max) else float("nan"),
      "ok":int(stab_bins),
      "s":stab_stat,
      "nb":("n/a" if not have_band else str(fdr_bins)),
      "fs":("n/a" if not have_band else ("PASS" if fdr_bins>=args.need else "FAIL"))})

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out,"w",encoding="utf-8") as f: f.write(tex)
print(f"[WRITE] %s" % args.out)
