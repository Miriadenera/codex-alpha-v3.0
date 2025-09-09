# telascura/bin/mk_postinflation_gonogo_v4.py
import argparse, csv, math, os, numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)
ap.add_argument("--growth",   required=True)
ap.add_argument("--out",      default="gonogo_table.tex")
ap.add_argument("--tol",      type=float, default=0.02)   # soglia <2%
ap.add_argument("--need",     type=int,   default=3)      # >=3 bin
args=ap.parse_args()

def read_generic(path):
    with open(path, newline="") as f:
        first=f.readline().strip(); f.seek(0)
        has_header = any(c.isalpha() for c in first)
        rows=[]
        if has_header:
            r=csv.DictReader(f); rows=list(r)
            keys=list(rows[0].keys())
            def pick(*subs, default=None):
                for k in keys:
                    s=k.lower()
                    if all(x in s for x in subs): return k
                return default
            kkey = pick("k") or keys[0]
            tkey = pick("t","k") or pick("transfer") or pick("t") or keys[1]
            lok  = pick("lo") if pick("lo") in keys else None
            hik  = pick("hi") if pick("hi") in keys else None
            used = pick("used") or pick("mask")
            out=[]
            for row in rows:
                def fnum(k):
                    try: return float(row.get(k,""))
                    except: return float("nan")
                out.append(dict(
                    k=fnum(kkey),
                    t=fnum(tkey),
                    lo=fnum(lok) if lok else float("nan"),
                    hi=fnum(hik) if hik else float("nan"),
                    used=(0 if (used and str(row.get(used,"1")).strip()=="0") else 1)
                ))
            return out
        else:
            r=csv.reader(f)
            for raw in r:
                if len(raw)<2: continue
                try:
                    k=float(raw[0]); t=float(raw[1])
                    lo=float(raw[2]) if len(raw)>2 else float("nan")
                    hi=float(raw[3]) if len(raw)>3 else float("nan")
                except:
                    continue
                rows.append(dict(k=k,t=t,lo=lo,hi=hi,used=1))
            return rows

# --- carica e ordina per k
tr = read_generic(args.transfer)
tr = [r for r in tr if np.isfinite([r["k"],r["t"]]).all()]
tr.sort(key=lambda r:r["k"])

# --- rebin x 1/2 e confronto stabilità
rb=[]
i=0
while i<len(tr):
    if i+1<len(tr):
        rb.append(dict(k=0.5*(tr[i]["k"]+tr[i+1]["k"]),
                       t=0.5*(tr[i]["t"]+tr[i+1]["t"])))
        i+=2
    else:
        rb.append(dict(k=tr[i]["k"], t=tr[i]["t"]))
        i+=1

def near_t(k):
    j=int(np.argmin([abs(k-r["k"]) for r in rb]))
    return rb[j]["t"]

diffs=[]
for r in tr:
    t2=near_t(r["k"])
    if np.isfinite([r["t"],t2]).all() and r["t"]!=0.0:
        diffs.append(abs(t2-r["t"])/abs(r["t"]))
stab_max  = (max(diffs) if diffs else float("nan"))
stab_bins = sum(1 for d in diffs if d<args.tol)
stab_stat = ("PASS" if (len(diffs)>=args.need and
                        np.isfinite(stab_max) and stab_max<args.tol and
                        stab_bins>=args.need)
             else ("n/a" if not diffs else "FAIL"))

# --- proxy FDR: conta bin “used” fuori banda null (se presenti lo/hi)
have_band = any(np.isfinite([r["lo"],r["hi"]]).any() for r in tr)
fdr_bins=0
if have_band:
    for r in tr:
        ok = (r["used"]==1) and (np.isfinite(r["lo"]) or np.isfinite(r["hi"]))
        out = ok and ( (np.isfinite(r["lo"]) and r["t"]<r["lo"]) or
                       (np.isfinite(r["hi"]) and r["t"]>r["hi"]) )
        if out: fdr_bins+=1

mx_txt = "n/a" if not np.isfinite(stab_max) else f"{100.0*stab_max:.2f}"
ok_txt = str(int(stab_bins))
fdr_txt= "n/a" if not have_band else str(int(fdr_bins))
fdr_st = "n/a" if not have_band else ("PASS" if fdr_bins>=args.need else "FAIL")

# --- template a segnaposto (niente % o {} speciali)
template = r"""
\begin{table}[t]
\centering
\caption{Go/No-Go criteria for post-inflation evolution (from Fig.~\ref{fig:growth_topology} data).}
\label{tab:postinflation_gonogo}
\begin{tabular}{llll}
\toprule
Test & Threshold & Measured & Status\\
\midrule
Spectral stability ($T_K$, rebin $\times \tfrac{1}{2}$) & $<2\%$ on $\ge 3$ bins & max$=\,__MX__\%$; bins OK$=\,__OK__$ & __S__\\
FDR (multi-bin, $q{=}0.01$) & $\ge 3$ significant bins & bins sig.$=\,__NB__$ & __FS__\\
Topology vs null (Betti/filamentarity) & $\ge 3$ bins outside $95\%$ null & n/a & TBD\\
\bottomrule
\end{tabular}
\end{table}
""".strip("\n")

tex = (template
       .replace("__MX__", mx_txt)
       .replace("__OK__", ok_txt)
       .replace("__S__",  stab_stat)
       .replace("__NB__", fdr_txt)
       .replace("__FS__", fdr_st))

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out,"w",encoding="utf-8") as f:
    f.write(tex+"\n")
print(f"[WRITE] {args.out}")
