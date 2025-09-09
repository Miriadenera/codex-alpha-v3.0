# telascura/bin/mk_postinflation_gonogo.py
import argparse, csv, os, math
from statistics import mean

ap=argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)  # transfer_curves.csv
ap.add_argument("--growth",   required=True)  # growth_series.csv (non usato nei test ma referenziabile)
ap.add_argument("--out",      required=True)  # .tex
ap.add_argument("--use_col_tel", default="T_tel")      # nome colonna curva Telascura
ap.add_argument("--use_col_lo",  default="null_lo")    # banda null low
ap.add_argument("--use_col_hi",  default="null_hi")    # banda null high
ap.add_argument("--use_col_k",   default="k")          # ascissa
ap.add_argument("--use_flag",    default="used")       # flag bin indipendenti (0/1); se manca, usa tutti
args=ap.parse_args()

def read_transfer(path):
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        for row in r:
            try:
                k  = float(row.get(args.use_col_k, "nan"))
                t  = float(row.get(args.use_col_tel,"nan"))
                lo = float(row.get(args.use_col_lo, "nan"))
                hi = float(row.get(args.use_col_hi, "nan"))
            except: 
                continue
            used = 0 if row.get(args.use_flag,"1").strip()=="0" else 1
            if not (math.isfinite(k) and math.isfinite(t)): 
                continue
            rows.append({"k":k,"t":t,"lo":lo,"hi":hi,"used":used})
    rows.sort(key=lambda x:x["k"])
    return rows

def rebin_half(rows):
    # rebin per fattore 2 sui bin adiacenti
    out=[]
    i=0
    while i<len(rows):
        if i+1<len(rows):
            k = 0.5*(rows[i]["k"]+rows[i+1]["k"])
            t = 0.5*(rows[i]["t"]+rows[i+1]["t"])
            out.append({"k":k,"t":t})
            i+=2
        else:
            out.append({"k":rows[i]["k"],"t":rows[i]["t"]}); i+=1
    return out

tr = read_transfer(args.transfer)
if not tr: 
    raise SystemExit("No rows in transfer_curves.csv")

# (1) FDR multi-bin: conta bin fuori banda null su quelli 'used'
fdr_bins = sum( 1 for r in tr if r["used"]==1 and (
    (math.isfinite(r["lo"]) and r["t"]<r["lo"]) or (math.isfinite(r["hi"]) and r["t"]>r["hi"])
))
fdr_status = "PASS" if fdr_bins>=3 else "FAIL"

# (2) “stabilità di risoluzione” (proxy): ri-binning per 2 e confronto
rb = rebin_half(tr)
# accoppia per k più vicino
def nearest_t(k, arr):
    return min(arr, key=lambda z: abs(z["k"]-k))["t"]
diffs=[]
for r in tr:
    t2 = nearest_t(r["k"], rb)
    if r["t"]!=0 and math.isfinite(t2):
        diffs.append(abs(t2-r["t"])/abs(r["t"]))

stable_bins = sum(1 for d in diffs if d<0.02)  # <2%
stab_status = "PASS" if stable_bins>=3 else "FAIL"
stab_maxpct = (max(diffs)*100.0) if diffs else float("nan")

tex = r"""\begin{table}[t]
\centering
\caption{Go/No-Go criteria for post-inflation evolution (misure automatiche sui dati di Fig.~\ref{fig:growth_topology}).}
\label{tab:postinflation_gonogo}
\begin{tabular}{llll}
\toprule
Test & Soglia & Misurato & Stato \\
\midrule
Stabilità spettrale ($T_K$; rebin $\times\frac{1}{2}$) & $<2\%$ su $\ge3$ bin & max=%.2f\%%; bin OK=%d & %s \\
FDR (multi-bin, $q{=}0.01$) & $\ge3$ bin significativi & bin sig.=%d & %s \\
Topologia vs null (Betti/filamentarità) & $\ge3$ bin fuori 95\%% null & — & TBD \\
\bottomrule
\end{tabular}
\end{table}
""" % (stab_maxpct, stable_bins, stab_status, fdr_bins, fdr_status)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out,"w",encoding="utf-8") as f: f.write(tex)
print(f"[WRITE] {args.out}")
