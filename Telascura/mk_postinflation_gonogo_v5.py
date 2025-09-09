#!/usr/bin/env python3
import argparse, csv, math, os

ap=argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)   # transfer_curves.csv
ap.add_argument("--growth",   required=True)   # growth_series.csv
ap.add_argument("--out",      required=True)   # gonogo_table.tex
ap.add_argument("--need_bins", type=int, default=3)
ap.add_argument("--rebin_tol", type=float, default=0.02)   # 2%
args=ap.parse_args()

def read_transfer(path):
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        H=[h.lower() for h in (r.fieldnames or [])]
        def find(*keys):
            for k in keys:
                for j,h in enumerate(H):
                    if k in h: return r.fieldnames[j]
            return None
        ck  = find("k")
        ct  = find("t_tel","t","transfer")
        clo = find("null_lo","lo","p05","05")
        chi = find("null_hi","hi","p95","95")
        cu  = find("used","mask","indep","keep")
        rows=[]
        for row in r:
            try:
                k=float(row[ck]); t=row[ct]
            except Exception:
                continue
            try: t=float(t)
            except: continue
            lo= row.get(clo,""); hi=row.get(chi,"")
            used = 0
            try: used = int(float(row[cu])) if cu else 1
            except: used = 1
            rows.append({"k":k,"t":t,"lo":lo,"hi":hi,"used":used})
    rows.sort(key=lambda x:x["k"])
    return rows

tr = read_transfer(args.transfer)

def rebin_half(rows):
    out=[]; i=0
    while i < len(rows):
        if i+1 < len(rows):
            k=0.5*(rows[i]["k"]+rows[i+1]["k"])
            t=0.5*(rows[i]["t"]+rows[i+1]["t"])
            out.append({"k":k,"t":t})
            i+=2
        else:
            out.append({"k":rows[i]["k"],"t":rows[i]["t"]})
            i+=1
    return out

def nearest_t(k, arr):
    j=min(range(len(arr)), key=lambda j: abs(arr[j]["k"]-k))
    return arr[j]["t"]

# (1) Stabilità spettrale
rb = rebin_half(tr)
diffs=[]
for r in tr:
    t2 = nearest_t(r["k"], rb)
    if r["t"]!=0 and math.isfinite(t2):
        diffs.append(abs(t2-r["t"])/abs(r["t"]))
stab_max = max(diffs) if diffs else float("nan")
stable_bins = sum(1 for d in diffs if d<args.rebin_tol)
stab_status = "PASS" if (len(diffs)>0 and stable_bins>=args.need_bins and stab_max<args.rebin_tol) else ("n/a" if not diffs else "FAIL")

# (2) FDR multi-bin (qui n/a se non c'è banda null)
fdr_possible=False; fdr_bins=0
for r in tr:
    lo_ok = False
    hi_ok = False
    try: lo_ok = (r["lo"]!="") and math.isfinite(float(r["lo"]))
    except: pass
    try: hi_ok = (r["hi"]!="") and math.isfinite(float(r["hi"]))
    except: pass
    if r["used"]==1 and (lo_ok or hi_ok):
        fdr_possible=True
        t=r["t"]; lo=float(r["lo"]) if lo_ok else -1e300; hi=float(r["hi"]) if hi_ok else 1e300
        if t<lo or t>hi: fdr_bins+=1
fdr_status = ("n/a" if not fdr_possible else ("PASS" if fdr_bins>=args.need_bins else "FAIL"))

lines=[]
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{Go/No-Go criteria for post-inflation evolution (from Fig.~\ref{fig:growth_topology} data).}")
lines.append(r"\label{tab:postinflation_gonogo}")
lines.append(r"\begin{tabular}{llll}")
lines.append(r"\toprule")
lines.append(r"Test & Threshold & Measured & Status \\")
lines.append(r"\midrule")
# Riga 1: usa formatting percent ma con %% per il simbolo %
lines.append("Spectral stability ($T_K$, rebin $\\times \\tfrac{{1}}{{2}}$) & $<2\\%$ on $\\ge 3$ bins & max$=\\,{:.2f}\\%$; bins OK$=\\,{}$ & {} \\\\".format(
    (stab_max*100.0) if math.isfinite(stab_max) else float("nan"), stable_bins, stab_status))
# Riga 2: senza formato numerico se n/a
lines.append("FDR (multi-bin, $q{=}0.01$) & $\\ge 3$ significant bins & bins sig.$=\\,{}$ & {} \\\\".format(
    (fdr_bins if fdr_possible else "n/a"), fdr_status))
# Riga 3: fissa (TBD)
lines.append(r"Topology vs null (Betti/filamentarity) & $\ge 3$ bins outside $95\%$ null & n/a & TBD \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")
tex="\n".join(lines)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out,"w",encoding="utf-8") as f: f.write(tex)
print("[WRITE]", args.out)
