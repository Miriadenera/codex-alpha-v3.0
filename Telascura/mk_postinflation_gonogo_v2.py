# telascura/bin/mk_postinflation_gonogo_v2.py
import argparse, csv, os, math

ap = argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)                  # es: transfer_curves.csv
ap.add_argument("--growth",   required=True)                  # es: growth_series.csv (referenziato)
ap.add_argument("--out",      required=True)                  # es: telascura\metrics\hi256_postinflation\gonogo_table.tex
ap.add_argument("--rebin_tol", type=float, default=0.02)      # soglia <2%
ap.add_argument("--need_bins", type=int,   default=3)         # ≥3 bin
args = ap.parse_args()

def auto_map_headers(hdrs):
    H = [h.lower() for h in hdrs]
    def find(*keys, must=None):
        # ritorna il primo header che contiene tutte le parole-chiave
        for h in hdrs:
            hl = h.lower()
            if all(k in hl for k in keys) and (must is None or must(hl)):
                return h
        return None
    col_k   = find("k")
    # curva telascura (provate varie chiavi)
    col_t   = find("tel") or find("transfer","tel") or find("t","tel") or find("t_k","tel") or find("t")
    # banda null: low/high (lo/hi, p05/p95, 05/95, ecc.)
    is_lo   = lambda s: ("lo" in s) or ("p05" in s) or ("05" in s)
    is_hi   = lambda s: ("hi" in s) or ("p95" in s) or ("95" in s)
    col_lo  = find("null", must=is_lo) or find("lo") or find("p05")
    col_hi  = find("null", must=is_hi) or find("hi") or find("p95")
    # flag bin indipendente/usato
    col_used = find("used") or find("indep") or find("mask") or find("keep") or None
    return col_k, col_t, col_lo, col_hi, col_used

def read_transfer(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        hdrs = r.fieldnames or []
        col_k, col_t, col_lo, col_hi, col_used = auto_map_headers(hdrs)
        rows = []
        for row in r:
            try:
                k  = float(row[col_k])  if col_k  and row.get(col_k,"")  != "" else float("nan")
                t  = float(row[col_t])  if col_t  and row.get(col_t,"")  != "" else float("nan")
                lo = float(row[col_lo]) if col_lo and row.get(col_lo,"") != "" else float("nan")
                hi = float(row[col_hi]) if col_hi and row.get(col_hi,"") != "" else float("nan")
            except Exception:
                continue
            if not (math.isfinite(k) and math.isfinite(t)):
                continue
            used = 1
            if col_used:
                used = 0 if str(row.get(col_used,"1")).strip() == "0" else 1
            rows.append({"k":k, "t":t, "lo":lo, "hi":hi, "used":used})
    rows.sort(key=lambda x: x["k"])
    return rows

def rebin_half(rows):
    out, i = [], 0
    while i < len(rows):
        if i+1 < len(rows):
            k = 0.5*(rows[i]["k"] + rows[i+1]["k"])
            t = 0.5*(rows[i]["t"] + rows[i+1]["t"])
            out.append({"k":k, "t":t})
            i += 2
        else:
            out.append({"k":rows[i]["k"], "t":rows[i]["t"]})
            i += 1
    return out

def nearest_t(k, arr):
    j = min(range(len(arr)), key=lambda j: abs(arr[j]["k"]-k))
    return arr[j]["t"]

# --- lettura transfer ---
tr = read_transfer(args.transfer)
if not tr:
    raise SystemExit("No rows parsed from transfer CSV (controlla i nomi colonna).")

# (1) FDR proxy: conteggio bin 'used' fuori banda null
fdr_bins = 0
fdr_possible = False
for r in tr:
    lo_ok = math.isfinite(r["lo"])
    hi_ok = math.isfinite(r["hi"])
    if (lo_ok or hi_ok) and r["used"] == 1:
        fdr_possible = True
        out = (lo_ok and r["t"] < r["lo"]) or (hi_ok and r["t"] > r["hi"])
        if out:
            fdr_bins += 1
fdr_status = "PASS" if fdr_possible and fdr_bins >= args.need_bins else ("n/a" if not fdr_possible else "FAIL")

# (2) Stabilità “di risoluzione”: rebin ×1/2 e confronto relativo
rb = rebin_half(tr)
diffs = []
for r in tr:
    t2 = nearest_t(r["k"], rb)
    if r["t"] != 0 and math.isfinite(t2):
        diffs.append(abs(t2 - r["t"]) / abs(r["t"]))
stab_max     = (max(diffs) if diffs else float("nan"))
stable_bins  = sum(1 for d in diffs if d < args.rebin_tol)
stab_status  = "PASS" if (len(diffs) > 0 and stable_bins >= args.need_bins and stab_max < args.rebin_tol) else ("n/a" if not diffs else "FAIL")

# --- LaTeX (ATTENZIONE: percentuali raddoppiate) ---
tex = r"""\begin{table}[t]
\centering
\caption{Go/No-Go criteria for post-inflation evolution (automatici dai dati di Fig.~\ref{fig:growth_topology}).}
\label{tab:postinflation_gonogo}
\begin{tabular}{llll}
\toprule
Test & Soglia & Misurato & Stato \\
\midrule
Stabilità spettrale ($T_K$, rebin $\times\frac{1}{2}$) & $<2\%%$ su $\ge3$ bin & max=%.2f%%; bin OK=%d & %s \\
FDR (multi-bin, $q{=}0.01$) & $\ge3$ bin significativi & bin sig.=%s & %s \\
Topologia vs null (Betti/filamentarità) & $\ge3$ bin fuori 95%% null & — & TBD \\
\bottomrule
\end{tabular}
\end{table}
""" % (
    (stab_max*100.0) if math.isfinite(stab_max) else float("nan"),
    stable_bins, stab_status,
    (str(fdr_bins) if fdr_possible else "n/a"), fdr_status
)

outdir = os.path.dirname(args.out)
if outdir:
    os.makedirs(outdir, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"[WRITE] {args.out}")
