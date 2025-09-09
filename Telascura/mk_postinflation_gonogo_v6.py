# -*- coding: utf-8 -*-
import argparse, csv, os, math

ap = argparse.ArgumentParser()
ap.add_argument("--transfer", required=True)   # es: transfer_curves.csv
ap.add_argument("--growth",   required=True)   # es: growth_series.csv (non usato nei calcoli tabella ma tenuto per compatibilità)
ap.add_argument("--out",      required=True)   # es: gonogo_table.tex
ap.add_argument("--rebin_tol", type=float, default=0.02)  # soglia <2%
ap.add_argument("--need_bins", type=int,   default=3)     # ≥3 bin
args = ap.parse_args()

# --------------------------- utilità robuste sui CSV ---------------------------

def _find(hdrs, *keys, must=None):
    """Ritorna il primo header che contiene tutte le parole-chiave in keys (case-insensitive)."""
    for h in hdrs:
        hl = h.lower()
        ok = all(k in hl for k in keys)
        if ok and (must is None or must(hl)):
            return h
    return None

def _is_lo(s): s=s.lower(); return ("lo" in s) or ("p05" in s) or ("5" in s and "95" not in s)
def _is_hi(s): s=s.lower(); return ("hi" in s) or ("p95" in s) or ("95" in s)

def read_transfer(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        hdrs = r.fieldnames or []
        col_k   = _find(hdrs, "k")
        # curva Telascura / transfer
        col_t   = (_find(hdrs, "t_k") or _find(hdrs, "transfer") or _find(hdrs, "tel") or
                   _find(hdrs, "t") or _find(hdrs, "tk"))
        # banda null (low/high)
        col_lo  = _find(hdrs, "null", must=_is_lo) or _find(hdrs, "lo") or _find(hdrs, "p05")
        col_hi  = _find(hdrs, "null", must=_is_hi) or _find(hdrs, "hi") or _find(hdrs, "p95")
        # maschera indipendenza bin (opzionale)
        col_used = _find(hdrs, "used") or _find(hdrs, "indep") or _find(hdrs, "mask") or _find(hdrs, "keep")

        for row in r:
            try:
                k  = float(row[col_k]) if col_k and row.get(col_k,"")!="" else float("nan")
                t  = float(row[col_t]) if col_t and row.get(col_t,"")!="" else float("nan")
                lo = float(row[col_lo]) if col_lo and row.get(col_lo,"")!="" else float("nan")
                hi = float(row[col_hi]) if col_hi and row.get(col_hi,"")!="" else float("nan")
            except Exception:
                continue
            if not (math.isfinite(k) and math.isfinite(t)):
                continue
            used = 1
            if col_used is not None:
                used = 0 if str(row.get(col_used,"1")).strip() in ("0","False","false") else 1
            rows.append({"k":k, "t":t, "lo":lo, "hi":hi, "used":used})
    rows.sort(key=lambda x: x["k"])
    return rows

def rebin_half(rows):
    out, i = [], 0
    while i < len(rows):
        if i+1 < len(rows):
            out.append({"k":0.5*(rows[i]["k"]+rows[i+1]["k"]),
                        "t":0.5*(rows[i]["t"]+rows[i+1]["t"])})
            i += 2
        else:
            out.append({"k":rows[i]["k"], "t":rows[i]["t"]})
            i += 1
    return out

def nearest_t(k, arr):
    if not arr: return float("nan")
    j = min(range(len(arr)), key=lambda j: abs(arr[j]["k"]-k))
    return arr[j]["t"]

# ---------------------------- carica dati e valuta ----------------------------

tr = read_transfer(args.transfer)
if not tr:
    raise SystemExit("No rows parsed from transfer CSV (controlla intestazioni).")

# (1) FDR proxy: conta i bin 'used' fuori dalle bande null
fdr_bins = 0
fdr_possible = False
for r in tr:
    lo_ok = math.isfinite(r["lo"])
    hi_ok = math.isfinite(r["hi"])
    if (lo_ok or hi_ok) and r["used"] == 1:
        fdr_possible = True
        out = (lo_ok and r["t"] < r["lo"]) or (hi_ok and r["t"] > r["hi"])
        if out: fdr_bins += 1
fdr_status = ("PASS" if fdr_possible and fdr_bins >= args.need_bins
              else ("n/a" if not fdr_possible else "FAIL"))

# (2) Stabilità di risoluzione: rebin ×1/2 e confronto relativo
rb = rebin_half(tr)
diffs = []
for r in tr:
    t2 = nearest_t(r["k"], rb)
    if r["t"] != 0 and math.isfinite(t2):
        diffs.append(abs(t2 - r["t"]) / abs(r["t"]))
stab_max = (max(diffs) if diffs else float("nan"))
stable_bins = sum(1 for d in diffs if d < args.rebin_tol)
stab_status = ("PASS" if (len(diffs) > 0 and stable_bins >= args.need_bins and math.isfinite(stab_max) and stab_max < args.rebin_tol)
               else ("n/a" if not diffs else "FAIL"))

# ------------------------------- scrivi LaTeX ---------------------------------
max_pct = (stab_max*100.0) if math.isfinite(stab_max) else float("nan")

lines = []
lines.append("\\begin{table}[t]")
lines.append("\\centering")
lines.append("\\caption{Go/No-Go criteria for post-inflation evolution (from Fig.~\\ref{fig:growth_topology} data).}")
lines.append("\\label{tab:postinflation_gonogo}")
lines.append("\\begin{tabular}{llll}")
lines.append("\\toprule")
lines.append("Test & Threshold & Measured & Status \\\\")
lines.append("\\midrule")

# NB: f-string con graffe escape raddoppiate per il LaTeX
lines.append(
    f"Spectral stability ($T_K$, rebin $\\times \\tfrac{{{{1}}}}{{{{2}}}}$) & $<2\\%$ on $\\ge 3$ bins & "
    f"max$=\\,{max_pct:.2f}\\%$; bins OK$=\\,{stable_bins}$ & {stab_status} \\\\"
)
lines.append(
    f"FDR (multi-bin, $q{{{{=}}}}0.01$) & $\\ge 3$ significant bins & bins sig.$=\\,{(fdr_bins if fdr_possible else 'n/a')}$ & {fdr_status} \\\\"
)
lines.append("Topology vs null (Betti/filamentarity) & $\\ge 3$ bins outside $95\\%$ null & n/a & TBD \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
tex = "\n".join(lines) + "\n"

out_dir = os.path.dirname(args.out)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write(tex)

print(f"[WRITE] {args.out}")
