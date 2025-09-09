#!/usr/bin/env python3
import argparse, os, csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True, help="topology_series.csv")
    ap.add_argument("--out-tex", required=True, help="topology_table.tex")
    ap.add_argument("--out-png", required=True, help="fig_topology_diag.png")
    args = ap.parse_args()

    # Leggi dati
    rows = []
    with open(args.topology, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'sigma': float(row['sigma']),
                'fil': float(row['filamentarity']),
                'lo': float(row['null_lo']),
                'hi': float(row['null_hi']),
                'used': int(row['used'])
            })
    
    if not rows:
        print("ERROR: No data in topology CSV")
        return
    
    # Calcola out-of-band
    n_used = sum(r['used'] for r in rows)
    n_outband = sum(1 for r in rows if r['used'] and (r['fil'] < r['lo'] or r['fil'] > r['hi']))
    
    status = "PASS" if n_outband >= 3 else "FAIL"
    
    print(f"[INFO] out-of-band used bins = {n_outband}/{n_used}  ==> {status}")
    
    # Genera tabella LaTeX
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Topology diagnostic: filamentarity vs smoothing scale.}\n")
        f.write("\\label{tab:topology_diag}\n")
        f.write("\\begin{tabular}{cccccc}\n")
        f.write("\\toprule\n")
        f.write("$\\sigma$ [vox] & Filamentarity & Null Lo & Null Hi & Used & Status \\\\\n")
        f.write("\\midrule\n")
        
        for r in rows:
            status_str = "out" if r['used'] and (r['fil'] < r['lo'] or r['fil'] > r['hi']) else "in"
            f.write(f"{r['sigma']:.1f} & {r['fil']:.3f} & {r['lo']:.3f} & {r['hi']:.3f} & {r['used']} & {status_str} \\\\\n")
        
        f.write("\\midrule\n")
        f.write(f"\\multicolumn{{4}}{{l}}{{Out-of-band bins}} & {n_outband}/{n_used} & {status} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    # Genera figura
    plt.figure(figsize=(8, 5))
    
    sigmas = [r['sigma'] for r in rows if r['used']]
    fils = [r['fil'] for r in rows if r['used']]
    los = [r['lo'] for r in rows if r['used']]
    his = [r['hi'] for r in rows if r['used']]
    
    plt.fill_between(sigmas, los, his, alpha=0.3, color='gray', label='95% null band')
    plt.plot(sigmas, fils, 'ro-', label='Measured', markersize=6)
    
    # Evidenzia out-of-band
    for i, (s, f, lo, hi) in enumerate(zip(sigmas, fils, los, his)):
        if f < lo or f > hi:
            plt.plot(s, f, 'bs', markersize=8, label='Out-of-band' if i == 0 else "")
    
    plt.xlabel('Smoothing scale σ [voxel]')
    plt.ylabel('Filamentarity')
    plt.title('Topology Diagnostic: Filamentarity vs Smoothing')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=150)
    print(f"[INFO] Saved {args.out_png}")

if __name__ == "__main__":
    main()