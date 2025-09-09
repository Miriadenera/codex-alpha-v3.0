#!/usr/bin/env python3
import argparse, csv, math
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def read_series(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for q in r:
            rows.append({
                "sigma": float(q["sigma"]),
                "nu": float(q["nu"]),
                "beta0": int(q["beta0"]),
                "lo": float(q["null_lo"]),
                "hi": float(q["null_hi"]),
                "used": int(q["used"]),
            })
    return rows

def group_by_sigma(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["sigma"]].append(r)
    for s in g:
        g[s].sort(key=lambda x: x["nu"])
    return dict(sorted(g.items()))

def make_table_tex(groups, out_tex):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Topology diagnostics (Betti--0) vs smoothing scale.}")
    lines.append("\\label{tab:topology_alt}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("$\\sigma$ & $\\beta_0(\\nu{=}0)$ & $[F^{95\\%}_{\\rm null}]$ & out & Status\\\\")
    lines.append("\\midrule")
    for s, rows in groups.items():
        # pick nu==0 for display
        r0 = min(rows, key=lambda r: abs(r["nu"] - 0.0))
        out = sum(r["used"] for r in rows)
        status = "PASS" if out >= 3 else "FAIL"
        lines.append(f"{s:.2f} & {r0['beta0']:d} & {int(math.floor(r0['lo']))}-{int(math.ceil(r0['hi']))} & {out:d} & {status}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(out_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"[WRITE] {out_tex}")

def pick_sigma_for_plot(groups):
    # choose sigma with max out-of-band; else closest to 2.0
    best = None
    best_out = -1
    for s, rows in groups.items():
        out = sum(r["used"] for r in rows)
        if out > best_out:
            best_out = out
            best = s
    if best is None:
        best = min(groups.keys(), key=lambda x: abs(x-2.0))
    return best

def make_plot(groups, out_png):
    s = pick_sigma_for_plot(groups)
    rows = groups[s]
    nu = np.array([r["nu"] for r in rows])
    b0 = np.array([r["beta0"] for r in rows], dtype=float)
    lo = np.array([r["lo"] for r in rows], dtype=float)
    hi = np.array([r["hi"] for r in rows], dtype=float)

    plt.figure(figsize=(6.6,3.6), dpi=200)
    # shaded null
    plt.fill_between(nu, lo, hi, alpha=0.25, label="null 95%")
    plt.plot(nu, b0, marker="o", lw=1.5, label=fr"measured ($\sigma={s:.2f}$)")
    # highlight out-of-band points
    oob = (b0 < lo) | (b0 > hi)
    plt.scatter(nu[oob], b0[oob], s=22)
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\beta_0(\nu)$")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[WRITE] {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--out-tex", default="topology_alt_table.tex")
    ap.add_argument("--out-png", default="fig_topology_alt.png")
    args = ap.parse_args()

    rows = read_series(args.series)
    groups = group_by_sigma(rows)
    make_table_tex(groups, args.out_tex)
    make_plot(groups, args.out_png)

if __name__ == "__main__":
    main()
