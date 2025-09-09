#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

def read_cfg(p):
    if yaml is None:
        raise ModuleNotFoundError("PyYAML non installato: pip install pyyaml")
    with open(p,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def must_read_csv(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manca {what}: {path}")
    return pd.read_csv(path)

def maybe_read_csv(path):
    return pd.read_csv(path) if os.path.isfile(path) else None

def safe_log_series(x, y):
    """Rimuove valori non-positivi/NaN; ritorna maschera e dice se log-log è lecito."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x2, y2 = x[m], y[m]
    ok_log = (x2.size >= 2) and (y2.size >= 2)
    return x2, y2, ok_log

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Fig.74–75: P(k) e xi(r) da CSV in telascura/metrics")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--metrics", required=True)   # es: telascura/metrics
    ap.add_argument("--out", required=True)       # es: telascura/figs
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    _ = read_cfg(args.cfg)  # per future usi
    os.makedirs(args.out, exist_ok=True)

    pk_real = must_read_csv(os.path.join(args.metrics, "pk_real.csv"), "pk_real.csv")
    xi_real = must_read_csv(os.path.join(args.metrics, "xi_real.csv"), "xi_real.csv")
    pk_null = maybe_read_csv(os.path.join(args.metrics, "pk_null.csv"))
    xi_null = maybe_read_csv(os.path.join(args.metrics, "xi_null.csv"))

    # -------- Fig. 74: P(k) --------
    k_ok, Pk_ok, ok_log = safe_log_series(pk_real["k"], pk_real["Pk"])

    fig, ax = plt.subplots(figsize=(6.0,4.2), dpi=160)
    if ok_log:
        ax.loglog(k_ok, Pk_ok, lw=2, label="Telascura")
    else:
        # fallback robusto se qualche punto è zero/negativo
        ax.plot(k_ok, Pk_ok, lw=2, label="Telascura")
        ax.set_xscale("log")

    if pk_null is not None and set(["k","null_lo","null_hi"]).issubset(pk_null.columns):
        kN, loN, hiN, okN = safe_log_series(pk_null["k"], pk_null["null_lo"])
        _,  hiN2, _   = safe_log_series(pk_null["k"], pk_null["null_hi"])
        # allinea le lunghezze
        n = min(len(kN), len(hiN2))
        if n >= 2:
            if ok_log and okN:
                ax.fill_between(kN[:n], loN[:n], hiN2[:n], alpha=0.25, label="Null (95% band)")
            else:
                ax.fill_between(kN[:n], loN[:n], hiN2[:n], alpha=0.25, label="Null (95% band)")
                ax.set_xscale("log")

    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    f74_png = os.path.join(args.out, "fig74_pk.png")
    f74_pdf = os.path.join(args.out, "fig74_pk.pdf")
    fig.tight_layout()
    fig.savefig(f74_png); fig.savefig(f74_pdf); plt.close(fig)

    # -------- Fig. 75: xi(r) --------
    # xi può essere <0: uso semi-log in x (come già facevamo)
    r = np.asarray(xi_real["r"], float)
    xi = np.asarray(xi_real["xi"], float)
    m = np.isfinite(r) & np.isfinite(xi) & (r > 0)
    r_ok, xi_ok = r[m], xi[m]

    fig, ax = plt.subplots(figsize=(6.0,4.2), dpi=160)
    ax.semilogx(r_ok, xi_ok, lw=2, label="Telascura")

    if xi_null is not None and set(["r","null_lo","null_hi"]).issubset(xi_null.columns):
        rn = np.asarray(xi_null["r"], float)
        lo = np.asarray(xi_null["null_lo"], float)
        hi = np.asarray(xi_null["null_hi"], float)
        m2 = np.isfinite(rn) & np.isfinite(lo) & np.isfinite(hi) & (rn > 0)
        if m2.sum() >= 2:
            ax.fill_between(rn[m2], lo[m2], hi[m2], alpha=0.25, label="Null (95% band)")

    ax.set_xlabel(r"$r\,[h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\xi(r)$")
    ax.axhline(0, lw=1, alpha=0.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    f75_png = os.path.join(args.out, "fig75_xi.png")
    f75_pdf = os.path.join(args.out, "fig75_xi.pdf")
    fig.tight_layout()
    fig.savefig(f75_png); fig.savefig(f75_pdf); plt.close(fig)

    if args.log:
        with open(args.log, "w", encoding="utf-8") as f:
            f.write(f"[WRITE] {f74_png}, {f75_png}\n")

if __name__ == "__main__":
    main()
