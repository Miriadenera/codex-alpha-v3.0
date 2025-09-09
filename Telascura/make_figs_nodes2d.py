import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML mancante. Installa con: py -3 -m pip install pyyaml")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path(cfg.get("output", {}).get("out_dir", "telascura/runs/nodes2d_pilot"))
    mon = runs_dir / "monitor_DK.csv"
    if not mon.exists():
        raise FileNotFoundError(f"monitor_DK.csv non trovato: {mon}")

    data = np.genfromtxt(mon, delimiter=",", names=True, dtype=None, encoding=None)
    a  = data["a"]
    dk = data["dk_rms"]

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(a, dk, lw=2, label="Telascura 2D pilot")
    plt.xlabel(r"$a$")
    plt.ylabel(r"$D_K \propto \mathrm{std}(K)$")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    png = out_dir / "fig_nodes2d.png"
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.close()

    if args.log:
        Path(args.log).write_text(f"wrote {png}\n", encoding="utf-8")

if __name__ == "__main__":
    main()
