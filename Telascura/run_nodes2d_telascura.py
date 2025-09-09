import argparse, time
from pathlib import Path
import numpy as np

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML mancante. Installa con: py -3 -m pip install pyyaml")

def gaussian_field_2d(n, k0):
    rng = np.random.default_rng()
    eta = rng.normal(0.0, 1.0, size=(n, n))
    fk = np.fft.rfft2(eta)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.rfftfreq(n)[None, :]
    kk = np.sqrt(kx*kx + ky*ky) + 1e-12
    fk *= np.exp(-(kk / k0)**2)        # filtro gaussiano in k
    f  = np.fft.irfft2(fk, s=(n, n))
    f -= f.mean()
    f /= (f.std() + 1e-12)
    return f.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))

    N          = int(cfg.get("box_size", 512))
    n_steps    = int(cfg.get("n_steps", 600))
    save_every = int(cfg.get("save_every", 50))
    dx         = float(cfg.get("dx", 1.0))
    k0         = float(cfg.get("init", {}).get("k0", 0.05))
    out_dir    = Path(cfg.get("output", {}).get("out_dir", "telascura/runs/nodes2d_pilot"))
    metrics_dir= Path(cfg.get("output", {}).get("metrics_dir", "telascura/metrics/nodes2d"))
    figs_dir   = Path(cfg.get("output", {}).get("figs_dir", "telascura/figs/nodes2d"))

    for p in (out_dir, metrics_dir, figs_dir, Path("telascura/logs")):
        p.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    K0 = gaussian_field_2d(N, k0)

    # timeline in a da 1/1000 a 1 (lineare)
    a_ini, a_fin = 1.0/1000.0, 1.0
    a_series = np.linspace(a_ini, a_fin, n_steps+1)

    rows = ["step,a,dk_rms,time_sec"]
    for i, a in enumerate(a_series):
        dk_rms = float(np.std(a * K0))
        if (i % max(1, save_every)) == 0 or i == n_steps:
            rows.append(f"{i},{a:.6e},{dk_rms:.6e},{time.time()-t0:.3f}")

    mon = out_dir / "monitor_DK.csv"
    mon.write_text("\n".join(rows) + "\n", encoding="utf-8")

    if args.log:
        Path(args.log).write_text(
            f"nodes2d pilot OK\nN={N} steps={n_steps} k0={k0}\nmonitor={mon}\n",
            encoding="utf-8"
        )

if __name__ == "__main__":
    main()
