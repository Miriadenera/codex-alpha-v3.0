#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse, json
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

# ---------- YAML / util ----------
def read_yaml(p):
    if yaml is None:
        raise ModuleNotFoundError("PyYAML not installed: pip install pyyaml")
    with open(p,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def pick(d, *keys, default=None):
    for k in keys:
        cur = d; ok = True
        for part in str(k).split('.'):
            if isinstance(cur, dict) and part in cur: cur = cur[part]
            else: ok=False; break
        if ok: return cur
    return default

# ---------- snapshot I/O ----------
def load_ics(npz_path, cfg):
    npz = np.load(npz_path, allow_pickle=True)
    # L
    if 'L' in npz.files: L = float(npz['L'])
    elif 'L_hMpc' in npz.files: L = float(npz['L_hMpc'])
    else: L = float(pick(cfg,'box.Lbox_hMpc','box_size_hMpc'))
    # pos
    pos = np.array(npz['pos'] if 'pos' in npz.files else npz['x'], np.float32)
    # vel opzionali
    vel = np.zeros_like(pos, dtype=np.float32)
    if 'vel' in npz.files: vel = np.array(npz['vel'], np.float32)
    # gradK per particella (se presente)
    gKp = np.array(npz['gradK'], np.float32) if 'gradK' in npz.files else None
    return L, pos, vel, gKp

def save_snap(out_dir, tag, L, pos, vel, a, z):
    ensure_dir(out_dir)
    np.savez_compressed(
        os.path.join(out_dir, f"snap_{tag}.npz"),
        L=L, pos=pos.astype(np.float32), vel=vel.astype(np.float32),
        a=np.float32(a), z=np.float32(z)
    )

# ---------- CIC deposit / sample ----------
def cic_deposit(pos, L, N, weights=None, vec=None):
    """Deposit scalar weights or vector field onto grid with CIC."""
    cell = L / N
    x = (pos / cell) % N
    i0 = np.floor(x).astype(np.int64)
    d  = x - i0
    i1 = (i0 + 1) % N
    wx = np.stack([1-d[:,0], d[:,0]], axis=1)
    wy = np.stack([1-d[:,1], d[:,1]], axis=1)
    wz = np.stack([1-d[:,2], d[:,2]], axis=1)

    if vec is None:
        grid = np.zeros((N,N,N), np.float32)
        w = 1.0 if weights is None else weights.astype(np.float32)
        for ix in (0,1):
            for iy in (0,1):
                for iz in (0,1):
                    contrib = (wx[:,ix]*wy[:,iy]*wz[:,iz]) * w
                    np.add.at(grid, ( (i0[:,0]+ix)%N, (i0[:,1]+iy)%N, (i0[:,2]+iz)%N ), contrib)
        return grid
    else:
        grid = np.zeros((3,N,N,N), np.float32)
        v = vec.astype(np.float32)
        for ix in (0,1):
            for iy in (0,1):
                for iz in (0,1):
                    wcell = (wx[:,ix]*wy[:,iy]*wz[:,iz])[:,None]
                    contrib = v * wcell
                    for c in range(3):
                        np.add.at(grid[c], ( (i0[:,0]+ix)%N, (i0[:,1]+iy)%N, (i0[:,2]+iz)%N ), contrib[:,c])
        return grid

def cic_sample(grid, pos, L):
    """Trilinear interpolation from grid to particle positions."""
    N = grid.shape[0]
    cell = L / N
    x = (pos / cell) % N
    i0 = np.floor(x).astype(np.int64)
    d  = x - i0
    i1 = (i0 + 1) % N
    wx0, wx1 = 1-d[:,0], d[:,0]
    wy0, wy1 = 1-d[:,1], d[:,1]
    wz0, wz1 = 1-d[:,2], d[:,2]
    def G(ix,iy,iz):
        return grid[(i0[:,0]+ix)%N, (i0[:,1]+iy)%N, (i0[:,2]+iz)%N]
    return (
        G(0,0,0)*wx0*wy0*wz0 + G(1,0,0)*wx1*wy0*wz0 +
        G(0,1,0)*wx0*wy1*wz0 + G(1,1,0)*wx1*wy1*wz0 +
        G(0,0,1)*wx0*wy0*wz1 + G(1,0,1)*wx1*wy0*wz1 +
        G(0,1,1)*wx0*wy1*wz1 + G(1,1,1)*wx1*wy1*wz1
    )

# ---------- PM forces ----------
def pm_gravity_accel(delta, L):
    """Return acceleration field a_grav(x) from density contrast delta (PM, units with 4πGρ̄=1)."""
    N = delta.shape[0]
    dk = 2*np.pi/L
    kx = np.fft.fftfreq(N, d=L/N)*2*np.pi
    ky = np.fft.fftfreq(N, d=L/N)*2*np.pi
    kz = np.fft.rfftfreq(N, d=L/N)*2*np.pi
    KX,KY,KZ = np.meshgrid(kx,ky,kz, indexing='ij')
    K2 = KX*KX + KY*KY + KZ*KZ
    delta_k = np.fft.rfftn(delta)
    # Poisson: Phi_k = - delta_k / K2 (convention unitario); a_k = i k Phi_k
    with np.errstate(divide='ignore', invalid='ignore'):
        Phi_k = np.where(K2>0, -delta_k / K2, 0.0)
    ax_k = 1j*KX*Phi_k; ay_k = 1j*KY*Phi_k; az_k = 1j*KZ*Phi_k
    ax = np.fft.irfftn(ax_k, s=delta.shape).real
    ay = np.fft.irfftn(ay_k, s=delta.shape).real
    az = np.fft.irfftn(az_k, s=delta.shape).real
    return np.stack([ax,ay,az], axis=0).astype(np.float32)

def rms3(v):
    return float(np.sqrt(np.mean(np.sum(v*v, axis=-1))))

# ---------- main integrator ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    cfg = read_yaml(args.cfg)
    L, pos, vel, gKp = load_ics(pick(cfg,'io.ics'), cfg)
    Ngrid = int(pick(cfg,'box.grid_pm_side','grid_pm_side', default=256))
    Npart_side = int(round(pos.shape[0] ** (1/3)))

    z0 = float(pick(cfg,'time.z_ini')); z1 = float(pick(cfg,'time.z_fin'))
    a0, a1 = 1/(1+z0), 1/(1+z1)
    nsteps = int(pick(cfg,'time.nsteps', default=400))
    save_z = list(pick(cfg,'time.save_z', default=[z0, 0.0]))

    # forces
    use_gK = bool(pick(cfg,'forces.enable_gradK', default=True))
    use_grav = bool(pick(cfg,'forces.enable_gravity', default=True))
    dk_strength = float(pick(cfg,'forces.dk_strength', default=1.0))
    growth_model = str(pick(cfg,'forces.dk_growth_model', default='a_over_aini'))
    grav_norm = float(pick(cfg,'forces.grav_norm', default=1.0))

    # integration
    soft = float(pick(cfg,'integration.softening_hMpc', default=0.2))
    cfl  = float(pick(cfg,'integration.adaptive_cfl', default=0.3))

    out_dir = pick(cfg,'io.out_dir'); logs_dir = pick(cfg,'io.logs_dir','telascura/logs')
    ensure_dir(out_dir); ensure_dir(logs_dir)
    log_csv = os.path.join(out_dir, "monitor_Dk.csv")
    with open(log_csv,'w',encoding='utf-8') as f:
        f.write("step,a,z,dt,vmax,dk_rms,deltaMaxOverMean,time_sec\n")

    # Precompute gradK grid (from particle samples) once; scale with growth
    if use_gK and gKp is not None:
        gK_grid = cic_deposit(pos, L, Ngrid, vec=gKp)  # 3,N,N,N
        # normalizza su densità per cella per avere media per-cella (optional)
        count_grid = cic_deposit(pos, L, Ngrid, weights=np.ones(pos.shape[0],np.float32))
        count_grid = np.maximum(count_grid, 1e-6)
        gK_grid /= count_grid[None,...]
        gK_rms0 = np.sqrt(np.mean(gK_grid**2))
    else:
        gK_grid = None; gK_rms0 = 1.0

    # helper growth
    def Dk_of_a(a):
        if growth_model == 'a_over_aini':
            return (a / a0)
        elif growth_model == 'one':
            return 1.0
        else:
            return 1.0

    # save initial
    save_snap(out_dir, "z%04.1f"%(z0), L, pos, vel, a0, z0)

    # timeline in a
    a_vals = np.linspace(a0, a1, nsteps+1)[1:]
    a_prev = a0
    t0 = time.time()

    for istep, a in enumerate(a_vals, start=1):
        z = 1.0/a - 1.0
        # dt "in a" e adattivo su CFL con v_max (si lavora in unità comoventi semplificate)
        da = a - a_prev
        # density
        rho = cic_deposit(pos, L, Ngrid, weights=np.ones(pos.shape[0],np.float32))
        rho /= np.mean(rho); delta = rho - 1.0

        # grav accel (PM)
        accel = np.zeros_like(pos, dtype=np.float32)
        if use_grav:
            gvec = pm_gravity_accel(delta, L) * grav_norm  # 3,N,N,N
            ax = cic_sample(gvec[0], pos, L)
            ay = cic_sample(gvec[1], pos, L)
            az = cic_sample(gvec[2], pos, L)
            accel += np.stack([ax,ay,az], axis=1)

        # gradK accel (static grid scaled by Dk)
        dk_meas = 0.0
        if use_gK and gK_grid is not None:
            scale = dk_strength * Dk_of_a(a)
            ax = cic_sample(gK_grid[0], pos, L) * scale
            ay = cic_sample(gK_grid[1], pos, L) * scale
            az = cic_sample(gK_grid[2], pos, L) * scale
            a_gk = np.stack([ax,ay,az], axis=1)
            accel += a_gk
            dk_meas = float(np.sqrt(np.mean((a_gk*a_gk).sum(axis=1)))) / (gK_rms0 + 1e-30)

        # CFL sul passo in "spazio": dt ~ cfl * (cell / vmax)
        vmax = float(np.max(np.linalg.norm(vel,axis=1))+1e-12)
        cell = L / Ngrid
        dt_cfl = cfl * (cell / max(vmax, 1e-6))
        dt = max(da, 1e-6)  # usa da come base; CFL solo per sicurezza

        # Leapfrog: KDK
        vel += accel * dt
        pos = (pos + vel * dt) % L

        # metriche
        deltaMaxOverMean = float(np.max(rho))
        tsec = time.time() - t0
        with open(log_csv,'a',encoding='utf-8') as f:
            f.write(f"{istep},{a:.6e},{z:.6e},{dt:.3e},{vmax:.3e},{dk_meas:.6e},{deltaMaxOverMean:.3e},{tsec:.3f}\n")

        # snapshot?
        if any(abs(z-sz)<1e-6 for sz in save_z):
            save_snap(out_dir, f"z{z:06.2f}", L, pos, vel, a, z)

        a_prev = a

    # final
    save_snap(out_dir, "final", L, pos, vel, a_prev, 1.0/a_prev - 1.0)

    if args.log:
        with open(args.log,'w',encoding='utf-8') as f:
            f.write(json.dumps({"out_dir":out_dir, "monitor":log_csv}, indent=2))

if __name__ == "__main__":
    main()
