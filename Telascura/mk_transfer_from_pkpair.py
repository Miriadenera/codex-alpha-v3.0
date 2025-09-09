#!/usr/bin/env python3
import argparse, csv, math, os

ap = argparse.ArgumentParser()
ap.add_argument("--pk-ini", required=True)      # es: telascura/metrics/hi256_z1000/pk_real.csv
ap.add_argument("--pk-fin", required=True)      # es: telascura/metrics/hi256_z000/pk_real.csv
ap.add_argument("--monitor", required=False)    # es: telascura/runs/hi256_seedA/monitor_DK_fix.csv
ap.add_argument("--out-prefix", default=".")    # dove salvare i CSV output
args = ap.parse_args()

def read_pk(path):
    def pick(h, *keys):
        hlow = [x.lower() for x in h]
        for k in keys:
            for j, name in enumerate(hlow):
                if k in name: return h[j]
        return None
    rows=[]
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        h=r.fieldnames or []
        ck = pick(h, "k")
        cp = pick(h, "pk","p_k","p(k)","power")
        cn = pick(h, "nmodes","modes","nmode","n")
        for row in r:
            try:
                k=float(row[ck]); p=float(row[cp])
                n=int(float(row[cn])) if cn and row.get(cn,"")!="" else 0
            except Exception:
                continue
            if math.isfinite(k) and math.isfinite(p):
                rows.append((k,p,n))
    return rows

def keyed(rows):
    # usa k arrotondato per unire i due spettri
    d={}
    for k,p,n in rows:
        kk = round(k, 10)
        d[kk]=(p,n)
    return d

ini = keyed(read_pk(args.pk_ini))
fin = keyed(read_pk(args.pk_fin))

common = sorted(set(ini.keys()) & set(fin.keys()))
out_transfer = os.path.join(args.out_prefix, "transfer_curves.csv")
with open(out_transfer,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["k","T_tel","null_lo","null_hi","used"])
    for kk in common:
        p0,n0 = ini[kk]
        p1,n1 = fin[kk]
        used = 1 if (p0>0 and p1>0 and min(n0,n1)>=10) else 0
        T = (p1/p0) if (p0>0 and p1>0) else ""
        w.writerow([kk, T, "", "", used])
print("[WRITE]", out_transfer)

# growth series (facoltativo, se hai il monitor)
if args.monitor:
    out_growth = os.path.join(args.out_prefix, "growth_series.csv")
    def pick(h, *keys):
        hlow = [x.lower() for x in h]
        for k in keys:
            for j, name in enumerate(hlow):
                if k in name: return h[j]
        return None
    with open(args.monitor, newline="") as f, open(out_growth,"w",newline="") as g:
        r=csv.DictReader(f)
        h=r.fieldnames or []
        ca = pick(h,"a")
        cdk= pick(h,"dk_rms","d_k","dk","d k")
        w=csv.writer(g); w.writerow(["a","dk_rms"])
        for row in r:
            try:
                a=float(row[ca]); dk=float(row[cdk])
                if math.isfinite(a) and math.isfinite(dk):
                    w.writerow([a,dk])
            except Exception:
                continue
    print("[WRITE]", out_growth)
