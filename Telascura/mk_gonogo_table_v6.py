# Leggi topologia
topo_status = "TBD"
topo_detail = "n/a"
if os.path.exists("telascura/metrics/hi256_z000/topology_series.csv"):
    with open("telascura/metrics/hi256_z000/topology_series.csv", 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            n_used = sum(int(r['used']) for r in rows)
            n_outband = sum(1 for r in rows if int(r['used']) and 
                          (float(r['filamentarity']) < float(r['null_lo']) or 
                           float(r['filamentarity']) > float(r['null_hi'])))
            topo_status = "PASS" if n_outband >= 3 else "FAIL"
            topo_detail = f"out-of-band = {n_outband}/{n_used}"

# Nella tabella LaTeX, sostituisci la riga topologia con:
f.write(f"Topology vs null & {topo_status} & {topo_detail} \\\\\n")