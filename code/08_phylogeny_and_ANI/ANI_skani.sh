#!/bin/bash
# Angle E step 2: skani ANI. One computation at a time, nice 19 / ionice idle, 4 threads.
set -euo pipefail
P3=./refs
E=$P3/rescue/E; W=$E/work; SK=$P3/tools/skani
PY=$CONDA_ENVS/omics-agent/bin/python
M_CDM1="$P3/../mag_bins/bins2/CDM-1/bin.4.fa"
M_CDM3="$P3/../mag_bins/bins2/CDM-3/bin.9.fa"
for f in $W/mag_vs_all.tsv $W/main_triangle.tsv $W/mag_vs_mag.tsv; do [ -e "$f" ] && { echo "exists: $f"; exit 1; }; done
echo "== start $(date '+%F %T')  load: $(cat /proc/loadavg)"
echo "== skani: $($SK --version)   sha256 $(sha256sum $SK | cut -d' ' -f1)"
ln -sfn "$(readlink -f "$M_CDM1")"  $W/CDM-1_crispatus.4.fna
ln -sfn "$(readlink -f "$M_CDM3")" $W/CDM-3_crispatus.9.fna
echo "== MAG sha256 (source, and tree7A_42 copy for identity check)"
sha256sum "$M_CDM1" "$M_CDM3" $P3/tree7A_42/input_genomes_Lc/CDM-1_crispatus.4.fna $P3/tree7A_42/input_genomes_Lc/CDM-3_crispatus.9.fna
printf '%s\n' $W/CDM-1_crispatus.4.fna $W/CDM-3_crispatus.9.fna > $W/mags.txt
# reference lists from the reclass table (python, no shell glob)
$PY - <<'PYE'
import pandas as pd, os
P3="./refs"; W=f"{P3}/rescue/E/work"
t=pd.read_csv(f"{P3}/rescue/E/T_E_habitat_reclass.csv")
allp=[f"{P3}/lc_fna/{a}.fna" for a in t.accession]; assert all(os.path.getsize(p)>0 for p in allp)
open(f"{W}/refs_all1185.txt","w").write("\n".join(allp)+"\n")
m=t[(t.ncbi_checkm_comp>=90)&(t.ncbi_checkm_cont<5)]
open(f"{W}/refs_main.txt","w").write("\n".join(f"{P3}/lc_fna/{a}.fna" for a in m.accession)+"\n")
print("refs all",len(allp),"main",len(m))
PYE
run(){ echo "== CMD: $*"; /usr/bin/time -f "   wall %es  maxRSS %MkB" nice -n 19 ionice -c3 "$@"; }
run $SK dist -t 4 --ql $W/mags.txt --rl $W/refs_all1185.txt --min-af 0 -s 0 --ci -n 5000 -o $W/mag_vs_all.tsv
run $SK dist -t 4 --ql $W/mags.txt --rl $W/mags.txt --min-af 0 -s 0 --ci -o $W/mag_vs_mag.tsv
run $SK triangle -t 4 -l $W/refs_main.txt --min-af 0 -s 0 --sparse -o $W/main_triangle.tsv
wc -l $W/mag_vs_all.tsv $W/mag_vs_mag.tsv $W/main_triangle.tsv
sha256sum $W/refs_all1185.txt $W/refs_main.txt $W/mag_vs_all.tsv $W/mag_vs_mag.tsv $W/main_triangle.tsv
echo "== end $(date '+%F %T')  load: $(cat /proc/loadavg)"
