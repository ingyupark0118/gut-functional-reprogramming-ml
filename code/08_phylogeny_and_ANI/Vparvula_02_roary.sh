#!/bin/bash
# Roary 3.13.0 (env roary_env, same env as mag_qc/step4). Original command (replot/fig S10 A-D/mag procedure.txt line 289):
#   roary -f roary_output_parvula -e -n -v prokka_gffs_parvula/*/*.gff
# Only change: -p 4 (threads). All other parameters default (-i 95, -cd 99, MCL inflation 1.5), as assumed in A5.
# Roary has no random-seed option; MCL/CD-HIT are deterministic for fixed input order (inputs passed in sorted order).
# CDM MAG GFFs = original Prokka 1.12 outputs (03_annotation/CDM/V.parvula_CDM*/CDM*.gff).
# Usage: vpr_25_roary.sh <main|sens> <accession list>
set -u
W=./refs/vp_reanalysis
S=$1; LIST=$2
ANN=../03_annotation/CDM
export PATH=$CONDA_ENVS/roary_env/bin:$PATH
cd $W/$S || exit 1
[ -e roary ] && { echo "roary output exists"; exit 1; }
mkdir -p gff
while read a; do [ -s $W/prokka/$a/$a.gff ] || { echo "missing gff $a"; exit 1; }; ln -sf $W/prokka/$a/$a.gff gff/$a.gff; done < $LIST
for m in 1_bin.1 4_bin.2 3_bin.12; do ln -sf $ANN/V.parvula_CDM$m/CDM$m.gff gff/CDM$m.gff; done
LOG=$W/logs/LOG_roary_$S.txt
{ echo "### $(date '+%F %T') set=$S roary $(roary --version 2>&1 | tail -1) n_gff=$(ls gff/*.gff | wc -l) load=$(cut -d' ' -f1 /proc/loadavg)"
  echo "tools: $(blastp -version | head -1); $(cd-hit -h | head -1 | tr -d '\t'); $(mcl --version | head -1); mafft $(mafft --version 2>&1 | head -1)"
  echo "CMD: nice -n 19 roary -f roary -e -n -v -p 4 gff/*.gff"
  sha256sum gff/*.gff; } > $LOG
s=$(date +%s)
nice -n 19 roary -f roary -e -n -v -p 4 gff/*.gff >> $LOG 2>&1
echo "exit=$? $(( $(date +%s)-s ))s $(date '+%F %T')" >> $LOG
cat roary/summary_statistics.txt >> $LOG
