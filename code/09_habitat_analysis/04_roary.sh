#!/bin/bash
# M5: Roary 3.13.0 (env roary_env), parameters of refs/rescue/A/A_20_roary.sh: roary -f roary -e -n -v -p 4 gff/*.gff
# Usage: m5_20_roary.sh <lc|vp>
set -u
M=./work_b/M5; S=$1
export PATH=$CONDA_ENVS/roary_env/bin:$PATH
cd $M/work || exit 1
[ -e roary_$S ] && { echo "roary output exists"; exit 1; }
n=$(wc -l < ${S}_genomes.tsv); [ $(ls gff_$S/*.gff | wc -l) -eq $n ] || { echo "gff count != $n"; exit 1; }
for g in gff_$S/*.gff; do [ -s $g ] || { echo "empty $g"; exit 1; }; done
LOG=$M/logs/LOG_roary_$S.txt
{ echo "### $(date '+%F %T') roary $(roary --version 2>&1 | tail -1) n_gff=$n load=$(cut -d' ' -f1 /proc/loadavg)"
  echo "tools: $(blastp -version | head -1); $(cd-hit -h | head -1 | tr -d '\t'); $(mcl --version | head -1); mafft $(mafft --version 2>&1 | head -1)"
  echo "design sha256 before roary: $(sha256sum $M/DESIGN_M5_prefixed.md)"
  echo "CMD: nice -n 19 ionice -c3 roary -f roary_$S -e -n -v -p 4 gff_$S/*.gff"
  sha256sum gff_$S/*.gff; } > $LOG
s=$(date +%s)
nice -n 19 ionice -c3 roary -f roary_$S -e -n -v -p 4 gff_$S/*.gff >> $LOG 2>&1
echo "exit=$? $(( $(date +%s)-s ))s $(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg)" >> $LOG
cat roary_$S/summary_statistics.txt >> $LOG
