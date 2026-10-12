#!/bin/bash
# Prokka 1.12 (env prokka_new = the version/env of the original 03_annotation logs) on reference genomes.
# Original command (03_annotation/*/*.log): prokka --outdir prokka_gffs_parvula/<name> --prefix <name> --quiet <name>.fna
# Only change: --cpus 4 (original used the default 8); prefix = accession. No random seed in Prokka (deterministic).
# Usage: vpr_20_prokka.sh <accession list file> ; output prokka/<acc>/ ; never overwrites.
W=./refs/vp_reanalysis
FNA=./refs/vp_refs/fna
export PATH=$CONDA_ENVS/prokka_new/bin:$PATH
cd $W
LOG=logs/LOG_prokka.txt
echo "### $(date '+%F %T') list=$1 $(prokka --version 2>&1 | tail -1) load=$(cut -d' ' -f1 /proc/loadavg)" >> $LOG
while read a; do
  [ -e prokka/$a ] && { echo "skip $a (exists)" >> $LOG; continue; }
  s=$(date +%s)
  echo "CMD: nice -n 19 prokka --outdir prokka/$a --prefix $a --quiet --cpus 4 $FNA/$a.fna" >> $LOG
  nice -n 19 prokka --outdir prokka/$a --prefix $a --quiet --cpus 4 $FNA/$a.fna >> $LOG 2>&1
  echo "exit=$? $a $(( $(date +%s)-s ))s CDS=$(grep -c $'\tCDS\t' prokka/$a/$a.gff 2>/dev/null) sha256_fna=$(sha256sum $FNA/$a.fna | cut -d' ' -f1)" >> $LOG
done < "$1"
echo "### done $(date '+%F %T')" >> $LOG
