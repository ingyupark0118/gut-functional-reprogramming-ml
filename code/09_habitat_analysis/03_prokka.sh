#!/bin/bash
# M5: Prokka 1.12 (env prokka_new), same command as refs/rescue/A/A_10_prokka.sh and refs/vp_reanalysis/vpr_20_prokka.sh
# (prokka --outdir X --prefix X --quiet --cpus 4 X.fna; default locus tag). Reuses an existing Prokka 1.12 GFF only where
# m5_02_select.py verified that the FASTA sha256 equals the one logged when it was annotated (column 3 of the list).
# Usage: m5_10_prokka.sh <lc|vp>
M=./work_b/M5; S=$1
export PATH=$CONDA_ENVS/prokka_new/bin:$PATH
cd $M; mkdir -p work/prokka_$S work/gff_$S
LOG=logs/LOG_prokka_$S.txt
echo "### $(date '+%F %T') $(prokka --version 2>&1 | tail -1) load=$(cut -d' ' -f1 /proc/loadavg)" >> $LOG
while IFS=$'\t' read a f g; do
  if [ "$g" != "-" ]; then ln -sfn "$g" work/gff_$S/$a.gff; echo "reuse $a <- $g sha256_gff=$(sha256sum "$g" | cut -d' ' -f1) sha256_fna=$(sha256sum "$f" | cut -d' ' -f1)" >> $LOG; continue; fi
  if [ ! -e work/prokka_$S/$a ]; then
    s=$(date +%s)
    echo "CMD: nice -n 19 ionice -c3 prokka --outdir work/prokka_$S/$a --prefix $a --quiet --cpus 4 $f" >> $LOG
    nice -n 19 ionice -c3 prokka --outdir work/prokka_$S/$a --prefix $a --quiet --cpus 4 "$f" >> $LOG 2>&1 < /dev/null
    echo "exit=$? $a $(( $(date +%s)-s ))s CDS=$(grep -c $'\tCDS\t' work/prokka_$S/$a/$a.gff 2>/dev/null) sha256_fna=$(sha256sum "$f" | cut -d' ' -f1)" >> $LOG
  else echo "skip $a (exists)" >> $LOG; fi
  ln -sfn $M/work/prokka_$S/$a/$a.gff work/gff_$S/$a.gff
done < work/${S}_genomes.tsv
echo "### done $(date '+%F %T') n_gff=$(ls work/gff_$S/*.gff | wc -l) load=$(cut -d' ' -f1 /proc/loadavg)" >> $LOG
