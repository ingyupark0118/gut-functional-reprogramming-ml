#!/bin/bash
# M5: direct CheckM v1.2.4 lineage_wf (same command as work_b/CheckM_recheck/scripts/ck_01_rerun.sh) on a list of FASTA.
# Usage: m5_01_checkm.sh <tag> <list of fasta paths>   -> work/checkm_<tag>/, tables/checkm_<tag>.tsv
set -uo pipefail
export PATH=$CONDA_ENVS/checkm_env/bin:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
M=./work_b/M5
T=$1; LIST=$2; L=$M/logs/LOG_checkm_$T.txt
[ -e $M/work/checkm_$T ] && { echo "exists"; exit 1; }
IN=$M/work/checkm_${T}_in; mkdir -p $IN
while read f; do ln -sf "$f" $IN/$(basename "${f%.*}").fna; done < $LIST
{ echo "### $(date '+%F %T') $(checkm -h 2>&1 | grep -m1 -i 'CheckM v') n=$(ls $IN | wc -l) load=$(cut -d' ' -f1 /proc/loadavg) avail=$(free -g|awk '/Mem:/{print $7}')G"
  echo "CMD: nice -n 19 ionice -c3 checkm lineage_wf -t 4 --pplacer_threads 4 -x fna $IN $M/work/checkm_$T ; checkm qa -t 4 -o 2 --tab_table"
  sha256sum $(cat $LIST); } > $L
s=$(date +%s)
nice -n 19 ionice -c3 checkm lineage_wf -t 4 --pplacer_threads 4 -x fna $IN $M/work/checkm_$T > $M/logs/checkm_$T.out 2>&1
rc=$?
nice -n 19 ionice -c3 checkm qa -t 4 -o 2 --tab_table -f $M/tables/checkm_$T.tsv $M/work/checkm_$T/lineage.ms $M/work/checkm_$T >> $M/logs/checkm_$T.out 2>&1
echo "exit=$rc $(( $(date +%s)-s ))s $(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg)" >> $L
