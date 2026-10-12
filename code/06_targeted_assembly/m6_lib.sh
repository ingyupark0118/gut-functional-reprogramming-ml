# Shared helpers for the M6 chain (sourced as: source m6_lib.sh <sample>). Parameterised copy of CDM-4/scripts/m6_lib.sh.
# One heavy command at a time, nice 19 / ionice idle, <=4 threads. Never writes into CDM-4/ (panel index etc. are read-only inputs).
export PATH=$CONDA_ENVS/mapping_env/bin:$CONDA_ENVS/megahit_m6/bin:$PATH
R=.
M6=$R/work_b/M6
S=${1:?usage: source m6_lib.sh <sample>}
case $S in Cirrhosis-1|CDM-3|CDM-2|CDM-1) ;; *) echo "m6_lib: sample $S is not in the pre-fixed list" >&2; exit 2;; esac
PY=python
FQ=$($PY -c "
import csv,sys
print([r['fastq'] for r in csv.DictReader(open('$R/work_b/U2_G/tables/T_G2_coverage_Lcrispatus.csv')) if r['sample']=='$S'][0])")
[ -s "$FQ" ] || { echo "m6_lib: FASTQ for $S not found: $FQ" >&2; exit 2; }
O=$M6/$S; W=$O/work
P=$M6/CDM-4/work          # shared, read-only: panel_decoy bowtie2 index, decoy.fa, blastdb/pd, skani_refs, decoy genomes
N="nice -n 19 ionice -c3"
SEED=20261005
BL=$CONDA_ENVS/prokka/bin
PROG=$M6/logs/progress_chain.log
mkdir -p $O/logs $W/.done $O/_discard $M6/logs
load1(){ cut -d' ' -f1 /proc/loadavg; }
guard(){ # priority guard before every step: STOP file -> exit 99; 1-min load > 60 -> wait 5 min; checkm needs MemAvailable >= 50 GB
  local n=$1 l p m why
  while :; do
    [ -e $M6/STOP ] && { echo "$(date +%T) $S $n STOP_file_found_exit_before_step $(load1)" >> $PROG; exit 99; }
    l=$(load1); m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    p=$(pgrep -u user -f "panphlan_map|inStrain profile|m3_02_map_profile|m1_01_map" | wc -l) || true   # pgrep exits 1 when nothing matches; under set -e + pipefail that killed the script (CDM-1 r1_recruit guard, 2026-10-06)
    if awk -v l=$l 'BEGIN{exit !(l>60)}'; then why="load1=$l>60"
    elif [ "$n" = checkm ] && [ "$m" -lt 50 ]; then why="MemAvailable=${m}GB<50GB"
    elif [ "$n" = checkm ] && pgrep -u user -f "checkm lineage_wf" > /dev/null; then why="another_checkm_lineage_wf_running"
    else break; fi
    echo "$(date +%T) $S $n WAIT_5min $why priority_procs=$p" >> $PROG
    sleep 300
  done
  echo "$(date '+%F %T') GUARD $n load1=$l MemAvailable=${m}GB priority_procs=$p" >> $O/logs/commands.log
}
fresh(){ # fresh <step> <path>...: if the step is not done, move leftovers of an interrupted run to _discard (no rm)
  local n=$1 f; shift
  [ -e $W/.done/$n ] && return 0
  for f in "$@"; do [ -e "$f" ] && mv "$f" "$O/_discard/$(date +%Y%m%d_%H%M%S)_$(basename $f)"; done; return 0
}
step(){ # step <name> <command string>: skip if done; guard; log command, wall time, peak RSS, load; done marker; progress line
  local n=$1; shift; local l0 t0 rc=0
  if [ -e $W/.done/$n ]; then echo "$(date '+%F %T') SKIP  $n (done marker)" >> $O/logs/commands.log; return 0; fi
  guard $n
  echo $n > $O/logs/current_step
  l0=$(load1); t0=$(date +%s)
  echo "$(date '+%F %T') START $n :: $*" >> $O/logs/commands.log
  /usr/bin/time -v -o $O/logs/$n.time bash -c "set -euo pipefail; $*" || rc=$?
  if [ $rc -ne 0 ]; then echo "$(date '+%F %T') FAIL  $n rc=$rc" >> $O/logs/commands.log; return $rc; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$n" "$(grep Elapsed $O/logs/$n.time | awk '{print $NF}')" \
    "$(grep 'Maximum resident' $O/logs/$n.time | awk '{print $NF}')" "$l0" "$(load1)" >> $O/logs/steps_timing.tsv
  touch $W/.done/$n
  echo "$(date +%T) $S $n $(( $(date +%s) - t0 )) $(load1)" >> $PROG
  echo "$(date '+%F %T') DONE  $n" >> $O/logs/commands.log
}
