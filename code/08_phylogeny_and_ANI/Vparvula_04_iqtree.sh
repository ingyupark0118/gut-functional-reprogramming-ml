#!/bin/bash
# E4 step 20: IQ-TREE on a concatenated single-copy core alignment (constant sites kept). Usage: E4_20_iqtree.sh <alignment> <prefix>   (copied from F_20_iqtree.sh)
set -u
E=./work_a/E4_fig7B; IQ=$CONDA_ENVS/roary_env/bin/iqtree3
cd $E || exit 1
ALN=$1; P=$2; LOG=logs/LOG_E4_20_iqtree_$P.txt
[ -e iqtree/$P.treefile ] && { echo "exists"; exit 1; }
{ echo "### $(date '+%F %T') $($IQ --version | head -1) load=$(cut -d' ' -f1 /proc/loadavg)"
  echo "sha256 $(sha256sum $ALN)"; echo "design sha256: $(sha256sum DESIGN_E4_prefixed.md)"
  echo "CMD: nice -n 19 ionice -c3 $IQ -s $ALN -m MFP -B 1000 -seed 42 -nt 4 --prefix iqtree/$P"; } > $LOG
s=$(date +%s)
nice -n 19 ionice -c3 $IQ -s $ALN -m MFP -B 1000 -seed 42 -nt 4 --prefix iqtree/$P > iqtree/$P.stdout 2>&1
echo "exit=$? $(( $(date +%s)-s ))s $(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg)" >> $LOG
