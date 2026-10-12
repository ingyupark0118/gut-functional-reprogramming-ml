#!/bin/bash
# T14 panel A chain: prep -> Roary x3 (parallel, 10 threads each) -> gene QC + concatenation x3 -> IQ-TREE main x3 -> relA removal + IQ-TREE x3.
# Reused code: refs/rescue/A/A_20_roary.sh (command), refs/rescue/F/F_10_gene_qc.py (via T14A_10), F_20_iqtree.sh (command), _crosscheck_F/X50_iqtree.sh (command).
# Sets: MAIN25 = 20 refs + 5 targeted; AUX27 = MAIN25 + 2 MAGs; NO_CDM3 = MAIN25 - CDM-3 (24). Flags in work/flags/<step>.{DONE,FAILED}. Re-run = resume.
set -euo pipefail
O=./work_d/fig6_build/A_tree; R=.
A=$R/refs/rescue/A; M5=$R/work_b/M5_ext/work/prokka; RE=$CONDA_ENVS/roary_env/bin; PY=python
cd $O; mkdir -p work/flags logs
say(){ echo "[$(date '+%F %T')] $* load=$(cut -d' ' -f1 /proc/loadavg)" | tee -a logs/chain_steps.log; }
step(){ n=$1; shift; [ -f work/flags/$n.DONE ] && { say "skip $n"; return 0; }; say "start $n"; s=$(date +%s)
  if "$@"; then touch work/flags/$n.DONE; say "DONE $n $(( $(date +%s)-s ))s"; else touch work/flags/$n.FAILED; say "FAILED $n"; touch work/flags/CHAIN.FAILED; exit 1; fi; }
T5="Cirrhosis-1 CDM-1 CDM-4 CDM-2 CDM-3"
members(){ case $1 in
  MAIN25) grep -v '^CDM' $A/genomes22.tsv | cut -f1; for t in $T5; do echo ${t}_Lcrispatus_targeted; done;;
  AUX27) cut -f1 $A/genomes22.tsv; for t in $T5; do echo ${t}_Lcrispatus_targeted; done;;
  NO_CDM3) grep -v '^CDM' $A/genomes22.tsv | cut -f1; for t in $T5; do [ $t = CDM-3 ] || echo ${t}_Lcrispatus_targeted; done;; esac; }
patients(){ members $1 | grep -E '^(CDM|Cirrhosis)' | paste -sd,; }
do_prep(){ for S in MAIN25 AUX27 NO_CDM3; do mkdir -p work/$S/gff work/$S/prokka
  for g in $(members $S); do if [ -d $A/prokka/$g ]; then src=$A/prokka/$g; else src=$M5/$g; fi
    [ -s $src/$g.gff ] && [ -s $src/$g.ffn ]; ln -sfn $src work/$S/prokka/$g; ln -sfn $src/$g.gff work/$S/gff/$g.gff; done
  members $S > work/$S/genomes.txt; sha256sum work/$S/gff/*.gff > work/$S/gff_sha256.txt; done
  [ $(wc -l < work/MAIN25/genomes.txt) -eq 25 ] && [ $(wc -l < work/AUX27/genomes.txt) -eq 27 ] && [ $(wc -l < work/NO_CDM3/genomes.txt) -eq 24 ]; }
roary_one(){ S=$1; cd $O/work/$S; export PATH=$RE:$PATH; [ -e roary ] && mv roary $O/_discard/roary_$S.$(date +%s)
  echo "### $(date '+%F %T') roary $(roary --version 2>&1 | tail -1) n_gff=$(ls gff/*.gff | wc -l) CMD: nice -n 10 roary -f roary -e -n -v -p 10 gff/*.gff" > $O/logs/LOG_roary_$S.txt
  s=$(date +%s); nice -n 10 roary -f roary -e -n -v -p 10 gff/*.gff >> $O/logs/LOG_roary_$S.txt 2>&1
  echo "exit=0 $(( $(date +%s)-s ))s" >> $O/logs/LOG_roary_$S.txt; cat roary/summary_statistics.txt >> $O/logs/LOG_roary_$S.txt; [ -s roary/core_gene_alignment.aln ]; }
par3(){ f=$1; pids=""; for S in MAIN25 AUX27 NO_CDM3; do ( $f $S ) & pids="$pids $!"; done; rc=0; for p in $pids; do wait $p || rc=1; done; return $rc; }
qc_one(){ S=$1; n=$(wc -l < $O/work/$S/genomes.txt); for f in T_F_gene_qc.csv F_concat.fasta; do [ -e $O/work/$S/$f ] && mv $O/work/$S/$f $O/_discard/${S}_$f.$(date +%s); done
  PATH=$RE:$PATH nice -n 10 $PY $O/script/T14A_10_gene_qc.py $O/work/$S $(patients $S) $n > $O/logs/LOG_gene_qc_$S.txt 2>&1; }
iq_one(){ S=$1; cd $O/work/$S; mkdir -p iqtree; ls iqtree/main.* >/dev/null 2>&1 && { mkdir -p $O/_discard/iq_$S.$$; mv iqtree/main.* $O/_discard/iq_$S.$$/; }
  echo "CMD: nice -n 10 iqtree3 -s F_concat.fasta -m MFP -B 1000 -seed 42 -nt 4 --prefix iqtree/main ; $($RE/iqtree3 --version | head -1) ; sha256 $(sha256sum F_concat.fasta)" > $O/logs/LOG_iqtree_main_$S.txt
  nice -n 10 $RE/iqtree3 -s F_concat.fasta -m MFP -B 1000 -seed 42 -nt 4 --prefix iqtree/main > iqtree/main.stdout 2>&1; [ -s iqtree/main.contree ]; }
norela_one(){ S=$1; cd $O/work/$S; [ -e F_concat_noRelA.fasta ] && mv F_concat_noRelA.fasta $O/_discard/${S}_F_concat_noRelA.fasta.$(date +%s)
  $PY $O/script/T14A_15_relA.py $O/work/$S > $O/logs/LOG_relA_$S.txt 2>&1
  [ -s F_concat_noRelA.fasta ] || { echo "relA not in concat -> no noRelA tree" >> $O/logs/LOG_relA_$S.txt; return 0; }
  M=$(grep -m1 'Best-fit model:' iqtree/main.log | sed 's/Best-fit model: \(\S*\) .*/\1/'); [ -n "$M" ]
  ls iqtree/noRelA.* >/dev/null 2>&1 && { mkdir -p $O/_discard/iqn_$S.$$; mv iqtree/noRelA.* $O/_discard/iqn_$S.$$/; }
  echo "CMD: nice -n 10 iqtree3 -s F_concat_noRelA.fasta -m $M -B 1000 -seed 42 -nt 4 --prefix iqtree/noRelA (model = best-fit of main, as in X50_iqtree.sh)" >> $O/logs/LOG_relA_$S.txt
  nice -n 10 $RE/iqtree3 -s F_concat_noRelA.fasta -m $M -B 1000 -seed 42 -nt 4 --prefix iqtree/noRelA > iqtree/noRelA.stdout 2>&1; [ -s iqtree/noRelA.contree ]; }
say "CHAIN START pid=$$ nproc=$(nproc)"
step prep do_prep
step roary par3 roary_one
step qc par3 qc_one
step iqtree_main par3 iq_one
step norelA par3 norela_one
touch work/flags/CHAIN.DONE; say "CHAIN DONE"
