#!/bin/bash
# M6 step 1: recruitment panel = L. crispatus (LC__) + decoy species (DC__), bowtie2 index.
# LC panel: (a) all 11 MAIN human-gut genomes (NCBI CheckM comp>=90, cont<5; T_E_ani_all.csv MAIN & new_label=human gut)
#           (b) all 115 habitat-blind dedup representatives (S2_dedup, ANI>=99.9 single linkage) = diverse non-gut habitats
#           (c) the two patient MAGs (CDM-1 bin.4 = CDM-1, CDM-3 bin.9 = CDM-3)
# Decoys: NCBI reference genomes of the other Lactobacillaceae species detected by MetaPhlAn in CDM-4 (downloaded 2026-10-05).
set -euo pipefail
R=.
O=$R/work_b/M6/CDM-4; W=$O/work
N="nice -n 19 ionice -c3"
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i; next} $1=="CDM-1_crispatus.4" && (($h["MAIN"]=="True" && $h["new_label"]=="human gut") || $h["S2_dedup"]=="True"){print $h["accession"]","$h["new_label"]","$h["MAIN"]","$h["S2_dedup"]}' \
  <(python -c "
import csv,sys
w=csv.writer(sys.stdout)
for r in csv.reader(open('$R/refs/rescue/E/T_E_ani_all.csv')): w.writerow([c.replace(',',';') for c in r])") | sort -u > $O/panel_members.csv
: > $W/panel.fa
while IFS=, read -r acc lab main dd; do
  f=$R/refs/lc_fna/$acc.fna; [ -s "$f" ] || { echo "MISSING $acc" >&2; exit 1; }
  sed "s/^>\([^ ]*\).*/>LC__${acc}__\1/" $f >> $W/panel.fa
done < $O/panel_members.csv
sed 's/^>\([^ ]*\).*/>LC__MAG_CDM1_CDM-1bin4__\1/'  $R/mag_bins/bins2/CDM-1/bin.4.fa >> $W/panel.fa
sed 's/^>\([^ ]*\).*/>LC__MAG_CDM3_CDM3bin9__\1/' $R/mag_bins/bins2/CDM-3/bin.9.fa  >> $W/panel.fa
: > $W/decoy.fa
for f in $W/decoy/ncbi_dataset/data/*/*.fna; do a=$(basename $(dirname $f)); sed "s/^>\([^ ]*\).*/>DC__${a}__\1/" $f >> $W/decoy.fa; done
cat $W/panel.fa $W/decoy.fa > $W/panel_decoy.fa
sha256sum $R/mag_bins/bins2/CDM-1/bin.4.fa $R/mag_bins/bins2/CDM-3/bin.9.fa $W/panel.fa $W/decoy.fa > $O/logs/panel_sha256.txt
grep -c '>' $W/panel.fa $W/decoy.fa; wc -l $O/panel_members.csv; cut -d, -f2 $O/panel_members.csv | sort | uniq -c
$N $CONDA_ENVS/mapping_env/bin/bowtie2-build --threads 4 --seed 20261005 $W/panel_decoy.fa $W/panel_decoy > $O/logs/m6_01_bt2build.log 2>&1
