#!/bin/bash
# M6 steps 3-5 (parameterised copy of CDM-4/scripts/m6_03_cov_blast.sh + the contig-table and filter calls that were run by hand for CDM-4):
# per-contig depth (recruited reads -> round-2 contigs, end-to-end), local blastn of round-1/2 contigs vs panel+decoy (shared DB),
# local blastn of round-2 contigs vs the other bins of the same sample, contig tables, PRE-FIXED contig filter (DESIGN_prefixed_remaining.md).
# usage: m6_03_cov_blast.sh <sample>
set -euo pipefail
source ./work_b/M6/scripts/m6_lib.sh "$1"
A=$W/asm_r2/final.contigs.fa
X=$R/mag_bins/bins2
case $S in   # other bins of the same sample = all bins2 bins not classified as L. crispatus by GTDB-Tk (DESIGN section 4)
  Cirrhosis-1)  OB="" ;;
  CDM-3) OB=$(for i in 1 2 3 4 5 6 7 8 10 11 12 13 14; do echo $X/CDM-3/bin.$i.fa; done) ;;
  CDM-2) OB="$X/CDM-2/bin.1.fa" ;;
  CDM-1)  OB=$(for i in 1 2 3 5 6 7 8 9; do echo $X/CDM-1/bin.$i.fa; done) ;;
esac
BF="-task megablast -num_threads 4 -perc_identity 90 -evalue 1e-10 -max_target_seqs 20000 -max_hsps 50 -outfmt '6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen'"
[ -s $P/blastdb/pd.nsq ] || { echo "shared blast DB missing: $P/blastdb/pd" >&2; exit 1; }
step ev_map "bowtie2-build --threads 4 --seed $SEED $A $W/asm_r2_idx > $O/logs/ev_bt2build.log 2>&1;
  $N bowtie2 -p 4 --seed $SEED --sensitive -x $W/asm_r2_idx -U $W/r2_reads.fq 2> $O/logs/ev_bt2.log | $N samtools sort -@ 2 -m 768M -T $W/tmp_ev -o $W/ev.bam -; samtools index $W/ev.bam;
  samtools coverage $W/ev.bam > $W/ev_coverage.tsv"
step ev_blast "for r in r1 r2; do $N $BL/blastn $BF -query $W/asm_\$r/final.contigs.fa -db $P/blastdb/pd | awk '\$4>=200' > $W/blast_\$r.tsv; done"
step ob_blast ": > $W/otherbins.fa; : > $W/blast_r2_otherbins.tsv; echo '$OB' | tr ' ' '\n' > $O/logs/otherbins_list.txt;
  for f in $(echo $OB); do [ -s \$f ]; b=\$(basename \$(dirname \$f))\$(basename \$f .fa | tr -d .); sed \"s/^>\([^ ]*\).*/>OB__\${b}__\1/\" \$f >> $W/otherbins.fa; done;
  if [ -s $W/otherbins.fa ]; then mkdir -p $W/blastdb_ob; $BL/makeblastdb -in $W/otherbins.fa -dbtype nucl -out $W/blastdb_ob/ob > $O/logs/ob_makeblastdb.log;
    $N $BL/blastn $BF -query $A -db $W/blastdb_ob/ob | awk '\$4>=200' > $W/blast_r2_otherbins.tsv; fi"
step ctab "$PY $M6/scripts/m6_04_contig_table.py $W/asm_r1/final.contigs.fa $W/blast_r1.tsv $W/contigs_r1.csv > $O/logs/m6_04_contig_table.out;
  $PY $M6/scripts/m6_04_contig_table.py $A $W/blast_r2.tsv $W/contigs_r2.csv $W/ev_coverage.tsv >> $O/logs/m6_04_contig_table.out"
fresh filter $O/T_M6_${S}_contigs.csv $O/${S}_Lcrispatus_targeted.fa $O/${S}_targeted_unplaced_NOT_genome.fa $O/${S}_targeted_excluded_NOT_genome.fa
step filter "cd $O; $PY $M6/scripts/m6_05_filter.py $S > $O/logs/m6_05_filter.out"
