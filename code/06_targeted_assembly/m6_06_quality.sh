#!/bin/bash
# M6 step 6-7 (parameterised copy of CDM-4/scripts/m6_06_quality.sh): CheckM v1.2.4 lineage_wf, skani 0.3.2 ANI (replaces GTDB-Tk),
# Prodigal gene count, barrnap/ARAGORN (rRNA/tRNA for MIMAG), SNV density, iterative-round gain, tables.
# skani references: 122 panel genomes + 2 patient MAGs + decoys + type strains (as CDM-4) + every finished M6 targeted assembly (CDM-4 and earlier samples).
# usage: m6_06_quality.sh <sample>
set -euo pipefail
source ./work_b/M6/scripts/m6_lib.sh "$1"
SK=$R/refs/tools/skani; F=$O/${S}_Lcrispatus_targeted.fa
[ -s $F ] || { echo "missing $F" >&2; exit 1; }
fresh checkm $W/checkm_in $W/checkm_out
step checkm "mkdir -p $W/checkm_in; cp $F $W/checkm_in/; cp $W/asm_r2/final.contigs.fa $W/checkm_in/${S}_round2_unfiltered.fa; cp $W/asm_r1/final.contigs.fa $W/checkm_in/${S}_round1_unfiltered.fa;
  export PATH=$CONDA_ENVS/checkm_env/bin:\$PATH LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6; $N checkm lineage_wf -t 4 --pplacer_threads 1 -x fa $W/checkm_in $W/checkm_out > $O/logs/checkm.out 2>&1; checkm qa -o 2 --tab_table -f $W/checkm_qa.tsv $W/checkm_out/lineage.ms $W/checkm_out >> $O/logs/checkm.out 2>&1"
step skani "cut -d, -f1 $M6/CDM-4/panel_members.csv | sed 's#^#$R/refs/lc_fna/#; s#\$#.fna#' > $W/skani_reflist.txt;
  ls $P/skani_refs/*.fa $P/decoy/ncbi_dataset/data/*/*.fna >> $W/skani_reflist.txt; for a in \$(cat $R/refs/acc_type.txt); do f=$R/refs/lc_fna/\$a.fna; [ -s \$f ] && ! grep -q \$a $W/skani_reflist.txt && echo \$f >> $W/skani_reflist.txt; done;
  ls $M6/*/*_Lcrispatus_targeted.fa | grep -v '/$S/' >> $W/skani_reflist.txt;
  $N $SK dist -q $F --rl $W/skani_reflist.txt --min-af 0 -s 0 --ci -t 4 -o $W/skani_final_vs_refs.tsv 2> $O/logs/skani.log"
step prodigal "$N $CONDA_ENVS/checkm_env/bin/prodigal -i $F -a $W/final.faa -o $W/final.gff -f gff -p single > $O/logs/prodigal.log 2>&1; grep -c '>' $W/final.faa > $W/final_gene_count.txt"
step rrna "export PATH=$BL:\$PATH; $N barrnap --kingdom bac --threads 4 $F > $W/rrna.gff 2> $O/logs/barrnap.log; $N aragorn -t -gcbact -w -o $W/trna.txt $F"
step snv "grep '>' $F | sed 's/.*orig=\([^ ]*\).*/\1/' > $W/keep_orig.txt; samtools faidx $W/asm_r2/final.contigs.fa; $N samtools mpileup -f $W/asm_r2/final.contigs.fa -q 20 -Q 20 -d 100000 $W/ev.bam 2> $O/logs/mpileup.log | $PY $M6/scripts/m6_snv.py $W/keep_orig.txt $W/snv_density.tsv"
step gain_blast "$N $BL/blastn -task megablast -query $F -subject $W/asm_r1/final.contigs.fa -perc_identity 95 -evalue 1e-10 -outfmt '6 qseqid sseqid pident length qstart qend sstart send evalue bitscore qlen' | awk '\$4>=200' > $W/blast_final_vs_r1.tsv"
step tables "cd $O; $PY $M6/scripts/m6_07_tables.py $S > $O/logs/m6_07_tables.out"
