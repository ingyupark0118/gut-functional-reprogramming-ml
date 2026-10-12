#!/bin/bash
# M6 step 2 (parameterised copy of CDM-4/scripts/m6_02_recruit_assemble.sh; commands unchanged, only paths):
# round-1 recruitment (shared panel+decoy index, competitive) -> MEGAHIT -> round-2 recruitment on round-1 contigs (+decoy) -> MEGAHIT.
# bowtie2 2.5.4 local mode, --score-min C,100; MEGAHIT v1.2.9 --presets meta-sensitive --min-contig-len 1000, single-read mode (-r).
# usage: m6_02_recruit_assemble.sh <sample>
set -euo pipefail
source ./work_b/M6/scripts/m6_lib.sh "$1"
BT="bowtie2 -p 4 --seed $SEED --local --sensitive-local --score-min C,100 --no-unal"
MH="megahit --presets meta-sensitive --min-contig-len 1000 -t 4 -m 16000000000"
# ---- round 1
step r1_map "zcat $FQ | $N $BT -x $P/panel_decoy -U - 2> $O/logs/r1_bt2.log | $N samtools view -b -o $W/r1.bam -"
step r1_recruit "samtools view $W/r1.bam | awk -v OFS='\t' '{print \$1,substr(\$3,1,2)}' > $W/r1_hits.tsv; zcat $FQ | $N $PY $M6/scripts/m6_recruit.py $W/r1_reads.fq $O/logs/r1_recruit_stats.txt $W/r1_hits.tsv"
fresh r1_megahit $W/asm_r1
step r1_megahit "$N $MH -r $W/r1_reads.fq -o $W/asm_r1 > $O/logs/r1_megahit.out 2>&1"
# bait = round-1 contigs with k-mer multiplicity >= 0.3 x length-weighted median (drops low-depth co-recruited fragments)
step r2_bait "awk '/^>/{split(\$3,m,\"=\"); split(\$4,l,\"=\"); print m[2]\"\t\"l[2]}' $W/asm_r1/final.contigs.fa | sort -k1,1g > $W/r1_multi_len.tsv;
  MED=\$(awk '{t+=\$2} END{print t/2}' $W/r1_multi_len.tsv | xargs -I{} awk -v h={} '{c+=\$2; if(c>=h){print \$1; exit}}' $W/r1_multi_len.tsv); echo \"r1_lenweighted_median_multi=\$MED\" > $O/logs/r2_bait_stats.txt;
  awk -v med=\$MED '/^>/{split(\$3,m,\"=\"); keep=(m[2]>=0.3*med); if(keep){n++; print \">LC__r1__\" substr(\$1,2)} else d++; next} keep{print; b+=length(\$0)} END{print \"bait_contigs=\"n\"\nbait_bp=\"b\"\ndropped_lowcov_contigs=\"d+0 >> \"$O/logs/r2_bait_stats.txt\"}' $W/asm_r1/final.contigs.fa > $W/r2_bait.fa;
  [ -s $W/r2_bait.fa ];
  cat $W/r2_bait.fa $P/decoy.fa > $W/r2_bait_decoy.fa; $N bowtie2-build --threads 4 --seed $SEED $W/r2_bait_decoy.fa $W/r2_bait_decoy > $O/logs/r2_bt2build.log 2>&1"
# ---- round 2 (all reads against round-1 contigs + decoys)
step r2_map "zcat $FQ | $N $BT -x $W/r2_bait_decoy -U - 2> $O/logs/r2_bt2.log | $N samtools view -b -o $W/r2.bam -"
step r2_recruit "samtools view $W/r2.bam | awk -v OFS='\t' '{print \$1,substr(\$3,1,2)}' > $W/r2_hits.tsv; zcat $FQ | $N $PY $M6/scripts/m6_recruit.py $W/r2_reads.fq $O/logs/r2_recruit_stats.txt $W/r1_hits.tsv $W/r2_hits.tsv"
fresh r2_megahit $W/asm_r2
step r2_megahit "$N $MH -r $W/r2_reads.fq -o $W/asm_r2 > $O/logs/r2_megahit.out 2>&1"
