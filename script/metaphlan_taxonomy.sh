#!/bin/bash
# Step 2: Taxonomic Profiling using MetaPhlAn 4
# Input: Cleaned reads from Step 1

SAMPLE_ID="Your_Sample_Name"

metaphlan results/01_kneaddata/${SAMPLE_ID}/${SAMPLE_ID}.fastq.gz \
    --input_type fastq \
    --nproc 32 \
    --bowtie2out results/02_metaphlan/${SAMPLE_ID}.bowtie2.bz2 \
    --samout results/02_metaphlan/${SAMPLE_ID}.sam.bz2 \
    -o results/02_metaphlan/${SAMPLE_ID}_profile.tsv