#!/bin/bash
# Step 1: Quality Control using KneadData
# Replace 'Your_Sample_Name' with the actual sample ID

SAMPLE_ID="Your_Sample_Name"  #

kneaddata \
    --input1 raw_data/${SAMPLE_ID}_1.fastq.gz \
    --input2 raw_data/${SAMPLE_ID}_2.fastq.gz \
    --output results/01_kneaddata/${SAMPLE_ID} \
    --output-prefix ${SAMPLE_ID} \
    --reference-db databases/kneaddata_db_human_genome \
    --trimmomatic databases/Trimmomatic-0.39 \
    --trimmomatic-options "SLIDINGWINDOW:4:20 MINLEN:50" \
    --serial --run-trf --cat-final-output --threads 16