#!/bin/bash
# Step 3: Functional Profiling using HUMAnN 3
# Note: Providing the taxonomic profile (from Step 2) accelerates the process.

SAMPLE_ID="Your_Sample_Name"

humann \
    --input results/01_kneaddata/${SAMPLE_ID}/${SAMPLE_ID}.fastq.gz \
    --output results/03_humann \
    --taxonomic-profile results/02_metaphlan/${SAMPLE_ID}_profile.tsv \
    --threads 16