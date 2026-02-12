#!/bin/bash
# Step 5: StrainPhlAn Analysis for Phocaeicola dorei
# Target Clade: t__SGB1814 (P. dorei) - *Check your SGB ID*

SAMPLE_ID="Your_Sample_Name"

# 5-1. Extract Markers (from MetaPhlAn SAM output)
sample2markers.py \
    --input results/02_metaphlan/${SAMPLE_ID}.sam.bz2 \
    --output_dir results/05_strainphlan/markers \
    --clades t__SGB1814 \
    --min_reads_aligning 1 --min_base_coverage 1 \
    --nproc 32

# 5-2. Build Tree (Run this ONCE after extracting markers for all samples)
# strainphlan \
#    --samples results/05_strainphlan/markers/*.json.bz2 \
#    --clade t__SGB1814 \
#    --output_dir results/05_strainphlan/output \
#    --phylophlan_mode accurate \
#    --mutation_rates