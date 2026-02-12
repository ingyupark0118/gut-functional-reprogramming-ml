#!/bin/bash

# ==============================================================================
# Shotgun Metagenomics Analysis Pipeline
# Steps: 
#   1. QC & Host Removal (KneadData)
#   2. Taxonomic Profiling (MetaPhlAn 4)
#   3. Functional Profiling (HUMAnN 3)
#
# Usage: ./run_metagenomics_pipeline_biobakery.sh
# Note: Please ensure all dependencies (Trimmomatic, Bowtie2, etc.) are installed.
# ==============================================================================

# 1. Configuration: Directories & Databases (Customize this section)
# ------------------------------------------------------------------------------
BASE_DIR=$(pwd)
INPUT_DIR="${BASE_DIR}/raw_data"        # Path to raw .fastq.gz files
OUTPUT_DIR="${BASE_DIR}/results"        # Main output directory

# Database Paths (Replace with your actual paths)
KNEADDATA_DB="/path/to/kneaddata_db_human_genome"
TRIMMOMATIC_PATH="/path/to/Trimmomatic-0.39"

# Parameters
THREADS=16
METAPHLAN_NPROC=32

# Create output subdirectories if they don't exist
mkdir -p "${OUTPUT_DIR}/01_kneaddata"
mkdir -p "${OUTPUT_DIR}/02_metaphlan"
mkdir -p "${OUTPUT_DIR}/03_humann"

echo "Starting Pipeline..."
echo "Input Directory: ${INPUT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"

# 2. Main Loop: Process each sample listed in 'samples.txt'
# ------------------------------------------------------------------------------
# Create a 'samples.txt' file containing sample IDs (e.g., ILDD151) line by line.

while IFS= read -r SAMPLE_ID; do
    
    echo "======================================================================"
    echo "Processing Sample: ${SAMPLE_ID}"
    echo "======================================================================"

    # --------------------------------------------------------------------------
    # Step 1: KneadData (Quality Control & Host DNA Removal)
    # --------------------------------------------------------------------------
    echo "[Step 1] Running KneadData..."
    
    kneaddata \
        --input1 "${INPUT_DIR}/${SAMPLE_ID}_1.fq.gz" \
        --input2 "${INPUT_DIR}/${SAMPLE_ID}_2.fq.gz" \
        --output "${OUTPUT_DIR}/01_kneaddata/${SAMPLE_ID}" \
        --output-prefix "${SAMPLE_ID}" \
        --reference-db "${KNEADDATA_DB}" \
        --trimmomatic "${TRIMMOMATIC_PATH}" \
        --trimmomatic-options "SLIDINGWINDOW:4:20" \
        --trimmomatic-options "MINLEN:50" \
        --serial \
        --run-trf \
        --cat-final-output \
        --threads "${THREADS}"

    # Compress the final cleaned output to save space (matches user workflow)
    echo "Compressing KneadData output..."
    gzip -f "${OUTPUT_DIR}/01_kneaddata/${SAMPLE_ID}/${SAMPLE_ID}.fastq"
    
    CLEAN_READS="${OUTPUT_DIR}/01_kneaddata/${SAMPLE_ID}/${SAMPLE_ID}.fastq.gz"

    # --------------------------------------------------------------------------
    # Step 2: MetaPhlAn (Taxonomic Profiling)
    # --------------------------------------------------------------------------
    echo "[Step 2] Running MetaPhlAn..."

    metaphlan "${CLEAN_READS}" \
        --input_type fastq \
        --nproc "${METAPHLAN_NPROC}" \
        --bowtie2out "${OUTPUT_DIR}/02_metaphlan/${SAMPLE_ID}.bowtie2.bz2" \
        --samout "${OUTPUT_DIR}/02_metaphlan/${SAMPLE_ID}.sam.bz2" \
        -o "${OUTPUT_DIR}/02_metaphlan/${SAMPLE_ID}_taxonomic_profile.tsv"

    # --------------------------------------------------------------------------
    # Step 3: HUMAnN (Functional Profiling)
    # --------------------------------------------------------------------------
    echo "[Step 3] Running HUMAnN..."
    # Note: Using MetaPhlAn output as input for HUMAnN accelerates the process
    
    humann \
        --input "${CLEAN_READS}" \
        --output "${OUTPUT_DIR}/03_humann" \
        --taxonomic-profile "${OUTPUT_DIR}/02_metaphlan/${SAMPLE_ID}_taxonomic_profile.tsv" \
        --o-log "${OUTPUT_DIR}/03_humann/${SAMPLE_ID}.log" \
        --threads "${THREADS}"

    echo "Finished processing ${SAMPLE_ID}"

done < samples.txt

echo "======================================================================"
echo "All samples processed successfully."