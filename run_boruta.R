# ==============================================================================
# Boruta Feature Selection Pipeline
#
# Description: This script performs feature selection using the Boruta algorithm
#              for multi-class classification datasets.
#
# Input: Processed CSV file (rows: samples, columns: features + 'Group')
# Output: 1. CSV file containing statistics of confirmed important features
#         2. PDF plot visualizing the importance of confirmed features
#
# Dependencies: Boruta
# ==============================================================================

# 1. Setup & Configuration -----------------------------------------------------
# Install package if not exists
if(!require(Boruta)) install.packages("Boruta")
library(Boruta)

# Clear environment
rm(list = ls())

# Set Parameters (Edit here)
PROJECT_DIR  <- getwd()                  # 현재 작업 경로 기준
DATA_DIR     <- file.path(PROJECT_DIR, "data", "processed")  # 데이터 폴더
RESULT_DIR   <- file.path(PROJECT_DIR, "results", "boruta")  # 결과 폴더
GROUP_NAME   <- ""                       # 파일명 접두사 (필요시)
DATA_NAME    <- "combined"               # 데이터셋 이름
FILENAME     <- paste0(GROUP_NAME, DATA_NAME, "_disc.csv")

# Create output directory if it doesn't exist
if(!dir.exists(RESULT_DIR)) dir.create(RESULT_DIR, recursive = TRUE)

# Set Seed for Reproducibility (Crucial for Boruta/RandomForest)
set.seed(42)

# 2. Load Data -----------------------------------------------------------------
input_path <- file.path(DATA_DIR, FILENAME)

# Check if file exists
if(!file.exists(input_path)){
  stop(paste("Error: File not found at", input_path))
}

message(paste("Loading dataset:", FILENAME))
dataset <- read.csv(input_path, stringsAsFactors = FALSE)

# Ensure 'Group' is a factor
if("Group" %in% colnames(dataset)){
  dataset$Group <- as.factor(dataset$Group)
} else {
  stop("Error: 'Group' column not found in the dataset.")
}

# 3. Run Boruta Algorithm ------------------------------------------------------
message("Running Boruta Feature Selection...")

boruta_output <- Boruta(
  Group ~ ., 
  data = dataset, 
  pValue = 0.05,      # Significance level
  mcAdj = TRUE,       # Multiple comparison adjustment (Bonferroni)
  maxRuns = 1000,     # Maximum number of importance source runs
  doTrace = 2         # Verbosity level (0=none, 1=some, 2=detailed)
)

# Tentative Fix (Optional: resolve tentative attributes using a rough fix)
# boruta_output <- TentativeRoughFix(boruta_output) 

print(boruta_output)

# 4. Extract & Save Important Features -----------------------------------------
message("Extracting and saving important features...")

# Get statistics for all attributes
boruta_stats <- attStats(boruta_output)

# Filter only 'Confirmed' attributes
confirmed_stats <- boruta_stats[boruta_stats$decision == "Confirmed", ]

# Sort by Mean Importance (Descending)
confirmed_stats_sorted <- confirmed_stats[order(-confirmed_stats$meanImp), ]

# Save to CSV
out_csv_name <- paste0(GROUP_NAME, DATA_NAME, "_importance_confirmed.csv")
write.csv(confirmed_stats_sorted, file.path(RESULT_DIR, out_csv_name), row.names = TRUE)

message(paste("Saved importance table to:", out_csv_name))

# 5. Visualization -------------------------------------------------------------
message("Generating Boruta Plot...")

out_pdf_name <- paste0(GROUP_NAME, DATA_NAME, "_boruta_plot.pdf")
pdf_path <- file.path(RESULT_DIR, out_pdf_name)

# Plotting specific attributes (Only Confirmed) using the original model
# Note: Instead of re-running Boruta on subset (which is biased), 
# we visualize the Z-scores from the original run for confirmed features.

confirmed_features <- rownames(confirmed_stats_sorted)

if(length(confirmed_features) > 0){
  cairo_pdf(file = pdf_path, width = 10, height = 8)
  
  # Adjust margins for variable names (Bottom, Left, Top, Right)
  par(mar = c(12, 5, 4, 2) + 0.1)
  
  plot(
    boruta_output, 
    which = confirmed_features,  # Plot only confirmed features
    xlab = "", 
    main = "Boruta Feature Importance (Confirmed Only)",
    las = 2,                     # Rotate axis labels
    cex.axis = 0.8               # Adjust axis text size
  )
  
  # Add Legend
  legend(
    "topleft", 
    legend = c("Confirmed", "Shadow Max", "Shadow Mean", "Shadow Min"), 
    fill = c("green", "blue", "blue", "blue"),
    cex = 0.8
  )
  
  dev.off()
  message(paste("Saved plot to:", out_pdf_name))
  
} else {
  warning("No confirmed features found. Plot generation skipped.")
}

message("Analysis Completed Successfully.")