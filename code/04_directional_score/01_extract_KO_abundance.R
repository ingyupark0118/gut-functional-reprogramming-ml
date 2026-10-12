#!/usr/bin/env Rscript
# A2-KOsum: per-sample KO-sum denominator for a cMD cohort (pre-fixed spec, 2026-09-30).
#
# Discovery/CNGB/MetaCardis relab = KO / sum(all community KO) (HUMAnN regroup to KO,
# UNMAPPED/UNGROUPED excluded). A2 used sum(all mapped UniRef90) instead. This script
# regroups EVERY community-level UniRef90 to KO with the same mapping file and returns
#     KO_sum = sum_KO sum_{u in KO} abundance(u) = sum_u abundance(u) * n_KO(u)
# (a UniRef90 assigned to several KOs counts once per KO, as humann_regroup_table does).
# The 9-KO numerators are unchanged and are read from <study>_9ko_relab.tsv.
#
# Usage: Rscript extract_ko_sum.R <study> <EH_id>
# Output: <study>_ko_sum.tsv (new file; refuses to overwrite)
suppressPackageStartupMessages({library(ExperimentHub); library(SummarizedExperiment)})

args <- commandArgs(trailingOnly = TRUE)
study <- args[1]; ehid <- args[2]
OUT <- "./dir_sfss"
MAP <- "databases/humann/utility_mapping/map_ko_uniref90.txt.gz"
f <- file.path(OUT, paste0(study, "_ko_sum.tsv"))
if (file.exists(f)) stop("exists, back up and remove first: ", f)

cat("reading UniRef90->KO mapping (all KOs)...\n")
lines <- readLines(gzfile(MAP))
u <- unlist(lapply(strsplit(lines, "\t"), `[`, -1))
mult <- table(u)                                    # n_KO per UniRef90
cat(sprintf("  %d KOs, %d distinct UniRef90, %d with >1 KO\n", length(lines), length(mult), sum(mult > 1)))

o <- ExperimentHub()[[ehid]]
m <- if (inherits(o, "SummarizedExperiment")) assay(o) else o
rn <- rownames(m)
cat(sprintf("%s (%s): rows=%d samples=%d\n", study, ehid, length(rn), ncol(m)))

comm <- !grepl("|", rn, fixed = TRUE)
hit <- which(comm & rn %in% names(mult))
w <- as.numeric(mult[rn[hit]])
cat(sprintf("  community rows=%d, KO-mappable=%d (weights: %d rows with n_KO>1)\n",
            sum(comm), length(hit), sum(w > 1)))

ko_sum <- numeric(ncol(m)); ko_rows_once <- numeric(ncol(m))
step <- 200000L
for (s in seq(1L, length(hit), by = step)) {
  j <- s:min(s + step - 1L, length(hit))
  blk <- as.matrix(m[hit[j], , drop = FALSE])
  ko_sum <- ko_sum + colSums(blk * w[j])
  ko_rows_once <- ko_rows_once + colSums(blk)
}
out <- data.frame(sample_id = colnames(m), KO_sum = ko_sum, KO_mappable_uniref_sum = ko_rows_once)
write.table(out, f, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[written] %s (%d samples)  KO_sum range %.4g .. %.4g\n", f, nrow(out), min(ko_sum), max(ko_sum)))
