#!/usr/bin/env Rscript
# A2 Step 1 input: 9-KO abundance AND the mapped-community denominator, per sample.
#
# The Dir-sFSS definition fixed in G1 needs
#     relab = KO / sum(mapped community-level features)
# The A2 gate run stored only the 9-KO sums, not the denominator, so the
# ExperimentHub gene_families object is read once more here.
#
# Usage: Rscript extract_9ko_relab.R <study> <EH_id>
suppressPackageStartupMessages({library(ExperimentHub); library(SummarizedExperiment)})

args <- commandArgs(trailingOnly = TRUE)
study <- args[1]; ehid <- args[2]
OUT <- "./dir_sfss"
MAP <- "databases/humann/utility_mapping/map_ko_uniref90.txt.gz"
KOS <- c("K00075","K01512","K03060","K03101","K03116","K03687","K03799","K04042","K10716")

cat("reading UniRef90->KO mapping...\n")
ko2u <- list(); con <- gzfile(MAP, "r")
while (length(l <- readLines(con, n = 1)) > 0) {
  p <- strsplit(l, "\t")[[1]]
  if (p[1] %in% KOS) ko2u[[p[1]]] <- p[-1]
}
close(con)
u2k <- character(); for (k in names(ko2u)) for (u in ko2u[[k]]) u2k[u] <- k
allu <- names(u2k)
cat(sprintf("  %d/%d KOs, %d target UniRef90 ids\n", length(ko2u), length(KOS), length(allu)))

eh <- ExperimentHub()
cat(sprintf("loading %s (%s)...\n", study, ehid))
o <- eh[[ehid]]
m <- if (inherits(o, "SummarizedExperiment")) assay(o) else o
rn <- rownames(m)
cat(sprintf("  rows=%d samples=%d\n", length(rn), ncol(m)))

comm <- !grepl("|", rn, fixed = TRUE)
unmapped <- grepl("^UNMAPPED", rn) | grepl("^UNGROUPED", rn)
keep_den <- comm & !unmapped
cat(sprintf("  community rows=%d, of which UNMAPPED/UNGROUPED=%d, denominator rows=%d\n",
            sum(comm), sum(comm & unmapped), sum(keep_den)))

# denominator: colSums over mapped community rows (chunked to bound memory)
idx <- which(keep_den)
total <- numeric(ncol(m))
step <- 200000L
for (s in seq(1L, length(idx), by = step)) {
  blk <- idx[s:min(s + step - 1L, length(idx))]
  total <- total + colSums(as.matrix(m[blk, , drop = FALSE]))
}

hit <- which(comm & (rn %in% allu))
cat(sprintf("  matched target UniRef90 (community-level)=%d\n", length(hit)))
sub <- as.matrix(m[hit, , drop = FALSE]); rownames(sub) <- rn[hit]
ann <- u2k[rn[hit]]
agg <- rowsum(sub, group = ann)                 # KO x samples

out <- as.data.frame(t(agg))
for (k in KOS) if (!k %in% colnames(out)) out[[k]] <- 0
out <- out[, KOS, drop = FALSE]
out$mapped_total <- total
out$sample_id <- colnames(m)
out <- out[, c("sample_id", KOS, "mapped_total")]

f <- file.path(OUT, paste0(study, "_9ko_relab.tsv"))
write.table(out, f, sep = "\t", quote = FALSE, row.names = FALSE)
cat(sprintf("[written] %s  (%d samples)\n", f, nrow(out)))
cat(sprintf("  mapped_total range: %.4g .. %.4g\n", min(total), max(total)))
