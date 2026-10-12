# MaAsLin2 re-run: only the input is re-normalised to the KO sum; settings identical to the original run.
#
# Original run (project/input_maaslin2/maaslin2/maaslin2_plot.txt)
#   input       = key_kogene_dataset.csv  (9-KO, original unit = relative to all gene families)
#   normalization = "NONE", transform = "LOG", analysis_method = "LM"
#   fixed_effects = Group, Age, Sex, BMI, Alcohol, PPI, Statin, Probiotics
#   reference     = Group,NC / Group,DM / Group,Cirrhosis  (3 runs)
#   max_significance = 0.25
# This run: settings as above; only the input is re-normalised to the KO sum (nine columns of project/nested_cv/ko_matrix.csv).
#
# Output: T_P3_4_maaslin2_kosum.csv, T_P3_4_maaslin2_side_by_side.csv,
#       logs/LOG_P3_4_maaslin2.txt, maaslin2_kosum_output/
suppressPackageStartupMessages({library(Maaslin2); library(dplyr)})
P3  <- "./refs"
SRC <- "./input_maaslin2/maaslin2"
A1  <- "./nested_cv"
KOS <- c("K00075","K01512","K03060","K03101","K03116","K03687","K03799","K04042","K10716")
outs <- c(file.path(P3,"T_P3_4_maaslin2_kosum.csv"),
          file.path(P3,"T_P3_4_maaslin2_side_by_side.csv"),
          file.path(P3,"logs/LOG_P3_4_maaslin2.txt"))
if (any(file.exists(outs))) stop("output exists: ", paste(outs[file.exists(outs)], collapse=", "))
logf <- outs[3]; cat("", file=logf)
P <- function(...) { s <- paste0(...); cat(s,"\n"); cat(s,"\n",file=logf,append=TRUE) }

P(strrep("=",78))
P("MaAsLin2 re-run (only the input re-normalised to the KO sum)")
P("settings: normalization=NONE, transform=LOG, analysis_method=LM, max_significance=0.25")
P("covariates: Group, Age, Sex, BMI, Alcohol, PPI, Statin, Probiotics | 3 reference runs")
P(strrep("=",78))

md <- read.csv(file.path(SRC,"metadata.csv"), check.names=FALSE)
# same renaming of the columns as in the original script (maaslin2_plot.txt)
colnames(md) <- c("SampleID","Group","PPI","Statin","Probiotics","Alcohol","Sex","Age","BMI")
md$Group <- factor(md$Group, levels=c("NC","DM","Cirrhosis","CDM"))
dir.create(file.path(P3,"maaslin2_kosum_output"), showWarnings=FALSE, recursive=TRUE)
ko_orig <- read.csv(file.path(SRC,"key_kogene_dataset.csv"), check.names=FALSE)
kosum <- read.csv(file.path(A1,"ko_matrix.csv"), row.names=1, check.names=FALSE)[, KOS, drop=FALSE]
P(paste("metadata", nrow(md), "rows | columns:", paste(names(md), collapse=", ")))

ids <- ko_orig$SampleID
stopifnot(all(ids %in% rownames(kosum)))
maaslin_input <- kosum[ids, KOS, drop=FALSE]
rownames(maaslin_input) <- ids
idcol <- names(md)[1]
maaslin_meta <- md; rownames(maaslin_meta) <- maaslin_meta[[idcol]]
maaslin_meta <- maaslin_meta[ids, , drop=FALSE]
P(paste("input", nrow(maaslin_input), "x", ncol(maaslin_input),
        "| value range", signif(min(maaslin_input[maaslin_input>0]),3), "-", signif(max(maaslin_input),3)))
P(paste("ratio to the original unit (median):",
        round(median(as.matrix(maaslin_input)[as.matrix(ko_orig[,KOS])>0] /
                     as.matrix(ko_orig[,KOS])[as.matrix(ko_orig[,KOS])>0]), 2)))

res <- list()
for (ref in c("NC","DM","Cirrhosis")) {
  P(paste0("\n--- reference = ", ref))
  fit <- Maaslin2(
    input_data = maaslin_input, input_metadata = maaslin_meta,
    output = file.path(P3, "maaslin2_kosum_output", paste0("ref_", ref)),
    fixed_effects = c("Group","Age","Sex","BMI","Alcohol","PPI","Statin","Probiotics"),
    reference = c(paste0("Group,", ref)),
    normalization = "NONE", transform = "LOG", analysis_method = "LM",
    max_significance = 0.25, plot_heatmap = FALSE, plot_scatter = FALSE, cores = 1)
  r <- fit$results
  r <- r[r$metadata == "Group", c("feature","value","coef","stderr","pval","qval")]
  r$comparison <- paste(r$value, "vs", ref); r$ref <- ref
  res[[ref]] <- r
  P(paste("  results", nrow(r), "rows | q<0.25:", sum(r$qval < 0.25), "| q<0.05:", sum(r$qval < 0.05)))
}
new <- bind_rows(res); new$unit <- "KO sum re-normalised"
write.csv(new, outs[1], row.names=FALSE)

old <- read.csv(file.path(SRC,"all_comparisons_summary.csv")); old$unit <- "original unit"
cmp <- merge(old[,c("feature","comparison","coef","qval")],
             new[,c("feature","comparison","coef","qval")],
             by=c("feature","comparison"), suffixes=c("_orig","_kosum"))
cmp$sign_flip <- sign(cmp$coef_orig) != sign(cmp$coef_kosum)
cmp$sig_orig <- cmp$qval_orig < 0.25
cmp$sig_kosum <- cmp$qval_kosum < 0.25
write.csv(cmp[order(cmp$comparison, cmp$feature),], outs[2], row.names=FALSE)

P(paste0("\n[comparison] 9 KO x ", length(unique(cmp$comparison)), " comparisons = ", nrow(cmp), " cells"))
P(paste("  sign flips:", sum(cmp$sign_flip)))
P(paste("  significant at q<0.25: original unit", sum(cmp$sig_orig), "-> KO sum", sum(cmp$sig_kosum)))
P(paste("  significant in both:", sum(cmp$sig_orig & cmp$sig_kosum),
        "| original unit only:", sum(cmp$sig_orig & !cmp$sig_kosum),
        "| KO sum only:", sum(!cmp$sig_orig & cmp$sig_kosum)))
P("\n[cells whose significance changed]")
ch <- cmp[cmp$sig_orig != cmp$sig_kosum, ]
if (nrow(ch)) for (i in seq_len(nrow(ch))) {
  x <- ch[i,]
  P(sprintf("  %-8s %-22s  beta %+.3f -> %+.3f   q %.4f -> %.4f",
            x$feature, x$comparison, x$coef_orig, x$coef_kosum, x$qval_orig, x$qval_kosum))
}
P(paste("\n[written]", paste(basename(outs), collapse=", ")))
