# A1-v3 step 1: ConQuR on the raw MetaPhlAn species matrix (fixed spec, 2026-09-30).
#
#   input   : species_s_matrix.csv (1,338 s__, 149, relab = A1-v2 input)
#   counts  : round(relab * estimated_reads_mapped_to_known_clades)  (ConQuR rounds
#             its output to integers, so it must see a count scale)
#   ConQuR  : ONE run over all 149; batchid = Data_group, batch_ref = "Site 1",
#             covariates = Age, Gender, BMI (no disease variable); defaults as in the
#             original profiles/ConQuR_analysis.txt call (logistic_lasso=F,
#             quantile_type="standard", interplt=F).
#   SEED    : ConQuR is NOT deterministic by default - ConQuR_each jitters counts with
#             quantreg::dither() (ConQuR_help_functions.R:240) without a seed. Here each
#             taxon ll gets set.seed(SEED + ll) -> identical output for any NCORE.
#             (The unseeded first run and its aborted ML run are in
#             backup_A1v3_unseeded_20260930/.)
#   per taxon: ConQuR:::ConQuR_each called directly so that taxa kept at their
#             original value are counted by reason (ConQuR does this silently).
#   output  : species_s_matrix_conqur.csv (corrected counts -> relab),
#             species_s_counts_conqur.csv (corrected integer counts; alpha Chao1),
#             T_A1v3_conqur_taxa.csv, T_A1v3_conqur_summary.csv, LOG_A1v3_conqur.txt
#             Refuses to run if any output already exists (back up + remove first).
suppressPackageStartupMessages({
  .libPaths(c("./nested_cv/Rlib", .libPaths()))
  library(ConQuR); library(vegan); library(parallel)
})
A1   <- "./nested_cv"
ROOT <- ".."
OUT  <- Sys.getenv("OUTDIR", A1)
NCORE <- as.integer(Sys.getenv("NCORE", "30"))
SEED <- 20260930L
outs <- file.path(OUT, c("species_s_matrix_conqur.csv", "species_s_counts_conqur.csv",
                         "T_A1v3_conqur_taxa.csv", "T_A1v3_conqur_summary.csv", "LOG_A1v3_conqur.txt"))
if (any(file.exists(outs))) stop("output exists, back it up and remove first: ",
                                 paste(basename(outs[file.exists(outs)]), collapse = ", "))
logf <- outs[5]; cat("", file = logf)
P <- function(...) { s <- paste0(...); cat(s, "\n"); cat(s, "\n", file = logf, append = TRUE) }

inp <- file.path(A1, "species_s_matrix.csv")
P("input md5 ", basename(inp), " = ", unname(tools::md5sum(inp)), " | seed base = ", SEED)
sp  <- read.csv(inp, row.names = 1, check.names = FALSE)
lab <- read.csv(file.path(A1, "labels.csv"), row.names = 1)
stopifnot(identical(rownames(sp), rownames(lab)))
meta <- read.delim(file.path(ROOT, "project/input_sparcc/dataset/total_meta.tsv"))
meta <- meta[match(rownames(sp), meta$Sample_ID), ]
stopifnot(!anyNA(meta$Sample_ID), !anyNA(meta[, c("Data_group", "Age", "Gender", "BMI")]))

# mapped reads per sample from the raw profile header
mapped <- sapply(rownames(sp), function(s) {
  f <- Sys.glob(file.path(ROOT, "profiles/raw_metaphlan", paste0(s, "_taxonomic_profile*.tsv")))
  stopifnot(length(f) == 1)
  h <- grep("^#estimated_reads_mapped_to_known_clades", readLines(f, n = 10), value = TRUE)
  as.numeric(sub(".*:", "", h))
})
cnt <- round(as.matrix(sp) * mapped)
lost <- sum(sp > 0 & cnt == 0)
P("A1-v3 ConQuR | n=", nrow(cnt), " taxa=", ncol(cnt), " cores=", NCORE)
P("mapped reads: min=", min(mapped), " median=", median(mapped), " max=", max(mapped))
P("nonzero relab rounded to 0 count: ", lost, " of ", sum(sp > 0), " nonzero cells")

# design exactly as ConQuR() builds it
batch_ref <- "Site 1"
batchid <- relevel(factor(meta$Data_group), ref = batch_ref)     # global: simple_QQ reads it
covariates <- data.frame(Age = as.numeric(meta$Age), Gender = factor(meta$Gender),
                         BMI = as.numeric(meta$BMI))
X <- data.frame(covariates, batchid)
X_span <- model.matrix(~., X)[, -1]
X_correct <- X; X_correct$batchid <- factor(batch_ref)
X_span_correct <- X_span; X_span_correct[, grep("batchid", colnames(X_span_correct))] <- 0
P("batch table: ", paste(names(table(batchid)), table(batchid), sep = "=", collapse = ", "))

each <- function(ll) {
  set.seed(SEED + ll)                     # dither() inside ConQuR_each
  y <- as.numeric(cnt[, ll]); nz <- sum(y > 0)
  reason <- if (nz <= 1) "nonzero<=1" else if (ncol(model.matrix(~., X[y > 0, , drop = FALSE])) >= nz) "p>=n" else NA
  out <- tryCatch(ConQuR:::ConQuR_each(y = y, X = X, X_span = X_span, X_correct = X_correct,
                    X_span_correct = X_span_correct, batch_ref = batch_ref, delta = 0.4999,
                    taus = seq(0.005, 0.995, by = 0.005), logistic_lasso = FALSE,
                    quantile_type = "standard", lambda_quantile = "2p/n", interplt = FALSE),
                  error = function(e) structure(y, err = conditionMessage(e)))
  err <- attr(out, "err"); out <- as.numeric(out); out[out < 0] <- 0
  if (!is.null(err)) reason <- "error"
  else if (is.na(reason) && identical(out, y)) reason <- "kept_other(singular/QQ)"
  list(y = out, reason = ifelse(is.na(reason), "corrected", reason), nz = nz)
}
t0 <- Sys.time()
res <- mclapply(seq_len(ncol(cnt)), each, mc.cores = NCORE)
P("ConQuR elapsed: ", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " min")

adj <- sapply(res, `[[`, "y"); dimnames(adj) <- dimnames(cnt)
tx <- data.frame(taxon = colnames(cnt), n_nonzero = sapply(res, `[[`, "nz"),
                 status = sapply(res, `[[`, "reason"),
                 zero_to_pos = colSums(cnt == 0 & adj > 0), pos_to_zero = colSums(cnt > 0 & adj == 0))
write.csv(tx, outs[3], row.names = FALSE)
for (s in names(table(tx$status))) P("  status ", s, ": ", sum(tx$status == s))

ra <- adj / rowSums(adj)
stopifnot(all(rowSums(adj) > 0))
write.csv(ra, outs[1])
write.csv(adj, outs[2])

# before/after summary
before <- cnt / rowSums(cnt)
pm <- function(M, v) { set.seed(1); a <- adonis2(vegdist(M, "bray") ~ v, permutations = 999)
                       c(R2 = a$R2[1], p = a$`Pr(>F)`[1]) }
g4 <- factor(lab$Group)
sm <- data.frame(
  metric = c("taxa corrected", "taxa kept original", "cells 0->pos", "cells pos->0",
             "rounding loss (nonzero->0 before ConQuR)",
             "PERMANOVA R2 Data_group", "PERMANOVA p Data_group",
             "PERMANOVA R2 Group(4)", "PERMANOVA p Group(4)",
             "median per-sample Bray-Curtis before vs after"),
  before = c(NA, NA, NA, NA, lost, pm(before, batchid), pm(before, g4), NA),
  after  = c(sum(tx$status == "corrected"), sum(tx$status != "corrected"),
             sum(tx$zero_to_pos), sum(tx$pos_to_zero), NA,
             pm(ra, batchid), pm(ra, g4),
             median(sapply(seq_len(nrow(ra)), function(i) sum(abs(before[i, ] - ra[i, ])) / 2))))
write.csv(sm, outs[4], row.names = FALSE)
print(sm)
P("[written] ", paste(basename(outs[1:4]), collapse = ", "), " -> ", OUT)
