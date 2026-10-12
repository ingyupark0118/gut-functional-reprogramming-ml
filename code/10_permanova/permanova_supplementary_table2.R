# T27: Supplementary Table 2 rebuilt on the inputs that give the Group R2 = 0.057 printed in Fig 1C
# (species = raw MetaPhlAn table, 1,338 species; KO = KO-sum matrix). Same calls as B1_beta/b1_beta.R: set.seed(1) before each test, 999 permutations.
suppressPackageStartupMessages({library(vegan); library(openxlsx)})
REV <- "."; OUT <- file.path(REV, "tables/supplementary")
fo <- file.path(OUT, c("TableS2_PERMANOVA.csv", "TableS2_PERMANOVA.xlsx", "TableS2_footnotes.txt")); stopifnot(!any(file.exists(fo)))
lab <- read.csv(file.path(REV, "nested_cv/labels.csv"), check.names = FALSE); names(lab)[1] <- "Sample_ID"
cov <- read.delim(file.path(REV, "input_beta/total_meta.tsv"), check.names = FALSE); cov <- cov[match(lab$Sample_ID, cov$Sample_ID), ]
stopifnot(identical(cov$Sample_ID, lab$Sample_ID), nrow(cov) == 149)
cov$Group <- factor(lab$Group, levels = c("NC", "DM", "Cirrhosis", "CDM"))
IN <- c(Microbiome = "nested_cv/species_s_matrix.csv", `KO gene` = "nested_cv/ko_matrix.csv")
FM <- c(A = "D ~ Group + Age + Sex + BMI", B = "D ~ Group + Age + Sex + BMI + PPI + Statin + Probiotics + Alcohol")
rows <- list()
for (ds in names(IN)) {
  X <- as.matrix(read.csv(file.path(REV, IN[[ds]]), row.names = 1, check.names = FALSE))[lab$Sample_ID, ]
  D <- vegdist(X, "bray"); set.seed(1); an <- anosim(D, cov$Group, permutations = 999)
  for (p in names(FM)) {
    f <- as.formula(FM[[p]])
    set.seed(1); w <- adonis2(f, data = cov, permutations = 999)                 # whole model
    set.seed(1); m <- adonis2(f, data = cov, permutations = 999, by = "margin")  # each term adjusted for all others
    g <- m["Group", ]
    rows[[length(rows) + 1]] <- data.frame(Panel = p, Dataset = ds, Distance = "Bray-Curtis", n_features = ncol(X),
      Df = paste0(w$Df[1], "/", w$Df[2]), Sumsq = w$SumOfSqs[1], R2 = w$R2[1], F = w$F[1], p = w$`Pr(>F)`[1],
      G_Df = g$Df, G_Sumsq = g$SumOfSqs, G_R2 = g$R2, G_F = g$F, G_p = g$`Pr(>F)`, ANOSIM_R = an$statistic, ANOSIM_p = an$signif)
  }
}
r <- do.call(rbind, rows); print(r, digits = 6)
# must reproduce the stored values (B1_beta/T_B1_beta_summary.csv) and the value printed in Fig 1C
s <- read.csv(file.path(REV, "diversity/T_B1_beta_summary.csv")); k <- function(d, p) r[r$Dataset == d & r$Panel == p, ]
for (i in 1:2) { d <- names(IN)[i]
  stopifnot(abs(k(d, "A")$R2 - s$model_R2_basic[i]) < 1e-10, abs(k(d, "B")$R2 - s$model_R2_extended[i]) < 1e-10,
            abs(k(d, "A")$G_R2 - s$Group_R2_marginal[i]) < 1e-10, abs(k(d, "A")$ANOSIM_R - s$ANOSIM_R[i]) < 1e-10) }
stopifnot(round(k("Microbiome", "A")$G_R2, 3) == 0.057); cat("checks vs B1_beta summary and Fig 1C value (0.057): passed\n")
f3 <- function(x) formatC(x, format = "f", digits = 3)
t <- data.frame(Panel = ifelse(r$Panel == "A", "(A) Base model", "(B) Extended covariate model"), Dataset = r$Dataset, Distance = r$Distance,
  `Whole model: Df (model/resid)` = r$Df, `Whole model: Sumsq` = f3(r$Sumsq), `Whole model: R2` = f3(r$R2), `Whole model: F` = f3(r$F), `Whole model: p-value` = f3(r$p),
  `Group term: Df` = r$G_Df, `Group term: Sumsq` = f3(r$G_Sumsq), `Group term: R2` = f3(r$G_R2), `Group term: F` = f3(r$G_F), `Group term: p-value` = f3(r$G_p),
  `ANOSIM: R` = f3(r$ANOSIM_R), `ANOSIM: p-value` = f3(r$ANOSIM_p), check.names = FALSE)
write.csv(t, fo[1], row.names = FALSE); write.xlsx(t, fo[2])
writeLines(c(
"Supplementary Table 2. Covariate-adjusted PERMANOVA and ANOSIM analyses of gut microbiome community structure. Beta-diversity was assessed using Bray-Curtis dissimilarity and tested by permutational multivariate analysis of variance (PERMANOVA; adonis2, 999 permutations), complemented by analysis of similarities (ANOSIM; 999 permutations, grouping by clinical group, without covariates). Analyses were applied to species-level profiles (Microbiome; 1,338 species, relative abundance without batch correction) and KO gene-level profiles (KO gene; 6,097 KOs) of the 149 participants across the four clinical groups (NC, DM, Cirrhosis, CDM).",
"(A) Base model (reported in Fig. 1C and Fig. 1E): adonis2(Bray-Curtis ~ Group + Age + Sex + BMI).",
"(B) Extended covariate model (sensitivity analysis): adonis2(Bray-Curtis ~ Group + Age + Sex + BMI + PPI + Statin + Probiotics + Alcohol).",
"Whole model: variance explained by all terms of the model together. Group term: marginal effect of clinical group, adjusted for all other terms of the model (by = 'margin'); this is the R2 reported in the text and in Fig. 1C. p-values are permutation p-values; 0.001 is the smallest value attainable with 999 permutations. ANOSIM does not depend on the covariate model and is therefore identical in (A) and (B)."), fo[3])
cat("written", fo, sep = "\n")
