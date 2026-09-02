################################################################################
# Run original R dGCNA pipeline on exported CSV data and write outputs for
# comparison with the Python implementation.
#
# Inputs:
#   Expr_beta.csv  – cells (rows) × genes (columns), with row/col names
#   Obs_beta.csv   – columns: Donor, Disease  (categories: T2D, normal)
#
# Outputs:
#   R_residuals_T2D.csv    – residuals for T2D group  (cells × genes)
#   R_residuals_normal.csv – residuals for normal group
#   R_corr_T2D.csv         – Pearson correlation matrix, T2D
#   R_corr_normal.csv      – Pearson correlation matrix, normal
#   R_diff_net.csv         – differential network (T2D − normal)
################################################################################

library(lme4)
source("code/basic_functions.R")

# ---------- 1. Read data -----------------------------------------------------
cat("Reading expression matrix...\n")
expr <- read.csv("Expr_beta.csv", row.names = 1, check.names = FALSE)
obs  <- read.csv("Obs_beta.csv",  row.names = 1, stringsAsFactors = FALSE)

# expr: cells × genes  →  need genes × cells per donor (panel list)
genes <- colnames(expr)
cat(sprintf("  %d cells, %d genes\n", nrow(expr), length(genes)))

# ---------- 2. Build panel list (genes × cells, one matrix per donor) --------
donors <- unique(obs$Donor)
cat(sprintf("  %d donors\n", length(donors)))

panel <- list()
for (d in donors) {
  idx <- which(obs$Donor == d)
  # genes × cells matrix
  mat <- t(as.matrix(expr[idx, , drop = FALSE]))
  colnames(mat) <- rownames(expr)[idx]
  panel[[d]] <- mat
}

# ---------- 3. Build sample_groups list ---------------------------------------
disease_per_donor <- tapply(obs$Disease, obs$Donor, function(x) x[1])
sample_groups <- list(
  T2D    = names(disease_per_donor[disease_per_donor == "T2D"]),
  normal = names(disease_per_donor[disease_per_donor == "normal"])
)
cat(sprintf("  T2D donors: %s\n",    paste(sample_groups$T2D, collapse = ", ")))
cat(sprintf("  normal donors: %s\n", paste(sample_groups$normal, collapse = ", ")))

# ---------- 4. Fit LMMs (no normalization – data is already log-transformed) --
cat("Fitting linear mixed models...\n")
lm_models <- GetLinearModel(panel, sample_groups, genes = genes,
                            model = "lmer", normalization = FALSE)

# ---------- 5. Extract residuals per condition --------------------------------
cat("Extracting residuals...\n")
resid_list <- lapply(lm_models, function(models_per_group) {
  sapply(models_per_group, residuals)
})

# resid_list$T2D and resid_list$normal are matrices (cells × genes)
# Ensure column names are gene names
for (g in names(resid_list)) {
  colnames(resid_list[[g]]) <- genes
}

# ---------- 6. Compute correlations ------------------------------------------
cat("Computing correlations...\n")
corr_list <- FindSimilaritiesForNestedData(data_lm = lm_models, method = "pearson")

# ---------- 7. Differential network ------------------------------------------
diff_net <- corr_list$T2D - corr_list$normal

# ---------- 8. Write outputs --------------------------------------------------
cat("Writing output files...\n")

write.csv(resid_list$T2D,    "R_residuals_T2D.csv",    quote = FALSE)
write.csv(resid_list$normal, "R_residuals_normal.csv",  quote = FALSE)
write.csv(corr_list$T2D,     "R_corr_T2D.csv",         quote = FALSE)
write.csv(corr_list$normal,  "R_corr_normal.csv",       quote = FALSE)
write.csv(diff_net,           "R_diff_net.csv",          quote = FALSE)

cat("Done.\n")
