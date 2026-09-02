"""
run_dgcna_correlation.py
========================
Reproduction of the dGCNA correlation analysis of Martinez-Lopez et al.

Method (as described in the paper and implemented in the authors' R code,
`basic_functions.R`):

1. Per gene, fit an intercept-only linear mixed model with donor as a random
   effect (`lmer(Expression ~ 1 + (1 | Donor))`) *separately within each group*
   and take the conditional residuals.
2. Pearson-correlate the residuals within each group and take the difference
   `diff = corr(g2) - corr(g1)` where g1 = controls and g2 = cases.
3. Repeat steps 1-2 on 512 random regroupings of donors. For each gene pair,
   count the fraction of random regroupings in which the observed difference is
   more extreme *in the direction of its own sign* than the random one
   (`CompareDifferences`). Set to zero every edge whose relative frequency is
   below 0.975 (`GetNetworkConnectionsFromMask`).

This module expects data that have ALREADY been gene- and donor-filtered, and
whose `.X` is on the log2(RPKM + 1) scale. No filtering or transformation is
applied here unless `min_cells_per_donor` is set explicitly.

Usage:
    import scanpy as sc
    adata = sc.read_h5ad("your_data.h5ad")

    from run_dgcna_correlation import run_dgcna_analysis
    results = run_dgcna_analysis(adata, genes=gene_list)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Linear mixed model: REML variance components for y_ij = mu + u_i + e_ij
# ---------------------------------------------------------------------------

def _reml_objective(gammas, counts, donor_means, ssw, n_cells):
    """
    Evaluate -2 x profiled REML log-likelihood for the intercept-only LMM.

    For y = 1*mu + Z*u + e with V = sigma2_e * (I + gamma * Z Z') and
    gamma = sigma2_u / sigma2_e, both mu and sigma2_e can be profiled out
    analytically, leaving a one-dimensional criterion in gamma:

        -2 logL_R(gamma) = sum_i log(1 + gamma * n_i)
                           + log(sum_i w_i)
                           + (N - 1) * log(R(gamma))

    with w_i = n_i / (1 + gamma * n_i),
         mu_hat = sum_i w_i * ybar_i / sum_i w_i,
         R(gamma) = SSW + sum_i w_i * (ybar_i - mu_hat)^2.

    This is the same criterion lme4/lmer optimises, so the resulting variance
    components are REML estimates rather than method-of-moments estimates.

    Parameters
    ----------
    gammas : np.ndarray, shape (K, G)
        Candidate gamma values per gene.
    counts : np.ndarray, shape (q,)
        Number of cells per donor.
    donor_means : np.ndarray, shape (q, G)
        Per-donor mean expression.
    ssw : np.ndarray, shape (G,)
        Within-donor residual sum of squares per gene.
    n_cells : int
        Total number of cells, N.

    Returns
    -------
    obj : np.ndarray, shape (K, G)
    mu : np.ndarray, shape (K, G)
    """
    # w has shape (K, q, G)
    gn = gammas[:, None, :] * counts[None, :, None]
    w = counts[None, :, None] / (1.0 + gn)

    sum_w = w.sum(axis=1)                                    # (K, G)
    mu = (w * donor_means[None, :, :]).sum(axis=1) / sum_w   # (K, G)

    dev = donor_means[None, :, :] - mu[:, None, :]           # (K, q, G)
    rss = ssw[None, :] + (w * dev**2).sum(axis=1)            # (K, G)

    logdet = np.log1p(gn).sum(axis=1)                        # (K, G)

    obj = logdet + np.log(sum_w) + (n_cells - 1) * np.log(np.maximum(rss, 1e-300))
    return obj, mu


def _donor_design(Y, donor_codes, n_donors):
    """Per-donor sufficient statistics: indicator matrix, counts, means, SSW."""
    Y = np.asarray(Y, dtype=np.float64)
    n_cells = Y.shape[0]

    Z = np.zeros((n_cells, n_donors), dtype=np.float64)
    Z[np.arange(n_cells), donor_codes] = 1.0
    counts = Z.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Every donor code must be present at least once")

    donor_means = (Z.T @ Y) / counts[:, None]
    # SSW = sum_ij (y_ij - ybar_i)^2, computed without forming the residuals
    ssw = np.maximum(
        (Y**2).sum(axis=0) - (counts[:, None] * donor_means**2).sum(axis=0), 0.0
    )
    return Y, Z, counts, donor_means, ssw


def fit_reml(Y, donor_codes, n_donors, n_grid=256, n_refine=24, n_passes=4):
    """
    Profiled-REML fit of y_ij = mu + u_i + e_ij for every gene at once.

    The one-dimensional criterion of `_reml_objective` is minimised over gamma
    by a coarse log-spaced grid followed by repeated bracket shrinking. The grid
    explicitly includes gamma = 0, so the boundary ("singular fit") case that
    `lmer` reports as a zero variance component is representable.

    Parameters
    ----------
    Y : np.ndarray, shape (N, G)
    donor_codes : np.ndarray, shape (N,)
        Integer donor label per cell, values in [0, n_donors).
    n_donors : int
    n_grid : int
        Number of gamma values in the coarse search.
    n_refine, n_passes : int
        Size and number of the local refinement passes.

    Returns
    -------
    gamma : np.ndarray, shape (G,)
        Fitted sigma2_u / sigma2_e per gene.
    mu : np.ndarray, shape (G,)
        Generalised-least-squares intercept per gene.
    """
    Y, _, counts, donor_means, ssw = _donor_design(Y, donor_codes, n_donors)
    n_cells, n_genes = Y.shape
    cols = np.arange(n_genes)

    # --- Stage 1: coarse log-spaced grid, including the singular fit gamma = 0
    grid = np.concatenate([[0.0], np.logspace(-8, 8, n_grid - 1)])
    cand = np.repeat(grid[:, None], n_genes, axis=1)
    obj, mu_all = _reml_objective(cand, counts, donor_means, ssw, n_cells)
    k_best = np.argmin(obj, axis=0)
    gamma_hat = cand[k_best, cols]
    mu_hat = mu_all[k_best, cols]
    lo = grid[np.maximum(k_best - 1, 0)]
    hi = grid[np.minimum(k_best + 1, len(grid) - 1)]

    # --- Stage 2: repeatedly shrink the bracket around the current optimum
    frac = np.linspace(0.0, 1.0, n_refine)[:, None]
    for _ in range(n_passes):
        # Interpolate geometrically, falling back to linear where lo == 0
        with np.errstate(divide='ignore', invalid='ignore'):
            geo = lo[None, :] * (hi[None, :] / np.maximum(lo[None, :], 1e-300)) ** frac
        lin = lo[None, :] + frac * (hi - lo)[None, :]
        cand = np.where(lo[None, :] > 0, geo, lin)

        obj, mu_all = _reml_objective(cand, counts, donor_means, ssw, n_cells)
        k_best = np.argmin(obj, axis=0)
        gamma_hat = cand[k_best, cols]
        mu_hat = mu_all[k_best, cols]

        lo = cand[np.maximum(k_best - 1, 0), cols]
        hi = cand[np.minimum(k_best + 1, n_refine - 1), cols]

    return gamma_hat, mu_hat


def _lmm_residuals_matrix(Y, donor_codes, n_donors, **fit_kwargs):
    """
    Conditional LMM residuals for every gene, vectorised across genes.

    Fits y_ij = mu + u_i + e_ij per gene by profiled REML (`fit_reml`), then
    returns r_ij = y_ij - mu_hat - u_hat_i, where

        u_hat_i = gamma * n_i * (ybar_i - mu_hat) / (1 + gamma * n_i)

    is the BLUP of the donor random effect. This is exactly the quantity
    `residuals()` returns for an `lmer` fit.

    Parameters
    ----------
    Y : np.ndarray, shape (N, G)
        Expression matrix on the log2(RPKM + 1) scale.
    donor_codes : np.ndarray, shape (N,)
        Integer donor label per cell, values in [0, n_donors).
    n_donors : int
    **fit_kwargs
        Forwarded to `fit_reml`.

    Returns
    -------
    residuals : np.ndarray, shape (N, G)
    """
    Y, Z, counts, donor_means, _ = _donor_design(Y, donor_codes, n_donors)
    gamma_hat, mu_hat = fit_reml(Y, donor_codes, n_donors, **fit_kwargs)

    # BLUP of the donor random effects (gamma = 0 correctly gives u = 0)
    gn = gamma_hat[None, :] * counts[:, None]
    u = gn * (donor_means - mu_hat[None, :]) / (1.0 + gn)

    return Y - mu_hat[None, :] - (Z @ u)


def get_lmm_residuals(adata, min_cells_per_donor=None, donor_col='Donor'):
    """
    Compute LMM residuals for an AnnData object.

    This removes donor-specific expression differences while preserving
    within-donor variation.

    Parameters
    ----------
    adata : AnnData
        `.X` must already be log2(RPKM + 1); must have `donor_col` in `.obs`.
    min_cells_per_donor : int or None
        If None (default), no donor filtering is applied -- the input is assumed
        to be pre-filtered. If given, donors with <= this many cells are dropped
        (strict `>`, matching "more than 30 cells/donor" in the paper).
    donor_col : str

    Returns
    -------
    residuals_df : pd.DataFrame
        Cells x Genes residuals matrix
    obs_df : pd.DataFrame
        `.obs` with the same index as `residuals_df`
    """
    subset = adata
    if min_cells_per_donor is not None:
        donor_counts = adata.obs[donor_col].value_counts()
        keep = donor_counts[donor_counts > min_cells_per_donor].index
        subset = adata[adata.obs[donor_col].isin(keep)]
        print(f"Kept {len(keep)} donors with > {min_cells_per_donor} cells")

    if subset.n_obs == 0:
        raise ValueError("No cells remaining after filtering")

    Y = subset.X.toarray() if hasattr(subset.X, 'toarray') else np.asarray(subset.X)

    donor_cat = pd.Categorical(subset.obs[donor_col])
    print(f"Cells: {subset.n_obs}, donors: {len(donor_cat.categories)}, "
          f"genes: {subset.n_vars}")

    residuals = _lmm_residuals_matrix(
        Y, donor_cat.codes.astype(np.intp), len(donor_cat.categories)
    )

    residuals_df = pd.DataFrame(
        residuals, index=subset.obs_names, columns=subset.var_names
    )
    return residuals_df, subset.obs.copy()


# ---------------------------------------------------------------------------
# Correlation matrices and the observed differential network
# ---------------------------------------------------------------------------

def _corr_from_matrix(X):
    """
    Pearson correlation across columns of X, as a plain ndarray.

    `np.corrcoef` can return diagonal entries that differ from 1.0 by ~1e-16.
    Left alone, those round-off errors give the differential network a non-zero
    diagonal which then passes the one-sided bootstrap comparison and inflates
    the edge count, so the diagonal is set exactly.
    """
    corr = np.corrcoef(np.asarray(X, dtype=np.float64), rowvar=False)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_correlation_matrix(df):
    """Compute Pearson correlation matrix using np.corrcoef."""
    return pd.DataFrame(_corr_from_matrix(df.values),
                        index=df.columns, columns=df.columns)


def build_differential_network(residuals_df, obs_df, disease_col='Disease',
                               t2d_value='type II diabetes', ctrl_value='normal'):
    """
    Build the differential correlation network between T2D and control.

    Returns
    -------
    corr_ctrl, corr_t2d, diff_net : pd.DataFrame
        Correlation matrices and their difference (T2D - control), matching
        `g2 - g1` in the authors' R code.
    """
    t2d_mask = obs_df[disease_col] == t2d_value
    ctrl_mask = obs_df[disease_col] == ctrl_value

    print(f"T2D cells: {t2d_mask.sum()}, Control cells: {ctrl_mask.sum()}")

    corr_t2d = compute_correlation_matrix(residuals_df.loc[t2d_mask])
    corr_ctrl = compute_correlation_matrix(residuals_df.loc[ctrl_mask])

    diff_net = corr_t2d - corr_ctrl

    return corr_ctrl, corr_t2d, diff_net


# ---------------------------------------------------------------------------
# Bootstrap / random-regrouping null
# ---------------------------------------------------------------------------

def run_bootstrap_streaming(expr_df, obs_df, obs_diff, disease_col='Disease',
                            t2d_value='type II diabetes', ctrl_value='normal',
                            donor_col='Donor', n_iterations=512,
                            replacement=True, seed=42):
    """
    Relative-frequency mask from random regroupings of donors.

    Faithful to `GetMaskFromRandomDonors` in the authors' `basic_functions.R`:
    the LMM is *refitted within each random group* at every iteration, so the
    residuals used for the null correlations are never conditioned on the true
    disease labels.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Cells x Genes log2(RPKM + 1) expression -- NOT residuals. Residuals are
        recomputed inside the loop for each random group.
    obs_df : pd.DataFrame
        Cell metadata aligned to `expr_df`, with `donor_col` and `disease_col`.
    obs_diff : pd.DataFrame
        Observed differential network (T2D - control).
    n_iterations : int
        Number of random regroupings (the paper uses 512).
    replacement : bool
        If True (the R default), the two random groups are drawn independently
        and *with replacement* from the pooled donor list, so a donor may appear
        in both groups or twice within a group. If False, the pooled donors are
        permuted and split, giving a label-permutation null.
    seed : int

    Returns
    -------
    mask : pd.DataFrame
        Relative frequency, per gene pair, of random regroupings in which the
        observed difference was more extreme than the random one in the
        direction of its own sign.
    """
    rng = np.random.default_rng(seed)

    donor_of_cell = obs_df[donor_col].to_numpy()
    donors = pd.unique(donor_of_cell)
    donor_index = {d: i for i, d in enumerate(donors)}
    # Cell positions belonging to each donor, so duplicated donors duplicate cells
    cells_of_donor = [np.flatnonzero(donor_of_cell == d) for d in donors]

    n_ctrl_donors = obs_df.loc[obs_df[disease_col] == ctrl_value, donor_col].nunique()
    n_case_donors = obs_df.loc[obs_df[disease_col] == t2d_value, donor_col].nunique()
    n_donors_total = len(donors)

    if not replacement and n_ctrl_donors + n_case_donors != n_donors_total:
        raise ValueError(
            "Permutation null requires n_ctrl + n_cases to equal the number of "
            f"donors (got {n_ctrl_donors} + {n_case_donors} != {n_donors_total})"
        )

    X = np.asarray(expr_df.values, dtype=np.float64)

    obs_vals = obs_diff.values           # signed, NOT absolute
    pos_mask = obs_vals > 0              # direction of each observed edge
    wins = np.zeros_like(obs_vals)

    def _group_residual_corr(group_donors):
        """Refit the LMM within a random group and correlate its residuals."""
        idx = np.concatenate([cells_of_donor[donor_index[d]] for d in group_donors])
        # Repeated donors keep a single shared label, as R's list subsetting does
        codes_full = np.array([donor_index[d] for d in group_donors])
        codes = np.concatenate([
            np.full(len(cells_of_donor[c]), c) for c in codes_full
        ])
        uniq, codes = np.unique(codes, return_inverse=True)
        res = _lmm_residuals_matrix(X[idx], codes.astype(np.intp), len(uniq))
        return _corr_from_matrix(res)

    desc = ("Random donor groups (with replacement)" if replacement
            else "Donor label permutations")
    for _ in tqdm(range(n_iterations), desc=desc):
        if replacement:
            g1_donors = donors[rng.integers(0, n_donors_total, size=n_ctrl_donors)]
            g2_donors = donors[rng.integers(0, n_donors_total, size=n_case_donors)]
        else:
            shuffled = rng.permutation(donors)
            g1_donors = shuffled[:n_ctrl_donors]        # pseudo-controls
            g2_donors = shuffled[n_ctrl_donors:]        # pseudo-cases

        corr_g1 = _group_residual_corr(g1_donors)
        corr_g2 = _group_residual_corr(g2_donors)
        boot_diff = corr_g2 - corr_g1                  # cases - controls

        # CompareDifferences: sign-aware, one-sided comparison
        wins += np.where(pos_mask, obs_vals > boot_diff, obs_vals < boot_diff)

    return pd.DataFrame(wins / n_iterations,
                        index=obs_diff.index, columns=obs_diff.columns)

def filter_network(obs_diff, mask, threshold=0.975):
    """
    Apply the relative-frequency mask to the differential network.

    Keep only edges whose observed Delta r beat the random background, in the
    direction of its own sign, in at least (threshold * 100)% of iterations.

    Equivalent to `GetNetworkConnectionsFromMask` except that the surviving
    weights are NOT divided by max(|Delta r|); raw Delta r values are returned.

    Parameters
    ----------
    obs_diff : pd.DataFrame
        Observed differential correlation (T2D - control)
    mask : pd.DataFrame
        Relative frequencies from `run_bootstrap_streaming`
    threshold : float
        Relative-frequency threshold (the paper uses 0.975)

    Returns
    -------
    filtered_net : pd.DataFrame
        Differential network with non-significant edges set to 0
    """
    keep = (mask >= threshold).to_numpy().copy()
    np.fill_diagonal(keep, False)          # self-edges are not edges

    filtered_net = obs_diff.where(
        pd.DataFrame(keep, index=obs_diff.index, columns=obs_diff.columns), 0.0
    )

    n_sig = int(keep.sum()) // 2
    n_total = (obs_diff.shape[0] * (obs_diff.shape[0] - 1)) // 2
    print(f"Significant edges: {n_sig} / {n_total} "
          f"({100 * n_sig / max(1, n_total):.1f}%)")

    return filtered_net


def run_dgcna_analysis(adata, genes=None, disease_col='Disease',
                       t2d_value='type II diabetes', ctrl_value='normal',
                       donor_col='Donor', min_cells_per_donor=None,
                       n_bootstrap=512, threshold=0.975,
                       replacement=True, seed=42):
    """
    Run the full dGCNA analysis pipeline.

    Parameters
    ----------
    adata : AnnData
        Single-cell data with `donor_col` and `disease_col` in `.obs`. `.X` must
        already be log2(RPKM + 1) and already gene/donor filtered.
    genes : list, optional
        Subset of genes to analyse (e.g. the NDCG genes)
    disease_col : str
        Column in `.obs` holding disease status
    t2d_value, ctrl_value : str
        Values of `disease_col` denoting cases and controls
    donor_col : str
        Column in `.obs` holding the donor identifier
    min_cells_per_donor : int or None
        Leave as None if the data are already filtered (the default). If given,
        donors with <= this many cells are dropped within each condition.
    n_bootstrap : int
        Number of random donor regroupings (the paper uses 512)
    threshold : float
        Relative-frequency threshold for the mask (the paper uses 0.975)
    replacement : bool
        Whether random donor groups are drawn with replacement, as in the
        authors' R default. See `run_bootstrap_streaming`.
    seed : int
        Random seed

    Returns
    -------
    dict with keys:
        'residuals', 'obs' : pd.DataFrame - LMM residuals and matching metadata
        'corr_ctrl', 'corr_t2d' : pd.DataFrame - per-group correlation matrices
        'diff_net' : pd.DataFrame - raw differential network (unscaled Delta r)
        'filtered_net' : pd.DataFrame - masked differential network
        'mask' : pd.DataFrame - bootstrap relative frequencies
    """
    # Subset genes if specified
    if genes is not None:
        genes_in_data = [g for g in genes if g in adata.var_names]
        print(f"Using {len(genes_in_data)} / {len(genes)} specified genes")
        adata = adata[:, genes_in_data].copy()

    # Step 1: LMM residuals, fitted separately within each condition
    print("\n=== Step 1: Computing LMM residuals (per condition) ===")
    t2d_sub = adata[adata.obs[disease_col] == t2d_value].copy()
    ctrl_sub = adata[adata.obs[disease_col] == ctrl_value].copy()

    res_t2d, obs_t2d = get_lmm_residuals(t2d_sub, min_cells_per_donor, donor_col)
    res_ctrl, obs_ctrl = get_lmm_residuals(ctrl_sub, min_cells_per_donor, donor_col)

    residuals_df = pd.concat([res_t2d, res_ctrl], axis=0)
    obs_df = pd.concat([obs_t2d, obs_ctrl], axis=0)

    # Raw expression aligned to obs_df, needed to refit the LMM inside the null
    aligned = adata[obs_df.index]
    X = aligned.X.toarray() if hasattr(aligned.X, 'toarray') else np.asarray(aligned.X)
    expr_df = pd.DataFrame(X, index=obs_df.index, columns=adata.var_names)

    # Step 2: Observed differential network
    print("\n=== Step 2: Building differential network ===")
    corr_ctrl, corr_t2d, diff_net = build_differential_network(
        residuals_df, obs_df, disease_col, t2d_value, ctrl_value
    )

    # Step 3: Random-regrouping mask
    print("\n=== Step 3: Bootstrap filtering ===")
    mask = run_bootstrap_streaming(
        expr_df=expr_df,
        obs_df=obs_df,
        obs_diff=diff_net,
        disease_col=disease_col,
        t2d_value=t2d_value,
        ctrl_value=ctrl_value,
        donor_col=donor_col,
        n_iterations=n_bootstrap,
        replacement=replacement,
        seed=seed
    )
    filtered_net = filter_network(diff_net, mask, threshold)

    return {
        'residuals': residuals_df,
        'obs': obs_df,
        'corr_ctrl': corr_ctrl,
        'corr_t2d': corr_t2d,
        'diff_net': diff_net,
        'filtered_net': filtered_net,
        'mask': mask
    }


if __name__ == "__main__":
    # Example usage
    print("Example usage:")
    print("  import scanpy as sc")
    print("  adata = sc.read_h5ad('your_data.h5ad')")
    print("  from run_dgcna_correlation import run_dgcna_analysis")
    print("  results = run_dgcna_analysis(adata, genes=ndcg_genes)")
    print("  results['filtered_net'].to_csv('filtered_diff_network.csv')")
