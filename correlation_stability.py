"""
correlation_stability.py
========================
Test stability of Pearson correlations under cell subsampling.

Demonstrates that correlation estimates in scRNA-seq are unstable when
a fraction of cells is removed, questioning whether dGCNA edges represent
robust biological signals.

Assumes you have:
  - residuals_df: pd.DataFrame (cells x genes) from LMM
  - obs_df: pd.DataFrame with 'Disease' and 'Donor' columns
  
Usage (in your notebook/session):
  from correlation_stability import run_stability_analysis
  results = run_stability_analysis(residuals_df, obs_df, genes=gene_list)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def compute_corr_numpy(X):
    """Fast Pearson correlation on numpy array (observations x features)."""
    return np.corrcoef(X, rowvar=False)


def subsample_correlations(residuals_df, obs_df, condition_col='Disease',
                           conditions=None, frac=0.8, n_iter=100, seed=42):
    """
    Subsample cells within each condition and recompute correlations.
    
    Parameters
    ----------
    residuals_df : pd.DataFrame (cells x genes)
    obs_df : pd.DataFrame with condition_col
    conditions : list of str, conditions to analyze (default: all unique values)
    frac : float, fraction of cells to keep per subsample
    n_iter : int, number of subsampling iterations
    seed : int
    
    Returns
    -------
    dict: {condition: np.ndarray of shape (n_iter, n_genes, n_genes)}
    """
    rng = np.random.default_rng(seed)
    
    if conditions is None:
        conditions = obs_df[condition_col].unique().tolist()
    
    n_genes = residuals_df.shape[1]
    results = {}
    
    for cond in conditions:
        mask = obs_df[condition_col] == cond
        cond_df = residuals_df.loc[mask]
        n_cells = len(cond_df)
        n_sample = int(n_cells * frac)
        
        print(f"\n{cond}: {n_cells} cells, sampling {n_sample} per iteration")
        
        corr_samples = np.zeros((n_iter, n_genes, n_genes))
        
        for i in tqdm(range(n_iter), desc=f"  Subsampling {cond}"):
            idx = rng.choice(n_cells, size=n_sample, replace=False)
            X = cond_df.values[idx]
            corr_samples[i] = compute_corr_numpy(X)
        
        results[cond] = corr_samples
    
    return results


def compute_stability_metrics(corr_samples, gene_names):
    """
    Compute per-edge stability metrics from subsampled correlations.
    
    Parameters
    ----------
    corr_samples : np.ndarray (n_iter, n_genes, n_genes)
    gene_names : list of str
    
    Returns
    -------
    stability_df : pd.DataFrame with columns:
        gene1, gene2, mean_r, sd_r, iqr_r, sign_flip_rate, cv
    """
    n_iter, n_genes, _ = corr_samples.shape
    
    rows = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            r_values = corr_samples[:, i, j]
            
            mean_r = r_values.mean()
            sd_r = r_values.std()
            iqr_r = np.percentile(r_values, 75) - np.percentile(r_values, 25)
            
            # Sign flip rate: fraction of iterations where sign differs from median
            median_sign = np.sign(np.median(r_values))
            sign_flip_rate = (np.sign(r_values) != median_sign).mean()
            
            rows.append({
                'gene1': gene_names[i],
                'gene2': gene_names[j],
                'mean_r': mean_r,
                'sd_r': sd_r,
                'iqr_r': iqr_r,
                'sign_flip_rate': sign_flip_rate,
            })
    
    return pd.DataFrame(rows)


def compute_delta_r_stability(corr_samples, conditions, gene_names):
    """
    Compute per-edge stability of Δr = r_cond2 − r_cond1 across subsamples.

    Parameters
    ----------
    corr_samples : dict {condition: np.ndarray (n_iter, n_genes, n_genes)}
    conditions : tuple/list of (ctrl_condition, case_condition)
        Δr is computed as case − ctrl.
    gene_names : list of str

    Returns
    -------
    delta_stability_df : pd.DataFrame with columns:
        gene1, gene2, mean_delta_r, sd_delta_r, sign_flip_rate_delta_r,
        ci_lower, ci_upper, ci_width
    """
    ctrl_key, case_key = conditions
    # Shape: (n_iter, n_genes, n_genes)
    delta_r_samples = corr_samples[case_key] - corr_samples[ctrl_key]
    n_iter, n_genes, _ = delta_r_samples.shape

    rows = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            dr = delta_r_samples[:, i, j]
            mean_dr = dr.mean()
            sd_dr = dr.std()
            median_sign = np.sign(np.median(dr))
            flip_rate = (np.sign(dr) != median_sign).mean()
            ci_lo = np.percentile(dr, 2.5)
            ci_hi = np.percentile(dr, 97.5)
            rows.append({
                'gene1': gene_names[i],
                'gene2': gene_names[j],
                'mean_delta_r': mean_dr,
                'sd_delta_r': sd_dr,
                'sign_flip_rate_delta_r': flip_rate,
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                'ci_width': ci_hi - ci_lo,
            })

    return pd.DataFrame(rows)


def compare_stability_to_dgcna(stability_df, edge_df):
    """
    Merge stability metrics with dGCNA edge results to see if 
    significant dGCNA edges are more/less stable.
    
    Parameters
    ----------
    stability_df : from compute_stability_metrics
    edge_df : your existing edge_df with delta_r, abs_delta_r columns
    
    Returns
    -------
    merged : pd.DataFrame
    """
    # Merge on gene pairs (handle order)
    stability_df['pair'] = stability_df.apply(
        lambda r: tuple(sorted([r['gene1'], r['gene2']])), axis=1
    )
    edge_df_copy = edge_df.copy()
    edge_df_copy['pair'] = edge_df_copy.apply(
        lambda r: tuple(sorted([r['gene1'], r['gene2']])), axis=1
    )
    
    merged = stability_df.merge(edge_df_copy[['pair', 'delta_r', 'abs_delta_r']], on='pair', how='left')
    merged['dgcna_significant'] = merged['abs_delta_r'] > 0
    
    return merged


def plot_stability(stability_df, merged_df=None, save_prefix='correlation_stability'):
    """
    Plot stability analysis results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ── Row 1: Overall stability ──
    
    # 1A: Distribution of SD(r) across edges
    ax = axes[0, 0]
    ax.hist(stability_df['sd_r'], bins=50, color='#4575b4', edgecolor='white', linewidth=0.3)
    ax.axvline(stability_df['sd_r'].median(), color='red', lw=1.5, ls='--', 
               label=f'Median = {stability_df["sd_r"].median():.3f}')
    ax.set_xlabel('SD(r) across subsamples')
    ax.set_ylabel('Number of edges')
    ax.set_title('A   Variability of correlation estimates')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    
    # 1B: Distribution of sign-flip rate
    ax = axes[0, 1]
    ax.hist(stability_df['sign_flip_rate'], bins=50, color='#d73027', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Sign-flip rate')
    ax.set_ylabel('Number of edges')
    ax.set_title('B   Fraction of subsamples with sign change')
    ax.spines[['top', 'right']].set_visible(False)
    pct_flips = (stability_df['sign_flip_rate'] > 0.1).mean() * 100
    ax.text(0.95, 0.95, f'{pct_flips:.1f}% of edges\nflip sign >10%', 
            transform=ax.transAxes, ha='right', va='top', fontsize=9)
    
    # 1C: SD(r) vs |mean_r| — weak correlations should be most unstable
    ax = axes[0, 2]
    ax.scatter(stability_df['mean_r'].abs(), stability_df['sd_r'], 
               alpha=0.1, s=3, color='#333333')
    ax.set_xlabel('|mean(r)|')
    ax.set_ylabel('SD(r)')
    ax.set_title('C   Stability vs correlation strength')
    ax.spines[['top', 'right']].set_visible(False)
    
    # ── Row 2: Comparison with dGCNA edges ──
    if merged_df is not None:
        # 2A: SD(r) for dGCNA significant vs non-significant
        ax = axes[1, 0]
        sig = merged_df[merged_df['dgcna_significant'] == True]['sd_r']
        nonsig = merged_df[merged_df['dgcna_significant'] == False]['sd_r']
        ax.hist(nonsig, bins=50, alpha=0.5, density=True, color='#4575b4', 
                label=f'Non-sig (n={len(nonsig)})')
        ax.hist(sig, bins=50, alpha=0.5, density=True, color='#d73027', 
                label=f'dGCNA sig (n={len(sig)})')
        ax.set_xlabel('SD(r) across subsamples')
        ax.set_ylabel('Density')
        ax.set_title('D   Stability: dGCNA sig vs non-sig edges')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        
        # 2B: Sign-flip rate for sig vs nonsig
        ax = axes[1, 1]
        data_plot = merged_df[['sign_flip_rate', 'dgcna_significant']].copy()
        data_plot['group'] = data_plot['dgcna_significant'].map(
            {True: 'dGCNA significant', False: 'Non-significant'}
        )
        sns.boxplot(data=data_plot, x='group', y='sign_flip_rate', ax=ax,
                    palette=['#4575b4', '#d73027'])
        ax.set_xlabel('')
        ax.set_ylabel('Sign-flip rate')
        ax.set_title('E   Sign instability: dGCNA sig vs non-sig')
        ax.spines[['top', 'right']].set_visible(False)
        
        # 2C: |Δr| vs SD(r)
        ax = axes[1, 2]
        ax.scatter(merged_df['sd_r'], merged_df['abs_delta_r'], 
                   alpha=0.1, s=3, color='#333333')
        ax.set_xlabel('SD(r) across subsamples')
        ax.set_ylabel('|Δr| (dGCNA)')
        ax.set_title('F   Correlation shift vs instability')
        ax.spines[['top', 'right']].set_visible(False)
    else:
        for ax in axes[1, :]:
            ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{save_prefix}.pdf', bbox_inches='tight')
    print(f"Saved {save_prefix}.png/.pdf")
    plt.show()


def run_stability_analysis(residuals_df, obs_df, genes=None, edge_df=None,
                           condition_col='Disease', conditions=None,
                           frac=0.8, n_iter=100, seed=42):
    """
    Full stability analysis pipeline.
    
    Parameters
    ----------
    residuals_df : pd.DataFrame (cells x genes) - LMM residuals
    obs_df : pd.DataFrame with condition column
    genes : list, optional - subset of genes to analyze
    edge_df : pd.DataFrame, optional - your existing dGCNA edge results
    condition_col : str
    conditions : list of conditions to test (default: all)
    frac : float - fraction of cells to keep per subsample
    n_iter : int - number of iterations
    seed : int
    
    Returns
    -------
    dict with:
        'corr_samples': dict of condition -> (n_iter, n_genes, n_genes) arrays
        'stability_ctrl': stability metrics for control
        'stability_t2d': stability metrics for T2D
        'merged': merged with edge_df if provided
    """
    # Subset genes
    if genes is not None:
        genes_in_data = [g for g in genes if g in residuals_df.columns]
        print(f"Using {len(genes_in_data)} / {len(genes)} genes")
        residuals_df = residuals_df[genes_in_data]
    
    gene_names = residuals_df.columns.tolist()
    
    # Run subsampling
    corr_samples = subsample_correlations(
        residuals_df, obs_df, condition_col=condition_col,
        conditions=conditions, frac=frac, n_iter=n_iter, seed=seed
    )
    
    # Compute stability metrics per condition
    stability_results = {}
    for cond, samples in corr_samples.items():
        print(f"\nComputing stability metrics for {cond}...")
        stability_results[cond] = compute_stability_metrics(samples, gene_names)
    
    # Use first condition for main plots (or combine)
    # Combine: take the max SD across conditions per edge (most conservative)
    all_stability = list(stability_results.values())
    if len(all_stability) == 1:
        combined_stability = all_stability[0]
    else:
        # Merge and take max SD per edge
        combined_stability = all_stability[0].copy()
        for other in all_stability[1:]:
            merged_tmp = combined_stability.merge(
                other[['gene1', 'gene2', 'sd_r', 'sign_flip_rate']], 
                on=['gene1', 'gene2'], suffixes=('', '_other')
            )
            combined_stability['sd_r'] = merged_tmp[['sd_r', 'sd_r_other']].max(axis=1)
            combined_stability['sign_flip_rate'] = merged_tmp[['sign_flip_rate', 'sign_flip_rate_other']].max(axis=1)
    
    # Compute Δr stability if we have exactly two conditions
    delta_stability = None
    cond_keys = list(corr_samples.keys())
    if len(cond_keys) == 2:
        print(f"\nComputing Δr stability ({cond_keys[1]} − {cond_keys[0]})...")
        delta_stability = compute_delta_r_stability(
            corr_samples, conditions=(cond_keys[0], cond_keys[1]),
            gene_names=gene_names
        )

    # Merge with dGCNA edges if provided
    merged = None
    if edge_df is not None:
        merged = compare_stability_to_dgcna(combined_stability, edge_df)
        # Also attach Δr stability to merged
        if delta_stability is not None:
            delta_stability['pair'] = delta_stability.apply(
                lambda r: tuple(sorted([r['gene1'], r['gene2']])), axis=1
            )
            merged = merged.merge(
                delta_stability[['pair', 'mean_delta_r', 'sd_delta_r',
                                 'sign_flip_rate_delta_r', 'ci_lower',
                                 'ci_upper', 'ci_width']],
                on='pair', how='left'
            )
    
    # Plot
    plot_stability(combined_stability, merged)
    
    # Print summary stats
    print("\n" + "=" * 60)
    print("STABILITY SUMMARY (per-condition r)")
    print("=" * 60)
    print(f"Total edges analyzed: {len(combined_stability)}")
    print(f"Median SD(r): {combined_stability['sd_r'].median():.4f}")
    print(f"Mean SD(r):   {combined_stability['sd_r'].mean():.4f}")
    print(f"Edges with sign-flip rate > 10%: {(combined_stability['sign_flip_rate'] > 0.1).sum()} "
          f"({100*(combined_stability['sign_flip_rate'] > 0.1).mean():.1f}%)")
    print(f"Edges with sign-flip rate > 25%: {(combined_stability['sign_flip_rate'] > 0.25).sum()} "
          f"({100*(combined_stability['sign_flip_rate'] > 0.25).mean():.1f}%)")
    print(f"Edges with SD(r) > 0.05: {(combined_stability['sd_r'] > 0.05).sum()} "
          f"({100*(combined_stability['sd_r'] > 0.05).mean():.1f}%)")

    if delta_stability is not None:
        print("\n" + "=" * 60)
        print("STABILITY SUMMARY (Δr = r_case − r_ctrl)")
        print("=" * 60)
        print(f"Median SD(Δr): {delta_stability['sd_delta_r'].median():.4f}")
        print(f"Mean SD(Δr):   {delta_stability['sd_delta_r'].mean():.4f}")
        print(f"Median 95% CI width: {delta_stability['ci_width'].median():.4f}")
        print(f"Mean 95% CI width:   {delta_stability['ci_width'].mean():.4f}")
        print(f"Edges with Δr sign-flip rate > 10%: "
              f"{(delta_stability['sign_flip_rate_delta_r'] > 0.1).sum()} "
              f"({100*(delta_stability['sign_flip_rate_delta_r'] > 0.1).mean():.1f}%)")
        print(f"Edges with Δr sign-flip rate > 25%: "
              f"{(delta_stability['sign_flip_rate_delta_r'] > 0.25).sum()} "
              f"({100*(delta_stability['sign_flip_rate_delta_r'] > 0.25).mean():.1f}%)")
    
    if merged is not None:
        sig_mask = merged['dgcna_significant']
        print(f"\ndGCNA significant edges:")
        print(f"  Median SD(r): {merged.loc[sig_mask, 'sd_r'].median():.4f}")
        print(f"  Sign-flip rate (r) > 10%: {(merged.loc[sig_mask, 'sign_flip_rate'] > 0.1).sum()}")
        if delta_stability is not None and 'sd_delta_r' in merged.columns:
            print(f"  Median SD(Δr): {merged.loc[sig_mask, 'sd_delta_r'].median():.4f}")
            print(f"  Median 95% CI width(Δr): {merged.loc[sig_mask, 'ci_width'].median():.4f}")
            print(f"  Sign-flip rate (Δr) > 10%: "
                  f"{(merged.loc[sig_mask, 'sign_flip_rate_delta_r'] > 0.1).sum()} "
                  f"({100*(merged.loc[sig_mask, 'sign_flip_rate_delta_r'] > 0.1).mean():.1f}%)")
            print(f"  Edges where CI width > |Δr|: "
                  f"{(merged.loc[sig_mask, 'ci_width'] > merged.loc[sig_mask, 'abs_delta_r']).sum()} "
                  f"({100*(merged.loc[sig_mask, 'ci_width'] > merged.loc[sig_mask, 'abs_delta_r']).mean():.1f}%)")
        print(f"Non-significant edges:")
        print(f"  Median SD(r): {merged.loc[~sig_mask, 'sd_r'].median():.4f}")
        print(f"  Sign-flip rate (r) > 10%: {(merged.loc[~sig_mask, 'sign_flip_rate'] > 0.1).sum()}")
        if delta_stability is not None and 'sd_delta_r' in merged.columns:
            print(f"  Median SD(Δr): {merged.loc[~sig_mask, 'sd_delta_r'].median():.4f}")
            print(f"  Median 95% CI width(Δr): {merged.loc[~sig_mask, 'ci_width'].median():.4f}")
            print(f"  Sign-flip rate (Δr) > 10%: "
                  f"{(merged.loc[~sig_mask, 'sign_flip_rate_delta_r'] > 0.1).sum()} "
                  f"({100*(merged.loc[~sig_mask, 'sign_flip_rate_delta_r'] > 0.1).mean():.1f}%)")
            print(f"  Edges where CI width > |Δr|: "
                  f"{(merged.loc[~sig_mask, 'ci_width'] > merged.loc[~sig_mask, 'abs_delta_r']).sum()} "
                  f"({100*(merged.loc[~sig_mask, 'ci_width'] > merged.loc[~sig_mask, 'abs_delta_r']).mean():.1f}%)")
    
    return {
        'corr_samples': corr_samples,
        'stability': stability_results,
        'combined_stability': combined_stability,
        'delta_stability': delta_stability,
        'merged': merged,
    }
