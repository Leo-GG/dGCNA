"""
Regression tests for the dGCNA reimplementation in `run_dgcna_correlation.py`.

The LMM tests check the vectorised profiled-REML solver against a brute-force
REML fit written independently with explicit covariance matrices, which is the
same criterion `lme4::lmer` optimises. The bootstrap tests pin down the
group-size convention and the sign-aware comparison taken from the authors'
`CompareDifferences` / `GetMaskFromRandomDonors` R functions.

Run with:  python -m pytest tests/test_dgcna_lmm.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_dgcna_correlation import (  # noqa: E402
    _lmm_residuals_matrix,
    _reml_objective,
    build_differential_network,
    filter_network,
    fit_reml,
    run_bootstrap_streaming,
)

LOG_GAMMA_BOUND = 18.0


# ---------------------------------------------------------------------------
# Brute-force reference implementation
# ---------------------------------------------------------------------------

def _brute_reml(y, donor_codes, n_donors):
    """
    Reference REML fit of y_ij = mu + u_i + e_ij using explicit matrices.

    Minimises  log|V| + log|X' V^-1 X| + (N - p) log( r' V^-1 r )
    over gamma = sigma2_u / sigma2_e, with V = I + gamma Z Z'.

    Because gamma is optimised on the log scale, this reference cannot represent
    the boundary solution gamma = 0; when the true optimum is at the boundary it
    returns a gamma pinned near exp(-LOG_GAMMA_BOUND). Callers must check for
    that case rather than treating the result as interior.

    Returns (gamma_hat, residuals).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    Z = np.zeros((n, n_donors))
    Z[np.arange(n), donor_codes] = 1.0
    X = np.ones((n, 1))
    ZZt = Z @ Z.T

    def objective(log_gamma):
        gamma = np.exp(log_gamma)
        V = np.eye(n) + gamma * ZZt
        Vi = np.linalg.inv(V)
        XtViX = X.T @ Vi @ X
        beta = solve(XtViX, X.T @ Vi @ y)
        r = y - X @ beta
        rss = float(r @ Vi @ r)
        return (np.linalg.slogdet(V)[1]
                + np.linalg.slogdet(XtViX)[1]
                + (n - 1) * np.log(rss))

    opt = minimize_scalar(objective,
                          bounds=(-LOG_GAMMA_BOUND, LOG_GAMMA_BOUND),
                          method='bounded', options={'xatol': 1e-12})
    gamma = float(np.exp(opt.x))

    V = np.eye(n) + gamma * ZZt
    Vi = np.linalg.inv(V)
    mu = float(np.ravel(solve(X.T @ Vi @ X, X.T @ Vi @ y))[0])
    counts = Z.sum(axis=0)
    donor_means = (Z.T @ y) / counts
    u = gamma * counts * (donor_means - mu) / (1.0 + gamma * counts)
    return gamma, y - mu - Z @ u


def _simulate(n_genes=12, cells_per_donor=(31, 45, 60, 88, 95, 40, 37),
              sigma_u=0.8, sigma_e=1.3, seed=0):
    rng = np.random.default_rng(seed)
    n_donors = len(cells_per_donor)
    donor_codes = np.repeat(np.arange(n_donors), cells_per_donor)
    n_cells = donor_codes.size

    mu = rng.normal(4.0, 1.0, size=n_genes)
    u = rng.normal(0.0, sigma_u, size=(n_donors, n_genes))
    e = rng.normal(0.0, sigma_e, size=(n_cells, n_genes))
    Y = mu[None, :] + u[donor_codes] + e
    return Y, donor_codes, n_donors


# ---------------------------------------------------------------------------
# LMM tests
# ---------------------------------------------------------------------------

def _objective_at(gamma, y, donor_codes, n_donors):
    """Evaluate the profiled REML criterion at a single gamma for one gene."""
    n = y.size
    Z = np.zeros((n, n_donors))
    Z[np.arange(n), donor_codes] = 1.0
    counts = Z.sum(axis=0)
    donor_means = ((Z.T @ y) / counts)[:, None]
    ssw = np.array([float(y @ y - (counts * donor_means[:, 0]**2).sum())])
    obj, _ = _reml_objective(np.array([[gamma]]), counts, donor_means, ssw, n)
    return float(obj[0, 0])


def test_reml_optimum_is_no_worse_than_brute_force():
    """
    The vectorised solver must find an optimum at least as good as an
    independently coded brute-force REML fit, for every gene.
    """
    Y, donor_codes, n_donors = _simulate(seed=1)
    gamma_fit, _ = fit_reml(Y, donor_codes, n_donors)

    for g in range(Y.shape[1]):
        gamma_ref, _ = _brute_reml(Y[:, g], donor_codes, n_donors)
        ours = _objective_at(gamma_fit[g], Y[:, g], donor_codes, n_donors)
        theirs = _objective_at(gamma_ref, Y[:, g], donor_codes, n_donors)
        assert ours <= theirs + 1e-9, f"gene {g}: {ours} > {theirs}"


def test_reml_residuals_match_brute_force_when_interior():
    """
    Where the brute-force optimum is interior (not pinned at the log-gamma
    bound), the residuals must agree to well below any level that could affect
    a Pearson correlation.
    """
    Y, donor_codes, n_donors = _simulate(seed=1)
    got = _lmm_residuals_matrix(Y, donor_codes, n_donors)

    n_checked = 0
    for g in range(Y.shape[1]):
        gamma_ref, expected = _brute_reml(Y[:, g], donor_codes, n_donors)
        if gamma_ref < 10 * np.exp(-LOG_GAMMA_BOUND):
            continue                      # boundary case, reference unusable
        n_checked += 1
        assert np.allclose(got[:, g], expected, atol=1e-6), f"gene {g}"

    assert n_checked >= 10, "expected most simulated genes to be interior"


def test_reml_recovers_variance_ratio():
    """With a real donor effect, the fitted gamma should track the truth."""
    sigma_u, sigma_e = 1.0, 2.0
    true_gamma = (sigma_u / sigma_e) ** 2
    Y, donor_codes, n_donors = _simulate(
        n_genes=60, sigma_u=sigma_u, sigma_e=sigma_e, seed=2
    )
    gamma_fit, _ = fit_reml(Y, donor_codes, n_donors)
    # Only 7 donors, so the estimate is noisy; check the median is in range
    assert 0.25 * true_gamma < float(np.median(gamma_fit)) < 4 * true_gamma


def test_residuals_remove_donor_means():
    """Donor means of the residuals must be shrunk far below the raw spread."""
    Y, donor_codes, n_donors = _simulate(sigma_u=1.5, sigma_e=0.5, seed=3)
    res = _lmm_residuals_matrix(Y, donor_codes, n_donors)

    def donor_mean_spread(M):
        return np.array([
            M[donor_codes == d].mean(axis=0) for d in range(n_donors)
        ]).std(axis=0)

    assert np.all(donor_mean_spread(res) < 0.2 * donor_mean_spread(Y))


def test_exchangeable_donors_shrink_random_effect_to_zero():
    """
    With no true donor effect, REML should drive gamma to (or very near) the
    boundary, so the residuals reduce to grand-mean-centred data. The grid in
    `fit_reml` includes gamma = 0 exactly, which is the singular fit `lmer`
    reports in this situation.
    """
    rng = np.random.default_rng(4)
    donor_codes = np.repeat(np.arange(6), 50)
    Y = rng.normal(0.0, 1.0, size=(donor_codes.size, 5))

    gamma_fit, mu_fit = fit_reml(Y, donor_codes, 6)
    assert np.all(gamma_fit < 0.1), gamma_fit
    assert np.allclose(mu_fit, Y.mean(axis=0), atol=1e-2)

    res = _lmm_residuals_matrix(Y, donor_codes, 6)
    # Genes at the exact boundary must reduce to plain grand-mean centring
    at_boundary = gamma_fit == 0.0
    if at_boundary.any():
        assert np.allclose(res[:, at_boundary],
                           (Y - Y.mean(axis=0))[:, at_boundary], atol=1e-12)
    assert np.abs(res - (Y - Y.mean(axis=0))).max() < 0.25


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------

def _toy_cohort(n_ctrl_donors=5, n_case_donors=3, cells=35, n_genes=4, seed=5):
    rng = np.random.default_rng(seed)
    donors, disease = [], []
    for i in range(n_ctrl_donors):
        donors += [f"C{i}"] * cells
        disease += ["normal"] * cells
    for i in range(n_case_donors):
        donors += [f"T{i}"] * cells
        disease += ["type II diabetes"] * cells

    obs = pd.DataFrame({'Donor': donors, 'Disease': disease},
                       index=[f"cell{i}" for i in range(len(donors))])
    expr = pd.DataFrame(rng.normal(size=(len(donors), n_genes)),
                        index=obs.index,
                        columns=[f"G{i}" for i in range(n_genes)])
    return expr, obs


def test_permutation_null_preserves_group_sizes():
    """g1 must get n_ctrl donors and g2 must get n_cases, as in the R code."""
    expr, obs = _toy_cohort(n_ctrl_donors=5, n_case_donors=3)
    seen = []

    import run_dgcna_correlation as mod
    original = mod._lmm_residuals_matrix

    def spy(Y, codes, n_donors, **kw):
        seen.append(n_donors)
        return original(Y, codes, n_donors, **kw)

    mod._lmm_residuals_matrix = spy
    try:
        diff = pd.DataFrame(np.zeros((4, 4)),
                            index=expr.columns, columns=expr.columns)
        run_bootstrap_streaming(expr, obs, diff, n_iterations=3,
                                replacement=False, seed=0)
    finally:
        mod._lmm_residuals_matrix = original

    # Calls alternate g1, g2, g1, g2, ...
    assert seen[0::2] == [5, 5, 5], seen
    assert seen[1::2] == [3, 3, 3], seen


def test_permutation_null_rejects_mismatched_donor_counts():
    expr, obs = _toy_cohort()
    obs = obs.copy()
    # Introduce a donor that belongs to neither group
    obs.loc[obs.index[:10], 'Disease'] = 'other'
    obs.loc[obs.index[:10], 'Donor'] = 'X0'
    diff = pd.DataFrame(np.zeros((4, 4)),
                        index=expr.columns, columns=expr.columns)
    with pytest.raises(ValueError, match="Permutation null"):
        run_bootstrap_streaming(expr, obs, diff, n_iterations=2,
                                replacement=False, seed=0)


def test_mask_is_sign_aware():
    """
    An observed difference at the extreme of its own sign must reach mask ~ 1,
    matching `CompareDifferences`.
    """
    expr, obs = _toy_cohort(n_genes=3, seed=7)
    _, _, diff = build_differential_network(
        pd.DataFrame(expr.values, index=expr.index, columns=expr.columns), obs
    )
    # Force one edge strongly positive and one strongly negative
    diff.iloc[0, 1] = diff.iloc[1, 0] = 10.0
    diff.iloc[0, 2] = diff.iloc[2, 0] = -10.0

    mask = run_bootstrap_streaming(expr, obs, diff, n_iterations=8,
                                   replacement=False, seed=1)
    assert mask.iloc[0, 1] == 1.0
    assert mask.iloc[0, 2] == 1.0
    # The diagonal must be exactly zero, not round-off noise, so that it can
    # never win the one-sided comparison against a zero random diagonal
    assert np.all(np.diag(diff.values) == 0.0)
    assert np.all(np.diag(mask.values) == 0.0)


def test_filter_network_returns_raw_deltas():
    """Surviving weights must be unscaled Delta r values."""
    idx = ['A', 'B', 'C']
    diff = pd.DataFrame([[0.0, 0.4, -0.2],
                         [0.4, 0.0, 0.1],
                         [-0.2, 0.1, 0.0]], index=idx, columns=idx)
    mask = pd.DataFrame([[0.0, 0.99, 0.99],
                         [0.99, 0.0, 0.50],
                         [0.99, 0.50, 0.0]], index=idx, columns=idx)
    out = filter_network(diff, mask, threshold=0.975)

    assert out.loc['A', 'B'] == pytest.approx(0.4)   # kept, not rescaled
    assert out.loc['A', 'C'] == pytest.approx(-0.2)  # kept, sign preserved
    assert out.loc['B', 'C'] == 0.0                  # masked out
