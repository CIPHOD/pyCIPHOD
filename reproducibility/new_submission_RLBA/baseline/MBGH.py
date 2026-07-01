"""
Implementation of Algorithm 1 from:
  "Identifying Causal Changes Between Linear Structural Equation Models"
  Malik, Bello, Ghoshal, Honorio

The algorithm directly estimates the Difference DAG (DDAG) between two
linear SEMs without estimating the individual DAG structures.

Core idea
---------
Two linear SEMs share a topological ordering.  Their difference precision
matrix Delta_Omega = Omega1 - Omega2 encodes the causal changes:

  * Terminal vertices of the DDAG (nodes with no outgoing changed edges)
    satisfy  Delta_Omega_{i,i} = 0  (Proposition 1 / Lemma 1).

  * For such a terminal i, the incoming edges are read from the
    off-diagonal entries:  Delta_Omega_{i,j} ∝ Delta_B_{i,j} / sigma_i^2

  * We iteratively peel off identified terminals, re-estimating
    Delta_Omega on the reduced variable set, until all variables are
    processed.  (Section 4.3 — "removing terminals all-at-once")

Identifiability conditions (Theorem 3)
---------------------------------------
  Assumption 2 — the minimal topological layering of the DDAG is a valid
                 topological layering of the union graph G_union.
  Assumption 3 — the DN-level of every vertex is >= its topological level.

Two estimators for Delta_Omega are provided:
  'admm'   — sparse ADMM estimator (Jiang et al. 2018); recommended in
             general as it exploits sparsity and has finite-sample guarantees.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding operator S_tau(X)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def estimate_delta_omega_admm(
    Sigma1: np.ndarray,
    Sigma2: np.ndarray,
    lam: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    ADMM estimator (Jiang et al. 2018): A Direct Approach for Sparse Quadratic Discriminant Analysis

    Variables:
        Ω (Omega)  : main variable (DeltaOmega)
        Ψ (Psi)    : auxiliary variable (for L1 penalty)
        Λ (Lambda) : dual variable

    Minimises:
        (1/2) Tr(Ω^T Σ1 Ω Σ2)
        + Tr(Ω (Σ1 - Σ2))
        + λ ||Ω||_1
    """

    m = Sigma1.shape[0]

    # --- Step 1: SVD / eigen decompositions ---
    d1, U1 = np.linalg.eigh(Sigma1)   # Σ1 = U1 diag(d1) U1^T
    d2, U2 = np.linalg.eigh(Sigma2)   # Σ2 = U2 diag(d2) U2^T

    # B_{jk} = 1 / (d1_j * d2_k + rho)
    B = 1.0 / (np.outer(d1, d2) + rho)

    # --- Initialize variables ---
    Omega = np.zeros((m, m))
    Psi   = np.zeros((m, m))
    Lambda = np.zeros((m, m))

    for _ in range(max_iter):
        Omega_prev = Omega.copy()
        Psi_prev = Psi.copy()

        # --- Step 2: Ω-update ---
        # A = (Σ1 - Σ2) - Λ + ρΨ
        A = (Sigma1 - Sigma2) - Lambda + rho * Psi

        # Ω = U1 [ B ∘ (U1^T A U2) ] U2^T
        A_tilde = U1.T @ A @ U2
        Omega = U1 @ (B * A_tilde) @ U2.T

        # --- Step 3: Ψ-update (soft-thresholding) ---
        Psi = soft_threshold(Omega + Lambda / rho, lam / rho)

        # --- Step 4: Λ-update ---
        Lambda = Lambda + rho * (Omega - Psi)

        # --- Convergence ---
        primal_res = np.linalg.norm(Omega - Psi, ord="fro")
        dual_change = np.linalg.norm(Psi - Psi_prev, ord="fro")
        omega_change = np.linalg.norm(Omega - Omega_prev, ord="fro")

        if max(primal_res, dual_change, omega_change) < tol:
            break

    DeltaOmega = 0.5 * (Psi + Psi.T)
    return DeltaOmega



# ============================================================
# Algorithm 1
# ============================================================

def learn_ddag(
    X1: np.ndarray,
    X2: np.ndarray,
    estimator: str = "admm",
    lam: float = 0.1,
    diagonal_tol: Optional[float] = None,
    edge_tol: Optional[float] = None, 
    diagonal_tol_fraction: float = 0.05,
    edge_tol_fraction: float = 0.05,
    admm_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    Algorithm 1 — directly estimate the Difference DAG (DDAG) between two
    linear SEMs, without estimating either individual DAG structure.

    Parameters
    ----------
    X1 : ndarray, shape (n1, p)
        i.i.d. samples from the first SEM.
    X2 : ndarray, shape (n2, p)
        i.i.d. samples from the second SEM.
    estimator : {'admm'}
        Estimator used for Delta_Omega at each iteration.
        'admm'  — sparse ADMM
    lam : float
        L1 regularisation for the ADMM estimator.
        Rule of thumb: set proportional to sqrt(log p / n).
    diagonal_tol : float or None
        Absolute threshold for declaring Delta_Omega_{i,i} = 0.
        If None (default), an adaptive per-iteration threshold is used:
            tol = diagonal_tol_fraction * max(|Delta_Omega|)
    diagonal_tol_fraction : float
        Fraction of the scale of Delta_Omega used as the adaptive threshold.
        Default 0.05 (5 % of the maximum absolute entry).
    admm_kwargs : dict, optional
        Extra keyword arguments forwarded to estimate_delta_omega_admm
        (rho, max_iter, tol).

    Returns
    -------
    ddag : ndarray, shape (p, p), dtype bool
        Adjacency matrix of the estimated DDAG.
        ddag[i, j] = True  means there is a directed edge  i ← j  in the DDAG
        (equivalently, B1[i,j] ≠ B2[i,j], following the paper's convention
        where B[i,j] is the weight of the edge j → i).

    Notes
    -----
    Under Assumptions 2 & 3 of the paper and with exact covariances,
    Theorem 3 guarantees that the returned DDAG equals supp(B1 - B2).
    With sample covariances, correctness follows from the finite-sample
    guarantees of the chosen Delta_Omega estimator.
    """
    if admm_kwargs is None:
        admm_kwargs = {}

    _, p = X1.shape

    # Compute sample covariance matrices (unbiased, ddof=1)
    Sigma1_hat = np.cov(X1, rowvar=False)
    Sigma2_hat = np.cov(X2, rowvar=False)

    # Adjacency matrix of the output DDAG (original node indices)
    ddag = np.zeros((p, p), dtype=bool)

    # Active variable set (original indices); processed all-at-once per layer
    active = list(range(p))

    # ------------------------------------------------------------------ #
    # Main loop — Algorithm 1                                             #
    # ------------------------------------------------------------------ #
    while len(active) > 1:  # == line 2 
        m = len(active)     # Nombre de node restants

        # Extract sub-covariance matrices for active variables
        S1 = Sigma1_hat[np.ix_(active, active)]
        S2 = Sigma2_hat[np.ix_(active, active)]

        # Step 3 — estimate Delta_Omega over V          ==   line 3
        if estimator == "admm":
            DeltaOmega = estimate_delta_omega_admm(S1, S2, lam=lam,                             
                                                   **admm_kwargs)
        else:
            raise ValueError(
                f"Unknown estimator '{estimator}'. Choose 'admm'."
            )
        
        DeltaOmega = 0.5 * (DeltaOmega + DeltaOmega.T)

        # Determine threshold for "approximately zero"                                          # Doesn't 
        scale = np.max(np.abs(DeltaOmega))
        diag_tol = diagonal_tol if diagonal_tol is not None else (
            diagonal_tol_fraction * scale if scale > 0 else 1e-8
        )
        e_tol = edge_tol if edge_tol is not None else (
            edge_tol_fraction * scale if scale > 0 else 1e-8
        )

        # Step 4 — S ← {i | (Delta_Omega)_{i,i} ≈ 0}
        # diag_abs = np.abs(np.diag(DeltaOmega))
        # terminal_local = [i for i in range(m) if diag_abs[i] < tol]                             # line 4 : set de terminal

        diag_abs = np.abs(np.diag(DeltaOmega))
        terminal_local = [i for i in range(m) if diag_abs[i] < diag_tol]

        # Guard: if no terminals found the algorithm cannot progress.
        # This can happen when estimates are noisy; warn and stop.
        if len(terminal_local) == 0:
            warnings.warn(
                f"No terminal vertices identified in an active set of size {m}. "
                "Stopping early. Consider: more samples, smaller lam, or "
                "adjusting diagonal_tol_fraction.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        terminal_local_set = set(terminal_local)

        # Steps 5–12 — identify incoming edges for each terminal vertex
        for i_loc in terminal_local:                                                            # line 5 : boucle sur 
            i_glob = active[i_loc]                                                              # prendre un des node dans le set
            for j_loc in range(m):                                                              # on loop sur les j, au lieu de créer Nj
                if j_loc in terminal_local_set:
                    continue
                if abs(DeltaOmega[i_loc, j_loc]) > e_tol:                                         # les j dont delta_omega est different de zero (donc plus grand que tolerance)
                    j_glob = active[j_loc]
                    # Condition (Algorithm 1, line 8):
                    #   edge not already recorded  AND  j is not itself a terminal
                    # Edge:  i_glob ← j_glob
                    ddag[i_glob, j_glob] = True                                             # line 9

        # Step 13 — remove terminals from active set
        terminal_global = {active[i] for i in terminal_local}                                   # remove terminals all at once
        active = [v for v in active if v not in terminal_global]                                # keeping the remaining active edges

    return ddag


# ============================================================
# Utility: generate data from a linear SEM
# ============================================================

def sample_sem(
    B: np.ndarray,
    D: np.ndarray,
    n: int,
    noise: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw n i.i.d. samples from the linear SEM  X = B X + epsilon.

    The model implies  X = (I - B)^{-1} epsilon.

    Parameters
    ----------
    B     : (p, p) autoregression matrix — B[i, j] is the weight of edge j → i.
    D     : (p, p) diagonal noise-variance matrix — D[i, i] = sigma_i^2.
    n     : number of samples to draw.
    noise : distribution for the noise terms: 'gaussian' | 'uniform' | 'gumbel'.
    rng   : numpy random Generator (created if not provided).

    Returns
    -------
    X : (n, p) ndarray of samples.
    """
    rng = rng or np.random.default_rng()
    p = B.shape[0]
    sigmas = np.sqrt(np.diag(D))

    if noise == "gaussian":
        eps = rng.normal(0.0, 1.0, size=(n, p)) * sigmas
    elif noise == "uniform":
        # Uniform on [-sqrt(3), sqrt(3)] has variance 1
        eps = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(n, p)) * sigmas
    elif noise == "gumbel":
        # Gumbel(0,1) has mean = Euler-Mascheroni constant ≈ 0.5772; centre it
        raw = rng.gumbel(0.0, 1.0, size=(n, p))
        raw -= raw.mean(axis=0)
        eps = raw * sigmas
    else:
        raise ValueError(
            f"Unknown noise type '{noise}'. Choose 'gaussian', 'uniform', or 'gumbel'."
        )

    IminusB_inv = np.linalg.inv(np.eye(p) - B)
    return (IminusB_inv @ eps.T).T  # shape (n, p)


if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    # Example 1 — 4-node toy (analogous to Figure 1 in the paper)        #
    # ------------------------------------------------------------------ #
    # print("=" * 62)
    # print("Example 1 — 4-node toy SEM pair")
    # print("=" * 62)
    # print("Nodes: X1(0), X2(1), X3(2), X4(3)  (0-indexed)")
    # print()

    # p = 4
    # sigma2 = 0.5

    # # SEM 1: X1→X2 (1), X4→X2 (2), X4→X3 (1)
    # B1 = np.zeros((p, p))
    # B1[1, 0] = 1.0   # X1 → X2
    # B1[1, 3] = 2.0   # X4 → X2
    # B1[2, 3] = 1.0   # X4 → X3

    # # SEM 2: same except X4→X3 weight changes 1 → 2
    # B2 = B1.copy()
    # B2[2, 3] = 2.0

    # D = np.diag([sigma2] * p)
    # true_ddag = (B1 != B2)

    # print("True B1 - B2:")
    # print(np.round(B1 - B2, 2))
    # print()
    # print("True DDAG  (ddag[i,j]=1 means edge i ← j):")
    # print(true_ddag.astype(int))
    # print()

    # rng = np.random.default_rng(42)
    # n = 50_000
    # X1 = sample_sem(B1, D, n, noise="gaussian", rng=rng)
    # X2 = sample_sem(B2, D, n, noise="gaussian", rng=rng)

    # ddag = learn_ddag(X1, X2, estimator="admm", lam=0.01)

    # print(f"Estimated DDAG  (n={n:,}, ADMM, lam=0.01):")
    # print(ddag.astype(int))
    # print()
    # correct = np.array_equal(ddag, true_ddag)
    # print(f"Exact recovery: {'YES ✓' if correct else 'NO ✗'}")

    # ------------------------------------------------------------------ #
    # Example 2 — larger random Erdős–Rényi experiment                   #
    # ------------------------------------------------------------------ #
    print()
    print("=" * 62)
    print("Example 2 — random experiment (p=20, Gumbel noise)")
    print("=" * 62)

    p = 20
    n = 10_000

    rng = np.random.default_rng(7)

    def random_dag(p, edge_prob, w_lo, w_hi, rng):
        """Lower-triangular (column index < row index) random DAG."""
        B = np.zeros((p, p))
        for i in range(p):
            for j in range(i):      # j < i  means j → i
                if rng.random() < edge_prob:
                    sign = rng.choice([-1, 1])
                    B[i, j] = sign * rng.uniform(w_lo, w_hi)
        return B

    B1 = random_dag(p, 0.4, 0.3, 0.8, rng)
    B2 = B1.copy()
    for i in range(p):
        for j in range(i):
            if rng.random() < 0.10:   # ~10 % of edges change
                sign = rng.choice([-1, 1])
                B2[i, j] = sign * rng.uniform(0.3, 0.8)

    D = np.diag(rng.uniform(0.25, 0.5, size=p))
    true_ddag = (B1 != B2)

    X1 = sample_sem(B1, D, n, noise="gaussian", rng=rng)
    X2 = sample_sem(B2, D, n, noise="gaussian", rng=rng)

    # Lambda heuristic (Section 4.6 of the paper): ~ sqrt(log p / n)
    lam = np.sqrt(np.log(p) / n) * 2

    ddag = learn_ddag(X1, X2, estimator="admm", lam=lam)

    #print(ddag)

    # TP = int(np.sum(ddag & true_ddag))
    # FP = int(np.sum(ddag & ~true_ddag))
    # FN = int(np.sum(~ddag & true_ddag))
    # precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    # recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # f1        = (2 * precision * recall / (precision + recall)
    #              if (precision + recall) > 0 else 0.0)

    print(f"True changed edges : {int(true_ddag.sum())}")
    print(f"Recovered edges    : {int(ddag.sum())}")
    # print(f"TP={TP}  FP={FP}  FN={FN}")
    # print(f"Precision : {precision:.3f}")
    # print(f"Recall    : {recall:.3f}")
    # print(f"F1 Score  : {f1:.3f}")
    # print()
    # print("(F1 improves with larger n; set n=100 000 for near-perfect recovery.)")