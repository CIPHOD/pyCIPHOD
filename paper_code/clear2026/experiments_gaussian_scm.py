#%%
# =========================================
# Imports and General Setup
# =========================================
import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

np.random.seed(2025)

# Add paths for DAG generators and baselines
root = Path(__file__).resolve().parent
sys.path.extend([
    str(root),
    str(root.parents[1] / "PyCIPHOD")
])

# Specific imports
from paper_code.clear2026.dags_generator import (
    random_DAG_identifiable_CDE,
    random_DAG_nonidentifiable_CDE
)
from baselines.Gupta_codes.ldecc import LDECCAlgorithm
from baselines.Gupta_codes.pc_alg import PCAlgorithm
from baselines.pyCausalFS.LSL.MBs.CMB.CMB import CMB
from baselines.pyCausalFS.LSL.MBs.MBbyMB import MBbyMB
from PyCIPHOD.causal_discovery.pc.pc import PC
from PyCIPHOD.causal_discovery.local.locpc import LocPC

# =========================================
# Utility Functions
# =========================================
def simulate_linearSCM_from_dag(dag, nb_obs=1, coef_range=(-1,1), sigma_range=(0.5,1)):
    """Simulate linear SCM data from a DAG with random coefficients."""
    nodes = list(nx.topological_sort(dag))
    p = len(nodes)
    coef = np.zeros((p,p))
    for j, vj in enumerate(nodes):
        for k, vk in enumerate(nodes):
            if vk in dag.predecessors(vj):
                while abs(coef[j,k]) < 0.2:
                    coef[j,k] = np.random.uniform(*coef_range)
    coef_inv = np.linalg.inv(np.eye(p) - coef)
    sigma_vec = np.random.uniform(*sigma_range, size=p)
    data = np.array([coef_inv @ np.random.normal(0, sigma_vec) for _ in range(nb_obs)])
    return pd.DataFrame(coef, index=nodes, columns=nodes), pd.DataFrame(data, columns=nodes)


def replace_indices_by_names(data, res):
    """Convert indices from CMB/MBbyMB outputs to variable names."""
    def idx_to_names(lst):
        return [data.columns[i] for i in lst] if lst else []
    return tuple(idx_to_names(lst) for lst in res[:4])


# =========================================
# CDE Estimation Methods
# =========================================
def locpc_CDE(data, treatment, outcome):
    alg = LocPC(data)
    res = alg.run_locPC_CDE(treatment, outcome)
    return {
        'identifiability': res['identifiability'],
        'adjustment_set': res['adjustment_set'],
        'nb_CI_tests': alg.nb_ci_tests
    }

def ldecc_CDE(data, treatment, outcome):
    alg = LDECCAlgorithm(treatment_node=outcome, outcome_node=outcome, use_ci_oracle=False)
    res = alg.run(data)
    ident = treatment in res['tmt_children'] or treatment not in (res['tmt_parents'] | res['tmt_children']) or not res['unoriented']
    return {
        'identifiability': ident,
        'adjustment_set': list(res['tmt_parents']) if ident else None,
        'nb_CI_tests': alg.nb_ci_tests
    }

def PC_CDE(data, treatment, outcome):
    pc_alg = PC(data)
    pc_alg.run()
    edges = pc_alg.cpdag.get_directed_edges()
    adj = pc_alg.cpdag.get_adjacencies(outcome)
    ident = (outcome, treatment) in edges or treatment not in adj or all(
        (n, outcome) in edges or (outcome, n) in edges for n in adj
    )
    adj_set = [p for (p, x) in edges if x==outcome] if ident else None
    return {'identifiability': ident, 'adjustment_set': adj_set, 'nb_CI_tests': pc_alg.nb_ci_tests}

def CMB_CDE(data, treatment, outcome):
    res = CMB(data, data.columns.get_loc(outcome), 0.05, is_discrete=False)
    parents, children, pc, unoriented = replace_indices_by_names(data, res)
    ident = (treatment in children) or (treatment not in pc) or not unoriented
    return {'identifiability': ident, 'adjustment_set': list(parents) if ident else None, 'nb_CI_tests': res[4]}

def MBbyMB_CDE(data, treatment, outcome):
    res = MBbyMB(data, data.columns.get_loc(outcome), 0.05, is_discrete=False)
    parents, children, pc, unoriented = replace_indices_by_names(data, res)
    ident = (treatment in children) or (treatment not in pc) or not unoriented
    return {'identifiability': ident, 'adjustment_set': list(parents) if ident else None, 'nb_CI_tests': res[4]}


# =========================================
# Experiment Runner
# =========================================
def run_experiments(dag_generator, methods, dag_sizes, n_experiments=50, m=1, n_samples=10000, count_non_identifiable=False):
    """
    Run experiments (identifiable or non-identifiable DAGs).
    count_non_identifiable=True → summary counts NON-identifiable proportion.
    """
    summary_rows, detailed_rows = [], []

    for size in dag_sizes:
        edge_prob = m / (size-1)
        print(f"\n=== DAG size {size}, edge prob ~{edge_prob:.3f}, sample size {n_samples} ===")

        ci_counts = {method: [] for method in methods}
        ident_counts = {method: 0 for method in methods}

        for i in range(n_experiments):
            print(f"--- Experiment {i+1}/{n_experiments} ---")
            g, y, x = dag_generator(size, edge_prob)
            _, data = simulate_linearSCM_from_dag(g, n_samples, sigma_range=(0.8,1))
            true_parents = list(g.predecessors(y))
            exp_id = f"{size}_{i}"

            for method in methods:
                print(f"Running method : {method}")
                res = {
                    'locpc': locpc_CDE,
                    'ldecc': ldecc_CDE,
                    'pc': PC_CDE,
                    'CMB': CMB_CDE,
                    'MBbyMB': MBbyMB_CDE
                }[method](data, x, y)

                # For non-identifiable experiments, count non-identifiable cases
                ident_counts[method] += 0 if count_non_identifiable else res['identifiability']
                if count_non_identifiable:
                    ident_counts[method] += 0 if res['identifiability'] else 1

                if res.get('nb_CI_tests') is not None:
                    ci_counts[method].append(res['nb_CI_tests'])

                detailed_rows.append({
                    'experiment_id': exp_id,
                    'dag_size': size,
                    'method': method,
                    'identifiability': res['identifiability'],
                    'adjustment_set': res['adjustment_set'],
                    'true_parents': true_parents,
                    'nb_CI_tests': res.get('nb_CI_tests')
                })

        # Summary
        for method in methods:
            proportion = ident_counts[method] / n_experiments
            summary_rows.append({
                'dag_size': size,
                'method': method,
                'identifiable_proportion': proportion,
                'avg_nb_CI_tests': np.mean(ci_counts[method]) if ci_counts[method] else None
            })
            label = "NON-identifiable" if count_non_identifiable else "Identifiable"
            print(f"{method.upper()}: {label} proportion {proportion:.2f}, Avg CI tests {np.mean(ci_counts[method]) if ci_counts[method] else None}")

    return pd.DataFrame(summary_rows), pd.DataFrame(detailed_rows)


# =========================================
# Run Experiments 
# =========================================
if __name__ == "__main__":
    N_SAMPLES = 5000
    N_EXP = 100
    M = 2

    os.makedirs("output_experiments_gaussian", exist_ok=True)

    # ----------------------------
    # Small DAGs (< 100 nodes)
    # ----------------------------
    small_sizes = [10, 20, 30, 40, 50]
    methods_small = ['locpc', 'ldecc', 'pc', 'CMB', 'MBbyMB']  # all baselines

    # Identifiable DAGs
    ID_summary_small, ID_detailed_small = run_experiments(
        random_DAG_identifiable_CDE, methods_small, small_sizes, N_EXP, M, N_SAMPLES, count_non_identifiable=False
    )
    ID_summary_small.to_csv("output_experiments_gaussian/summary_identifiable_small.csv", index=False)
    ID_detailed_small.to_csv("output_experiments_gaussian/final_results_identifiable_small.csv", index=False)

    # Non-identifiable DAGs
    NONID_summary_small, NONID_detailed_small = run_experiments(
        random_DAG_nonidentifiable_CDE, methods_small, small_sizes, N_EXP, M, N_SAMPLES, count_non_identifiable=True
    )
    NONID_summary_small.to_csv("output_experiments_gaussian/summary_nonidentifiable_small.csv", index=False)
    NONID_detailed_small.to_csv("output_experiments_gaussian/final_results_nonidentifiable_small.csv", index=False)

    # ----------------------------
    # Large DAGs (>= 100 nodes)
    # ----------------------------
    large_sizes = [100, 150, 200]
    methods_large = ['locpc', 'ldecc', 'CMB', 'MBbyMB']  # PC excluded

    # Identifiable DAGs
    ID_summary_large, ID_detailed_large = run_experiments(
        random_DAG_identifiable_CDE, methods_large, large_sizes, N_EXP, M, N_SAMPLES, count_non_identifiable=False
    )
    ID_summary_large.to_csv("output_experiments_gaussian/summary_identifiable_large.csv", index=False)
    ID_detailed_large.to_csv("output_experiments_gaussian/final_results_identifiable_large.csv", index=False)

    # Non-identifiable DAGs
    NONID_summary_large, NONID_detailed_large = run_experiments(
        random_DAG_nonidentifiable_CDE, methods_large, large_sizes, N_EXP, M, N_SAMPLES, count_non_identifiable=True
    )
    NONID_summary_large.to_csv("output_experiments_gaussian/summary_nonidentifiable_large.csv", index=False)
    NONID_detailed_large.to_csv("output_experiments_gaussian/final_results_nonidentifiable_large.csv", index=False)