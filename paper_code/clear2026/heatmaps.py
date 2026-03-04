#%%
# =========================================
# Imports and General Setup
# =========================================
import os
import sys
from pathlib import Path

import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

import random

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

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

from PyCIPHOD.causal_discovery.pc.pc import PC
from PyCIPHOD.causal_discovery.local.locpc import LocPC

root = Path(__file__).resolve().parent
sys.path.extend([
    str(root),
    str(root.parents[1] / "PyCIPHOD")
])

# =========================================================
# Reproducibility
# =========================================================
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# Linear SCM Simulation
# =========================================================
def simulate_linear_scm_from_dag(dag, n_obs=10000,
                                 coef_range=(-1, 1),
                                 sigma_range=(0.5, 1.0)):
    """
    Simulate data from a linear SCM defined by a DAG.
    """
    nodes = list(nx.topological_sort(dag))
    p = len(nodes)

    adj = nx.to_numpy_array(dag, nodelist=nodes)
    coef = np.random.uniform(*coef_range, size=(p, p)) * adj.T
    sigma_vec = np.random.uniform(*sigma_range, size=p)

    coef_inv = np.linalg.inv(np.eye(p) - coef)
    data = np.random.normal(0, sigma_vec, size=(n_obs, p)) @ coef_inv.T

    return pd.DataFrame(data, columns=nodes)


# =========================================================
# Neighborhood computation
# =========================================================
def compute_neighborhood(graph: nx.DiGraph, target: str):
    """
    Compute hop-distance from target in undirected version of graph.
    Returns dict {node: hop_distance}.
    """
    undirected = graph.to_undirected()

    visited = {target}
    distances = {target: 0}
    queue = deque([(target, 0)])

    while queue:
        node, dist = queue.popleft()
        for neigh in undirected.neighbors(node):
            if neigh not in visited:
                visited.add(neigh)
                distances[neigh] = dist + 1
                queue.append((neigh, dist + 1))

    return distances


# =========================================================
# Heatmap Plotting
# =========================================================
def plot_pc_vs_locpc_heatmaps(pc_obj, locpc_list, target, graph):
    """
    Plot PC and LocPC (hop=1,2,3) cumulative TRIANGULAR heatmaps
    in a 2x2 grid using a shared global color scale.

    Each independence test (X,Y,S) is counted once
    and stored in a canonical upper-triangular position.

    - Displays total number of CI tests in upper-left
    - For LocPC, highlights nodes in _visited on the diagonal in blue
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # -----------------------------------------------------
    # Node ordering based on hop-distance from target
    # -----------------------------------------------------
    neighborhood = compute_neighborhood(graph, target)
    nodes = list(graph.nodes)

    sorted_nodes = sorted(
        nodes,
        key=lambda x: neighborhood.get(x, max(neighborhood.values()) + 1)
    )

    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    # -----------------------------------------------------
    # Build triangular grids
    # -----------------------------------------------------
    grids = []
    objects = [pc_obj] + locpc_list

    def build_triangular_grid(test_list):
        grid = np.zeros((n, n))
        for i, j, _ in test_list:
            xi = node_to_idx[i]
            yi = node_to_idx[j]
            a, b = sorted([xi, yi])  # upper triangle
            grid[a, b] += 1
        return grid

    for obj in objects:
        grids.append(build_triangular_grid(obj.performed_tests))

    # -----------------------------------------------------
    # Global color scale
    # -----------------------------------------------------
    global_max = max(grid.max() for grid in grids)
    vmin = 0
    vmax = global_max if global_max > 0 else 1

    # Mask lower triangle
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    # -----------------------------------------------------
    # Plot 2x2 grid
    # -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    titles = ["PC", "LocPC hop=1", "LocPC hop=2", "LocPC hop=3"]

    for ax, grid, title, obj in zip(axes.flat, grids, titles, objects):
        masked_grid = np.ma.array(grid, mask=mask)
        ax.imshow(masked_grid, cmap="Reds", vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # -------------------------------------------------
        # Display total number of CI tests
        # -------------------------------------------------
        ax.text(
            0.02, 0.98,
            f"Total # of tests: {obj.nb_ci_tests}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

        # -------------------------------------------------
        # Highlight diagonal for LocPC only
        # -------------------------------------------------
        if hasattr(obj, "_visited") and obj._visited:
            for node in obj._visited:
                if node in node_to_idx:
                    idx = node_to_idx[node]
                    # small blue square at (idx, idx)
                    ax.add_patch(
                        plt.Rectangle(
                            (idx - 0.5, idx - 0.5), 1, 1,
                            facecolor="blue",
                            edgecolor=None,
                            alpha=0.6
                        )
                    )

    plt.tight_layout()
    plt.show()
    
# =========================================================
# Main execution
# =========================================================
if __name__ == "__main__":

    # Generate identifiable DAG
    n = 100
    m = 3
    g, y, x = random_DAG_identifiable_CDE(n, m / (n + 1))

    # Simulate data
    data = simulate_linear_scm_from_dag(g, n_obs=10000)

    # Run PC
    pc = PC(data=data)
    pc.run()

    # Run LocPC for hop = 1, 2, 3
    locpc_hop1 = LocPC(data)
    locpc_hop1.run(target=y, hop=1)

    locpc_hop2 = LocPC(data)
    locpc_hop2.run(target=y, hop=2)

    locpc_hop3 = LocPC(data)
    locpc_hop3.run(target=y, hop=3)

    # Plot heatmaps
    plot_pc_vs_locpc_heatmaps(
        pc,
        [locpc_hop1, locpc_hop2, locpc_hop3],
        target=y,
        graph=g
    )