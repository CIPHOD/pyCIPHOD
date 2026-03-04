# pc_locpc_static_heatmaps.py
# ============================

import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

from paper_code.clear2026.dags_generator import random_DAG_identifiable_CDE
from PyCIPHOD.causal_discovery.pc.pc import PC
from PyCIPHOD.causal_discovery.local.locpc import LocPC

import sys
from pathlib import Path

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
    Plot PC and LocPC (hop=1,2,3) cumulative heatmaps in a 2x2 grid.
    """
    # Node ordering based on hop-distance from target
    neighborhood = compute_neighborhood(graph, target)
    nodes = list(graph.nodes)

    sorted_nodes = sorted(
        nodes,
        key=lambda x: neighborhood.get(x, max(neighborhood.values()) + 1)
    )

    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    # Build cumulative grids
    grids = []

    # PC
    grid_pc = np.zeros((n, n))
    for i, j, _ in pc_obj.performed_tests:
        x, y = node_to_idx[i], node_to_idx[j]
        grid_pc[y, x] += 1
    grids.append(grid_pc)

    # LocPC for hops
    for locpc_obj in locpc_list:
        grid_loc = np.zeros((n, n))
        for i, j, _ in locpc_obj.performed_tests:
            x, y = node_to_idx[i], node_to_idx[j]
            grid_loc[y, x] += 1
        grids.append(grid_loc)

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    titles = ["PC", "LocPC hop=1", "LocPC hop=2", "LocPC hop=3"]

    for ax, grid, title in zip(axes.flat, grids, titles):
        ax.imshow(grid, cmap="Reds", vmin=0)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# =========================================================
# Main execution
# =========================================================
if __name__ == "__main__":

    # Generate identifiable DAG
    n = 25
    m = 4
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