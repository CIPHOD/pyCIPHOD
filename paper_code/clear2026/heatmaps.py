# =========================================================
# PC vs LocPC – Heatmaps and Animation Script
# =========================================================

# =========================================================
# Imports and General Setup
# =========================================================
import sys
import random
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =========================================================
# Reproducibility
# =========================================================
SEED = 2026
random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# Paths
# =========================================================
root = Path(__file__).resolve().parent
sys.path.extend([
    str(root),
    str(root.parents[1] / "PyCIPHOD")
])


# =========================================================
# External Imports
# =========================================================
from paper_code.clear2026.dags_generator import (
    random_DAG_identifiable_CDE,
    random_DAG_nonidentifiable_CDE
)

from PyCIPHOD.causal_discovery.pc.pc import PC
from PyCIPHOD.causal_discovery.local.locpc import LocPC


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
# Linear SCM Simulation (coef + data version)
# =========================================================
def simulate_linearSCM_from_dag(dag, n_obs=1,
                                coef_range=(-1,1),
                                sigma_range=(0.5,1)):
    nodes = list(nx.topological_sort(dag))
    p = len(nodes)

    adj = nx.to_numpy_array(dag, nodelist=nodes)
    coef = np.random.uniform(coef_range[0], coef_range[1], size=(p,p)) * adj.T
    sigma_vec = np.random.uniform(*sigma_range, size=p)

    coef_inv = np.linalg.inv(np.eye(p) - coef)
    data = np.random.normal(0, sigma_vec, size=(n_obs, p)) @ coef_inv.T

    return (
        pd.DataFrame(coef, index=nodes, columns=nodes),
        pd.DataFrame(data, columns=nodes)
    )


# =========================================================
# Neighborhood computation
# =========================================================
def compute_neighborhood(graph: nx.DiGraph, target: str):
    """
    Compute hop-distance from target in undirected graph.
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


def neighborhood(g: nx.DiGraph, target: str):
    """
    Alternative neighborhood function (same logic).
    """
    if target not in g:
        raise ValueError(f"Target node '{target}' must be in the graph")

    undirected = g.to_undirected()
    visited = {target}
    neighborhood_dict = {target: 0}
    queue = deque([(target, 0)])

    while queue:
        node, h = queue.popleft()
        for neighbor in undirected.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                neighborhood_dict[neighbor] = h + 1
                queue.append((neighbor, h + 1))

    return neighborhood_dict


# =========================================================
# Heatmap Plotting
# =========================================================
def plot_pc_vs_locpc_heatmaps(pc_obj, locpc_list, target, graph):

    neighborhood = compute_neighborhood(graph, target)
    nodes = list(graph.nodes)

    sorted_nodes = sorted(
        nodes,
        key=lambda x: neighborhood.get(x, max(neighborhood.values()) + 1)
    )

    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    grids = []
    objects = [pc_obj] + locpc_list

    def build_triangular_grid(test_list):
        grid = np.zeros((n, n))
        for i, j, _ in test_list:
            xi = node_to_idx[i]
            yi = node_to_idx[j]
            a, b = sorted([xi, yi])
            grid[a, b] += 1
        return grid

    for obj in objects:
        grids.append(build_triangular_grid(obj.performed_tests))

    global_max = max(grid.max() for grid in grids)
    vmax = global_max if global_max > 0 else 1

    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ["PC", "LocPC hop=1", "LocPC hop=2", "LocPC hop=3"]

    for ax, grid, title, obj in zip(axes.flat, grids, titles, objects):

        masked_grid = np.ma.array(grid, mask=mask)
        ax.imshow(masked_grid, cmap="Reds", vmin=0, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(
            0.02, 0.98,
            f"Total # of tests: {obj.nb_ci_tests}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

        if hasattr(obj, "_visited") and obj._visited:
            for node in obj._visited:
                if node in node_to_idx:
                    idx = node_to_idx[node]
                    ax.add_patch(
                        plt.Rectangle(
                            (idx - 0.5, idx - 0.5),
                            1, 1,
                            facecolor="blue",
                            alpha=0.6
                        )
                    )

    plt.tight_layout()
    plt.show()


# =========================================================
# Animation
# =========================================================
def animate_pc_vs_locpc_hops(pc_obj, locpc_objs, target, g,
                             interval=100, decay=0.8):

    neighborhood_dict = neighborhood(g, target)
    all_nodes = list(g.nodes)

    sorted_nodes = sorted(
        all_nodes,
        key=lambda x: neighborhood_dict.get(
            x, max(neighborhood_dict.values(), default=0)+1
        )
    )

    n = len(sorted_nodes)
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}

    fig, axes = plt.subplots(1, 4, figsize=(18,5))
    ims = []
    grids = [np.zeros((n,n)) for _ in range(4)]

    names = ["PC"] + [f"LocPC hop={h}" for h in [1,2,3]]

    for ax, name in zip(axes, names):
        im = ax.imshow(np.zeros((n,n)), cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        ax.invert_yaxis()
        ims.append(im)

    pc_tests = [(node_to_idx[i], node_to_idx[j])
                for (i,j,_) in pc_obj.performed_tests]

    locpc_tests_list = [
        [(node_to_idx[i], node_to_idx[j])
         for (i,j,_) in loc_obj.performed_tests]
        for loc_obj in locpc_objs
    ]

    n_frames = max(len(pc_tests),
                   *(len(tests) for tests in locpc_tests_list))

    def update(frame):

        for k in range(4):
            grids[k] *= decay

        if frame < len(pc_tests):
            x, y = pc_tests[frame]
            grids[0][y, x] = 1

        for k, loc_tests in enumerate(locpc_tests_list):
            if frame < len(loc_tests):
                x, y = loc_tests[frame]
                grids[k+1][y, x] = 1

        for im_idx, im in enumerate(ims):
            im.set_data(grids[im_idx])

        return ims

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=interval,
        blit=True,
        repeat=False
    )

    plt.show()


# =========================================================
# Main Execution
# =========================================================
def main():

    # ---------------------------------
    # Generate identifiable DAG
    # ---------------------------------
    n = 40
    m = 2
    g, y, x = random_DAG_identifiable_CDE(n, m/(n+1))

    # ---------------------------------
    # Simulate data
    # ---------------------------------
    data = simulate_linear_scm_from_dag(g, n_obs=10000)

    # ---------------------------------
    # Run PC
    # ---------------------------------
    pc = PC(data=data)
    pc.run()

    # ---------------------------------
    # Run LocPC (hop 1,2,3)
    # ---------------------------------
    locpc_list = []
    for hop in [1,2,3]:
        locpc_obj = LocPC(data)
        locpc_obj.run(target=y, hop=hop)
        locpc_list.append(locpc_obj)

    # ---------------------------------
    # Static heatmaps
    # ---------------------------------
    plot_pc_vs_locpc_heatmaps(
        pc,
        locpc_list,
        target=y,
        graph=g
    )

    # ---------------------------------
    # Animation
    # ---------------------------------
    animate_pc_vs_locpc_hops(
        pc,
        locpc_list,
        target=y,
        g=g,
        interval=1
    )


# =========================================================
if __name__ == "__main__":
    main()