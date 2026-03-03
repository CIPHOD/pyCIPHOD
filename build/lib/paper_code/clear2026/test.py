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

import random

SEED = 2026
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
from baselines.Gupta_codes.ldecc import LDECCAlgorithm
from baselines.Gupta_codes.pc_alg import PCAlgorithm
from baselines.pyCausalFS.LSL.MBs.CMB.CMB import CMB
from baselines.pyCausalFS.LSL.MBs.MBbyMB import MBbyMB
from PyCIPHOD.causal_discovery.pc.pc import PC
from PyCIPHOD.causal_discovery.local.locpc import LocPC

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

# %%
n = 20
m = 2
g, y, x = random_DAG_nonidentifiable_CDE(n, m/(n+1))

_, data = simulate_linearSCM_from_dag(g, 5000)
# %%
locpc = LocPC(data)
# %%
locpc.run(y, 2)
locpc.leg.draw_graph({y})
# %%
locpc = LocPC(data)
locpc.run_locPC_CDE(x,y)
# %%

