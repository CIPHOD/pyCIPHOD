
#%%
import itertools
import warnings

import numpy as np

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import (
    SCGOrienter, OracleCI, NOT_ADJACENT, NOT_S_ORIENTABLE
)


# --- Random SCM -> FT-DAG -> SCG
seed = np.random.poisson(1000)
p, p_edge, gamma_max = 5, 0.5, 1
scm = create_random_linear_dt_dynamic_scm(
    num_ts=p, p_edge=p_edge, causally_stationary=True,
    max_delay=gamma_max, allow_instantaneous=True,
    allow_unmeasured_confounding=False,
    weight_low=-1 / (p * p_edge * gamma_max),
    weight_high=1 / (p * p_edge * gamma_max),
    intercept_low=0.0, intercept_high=0.0,
    u_dist=lambda: np.random.normal(0, 1), seed=seed,
)
ft_dag = scm[0].induced_ft_dag()
scg = ft_dag.get_summary_causal_graph()
gamma_max = scg.get_lag_max()
names = sorted({n for e in scg.get_directed_edges() for n in e})

orienter = SCGOrienter(scg, OracleCI(ft_dag, gamma_max))

#%% --- Check 1: adjacency of lagged pairs (Y_{t-l} vs X_t) against the FT-DAG
ft_dag_edges = set(ft_dag.get_directed_edges())
n_bad = 0
for x, y in itertools.product(names, names):
    for lag in range(1, gamma_max + 1):
        true_adj = any(a.get_name() == y and b.get_name() == x and b.get_time() - a.get_time() == lag
                       for a, b in ft_dag_edges)
        found_adj = orienter.find_sepset(DTimeVar(x, 0), DTimeVar(y, -lag)) is None
        if true_adj != found_adj:
            n_bad += 1
            print(f"adjacency mismatch: {x}_t vs {y}_(t-{lag}): FT-DAG {true_adj}, ours {found_adj}")
print(f"lagged adjacency mismatches: {n_bad}\n")

#%% --- Check 2: orientation of every instantaneous pair
def true_orientation(u, v):
    if (DTimeVar(u, 0), DTimeVar(v, 0)) in ft_dag_edges:
        return "->"
    if (DTimeVar(v, 0), DTimeVar(u, 0)) in ft_dag_edges:
        return "<-"
    return NOT_ADJACENT


print(f"seed = {seed}")
n_ok = n_wrong = n_skip = 0
for u, v in itertools.combinations(names, 2):
    expected = true_orientation(u, v)
    predicted = orienter.orient_edge(DTimeVar(u, 0), DTimeVar(v, 0))
    if predicted == NOT_S_ORIENTABLE:
        n_skip += 1
        status = "skip"
    elif predicted == expected:
        n_ok += 1
        status = "OK"
    else:
        n_wrong += 1
        status = "WRONG"
    print(f"{u} -- {v} | true: {expected} | ours: {predicted} | {status}")

print(f"\nOK: {n_ok} | WRONG: {n_wrong} | not s-orientable: {n_skip}")