# %% Full pipeline — generate SCM, FT-DAG, SCG, data, and orient ALL instantaneous edges

import numpy as np
import matplotlib.pyplot as plt
import itertools

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import (
    OrientInstantaneousEdge,
    sOrientability,
)
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.stat_tests.independence_tests import FisherZTest
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph

seed = np.random.poisson(1000)
print(seed)

p, p_edge, gamma_max, alpha = 3, 0.3, 1, 0.05

# %% SCM
scm = create_random_linear_dt_dynamic_scm(
    num_ts=p,
    p_edge=p_edge,
    causally_stationary=True,
    max_delay=gamma_max,
    allow_instantaneous=True,
    allow_unmeasured_confounding=False,
    weight_low=-1 / (p * p_edge * gamma_max),
    weight_high=1 / (p * p_edge * gamma_max),
    intercept_low=0.0,
    intercept_high=0.0,
    u_dist=lambda: np.random.normal(0, 1),
    seed=seed,
)

# %% FT-DAG -> SCG
ft_dag = scm.induced_ft_dag()
scg = ft_dag.get_summary_causal_graph()
gamma_max = scg.get_lag_max()

print("FT-DAG vertices:", sorted(ft_dag.get_vertices()))
print("FT-DAG directed edges:", ft_dag.get_directed_edges())
ft_dag.draw_graph()

print("SCG vertices:", sorted(scg.get_vertices()))
print("SCG directed edges:", scg.get_directed_edges())
print("SCG lag_max:", gamma_max)
scg.draw_graph()

# %% Data
data_full = scm.generate_causally_stationary_time_series_from_latest_mechanisms(
    n_timepoints=10000,
    burn_in=200,
    include_latent=False,
    time_series_format="ts_lagged",
    seed=seed,
)

data = data_full[[c for c in data_full.columns if 0 <= -c.time <= gamma_max]].copy()

print(data.shape)
print(sorted(data.columns, key=lambda c: (c.name, c.time)))
data.head()

# %% Plot lag 0
lag0_cols = sorted((c for c in data.columns if c.time == 0), key=lambda c: c.name)

fig, axes = plt.subplots(
    len(lag0_cols), 1, figsize=(10, 2.2 * len(lag0_cols)), sharex=True
)
axes = np.atleast_1d(axes)

for ax, col in zip(axes, lag0_cols):
    ax.plot(data[col].to_numpy()[:1000], lw=0.8)
    ax.set_ylabel(col.name)
    ax.axhline(0, color="grey", lw=0.5, ls="--")

axes[-1].set_xlabel("t")
fig.suptitle("Séries temporelles (lag 0) — premiers points")
fig.tight_layout()
plt.show()

# %% Orientation des arêtes instantanées

ft_dag_edges = set(ft_dag.get_directed_edges())

instantaneous_pairs = sorted(
    {
        frozenset((u, v))
        for u, v in scg.get_directed_edges()
        if u != v
    },
    key=lambda x: tuple(sorted(x)),
)


def true_instantaneous_edge(u, v):
    u2v = (DTimeVar(u, 0), DTimeVar(v, 0)) in ft_dag_edges
    v2u = (DTimeVar(v, 0), DTimeVar(u, 0)) in ft_dag_edges
    if u2v and not v2u:
        return f"{u}t -> {v}t"
    if v2u and not u2v:
        return f"{u}t <- {v}t"
    if u2v:
        return f"{u}t <-> {v}t"
    return "not adjacent"


results = []

for pair in instantaneous_pairs:
    u, v = sorted(pair)

    orienter = OrientInstantaneousEdge(
        data=data,
        A=DTimeVar(u, 0),
        B=DTimeVar(v, 0),
        scg=scg,
        alpha=alpha,
        ci_test=None,
        oracle = True,
        true_ftdag = ft_dag
    )

    predicted,_ = orienter.orient()
    truth = true_instantaneous_edge(u, v)
    nb_tests = orienter.nb_ci_tests

    s_orientable = (
        sOrientability(scg, DTimeVar(u, 0), DTimeVar(v, 0)).is_s_orientable()
        if u in scg.get_adjacencies(v)
        else None
    )

    expected = (
        "not adjacent"
        if truth == "not adjacent"
        else truth if s_orientable is not False else "non s-orientable"
    )

    results.append(
        (
            u,
            v,
            truth,
            s_orientable,
            expected,
            predicted,
            nb_tests,
            "OK" if predicted == expected else "MISMATCH",
        )
    )

# %% Résultats

header = (
    f"{'A':>4} {'B':>4} | {'FT-DAG (vérité)':<15} | {'s-orient?':<9} | "
    f"{'attendu':<15} | {'orient() (prédit)':<15} | {'#tests':>6} | résultat"
)

print(header)
print("-" * len(header))

for u, v, truth, s_orientable, expected, predicted, nb_tests, match in results:
    print(
        f"{u:>4} {v:>4} | {truth:<15} | "
        f"{('-' if s_orientable is None else s_orientable)!s:<9} | "
        f"{expected:<15} | {predicted:<15} | {nb_tests:>6} | {match}"
    )

n_ok = sum(r[-1] == "OK" for r in results)
total_tests = sum(r[6] for r in results)

print(
    f"\n{n_ok}/{len(results)} arêtes correctement traitées "
    "(vs. attendu, s-orientabilité incluse)."
)
print(f"Nombre total de tests d'indépendance conditionnelle : {total_tests}")



#%%


def bk_from_scg(scg: SummaryCausalGraph, gamma_max: int) -> BackgroundKnowledge:
    nodes = list(scg.get_vertices())
    bk = BackgroundKnowledge()

    adj = {x: set(scg.get_adjacencies(x)) for x in nodes}
    parents = {x: set(scg.get_parents(x)) for x in nodes}
    children = {x: set(scg.get_children(x)) for x in nodes}

    lags = range(gamma_max + 1)
    all_kl = list(itertools.product(lags, lags))
    neq_kl = [(k, l) for k, l in all_kl if k != l]

    for x, y in itertools.combinations(nodes, 2):
        if y not in adj[x]:
            for k, l in all_kl:
                bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(y, -l))
                bk.add_forbidden_edge(DTimeVar(y, -k), DTimeVar(x, -l))
        else:
            if x in parents[y] and x not in children[y]:
                bk.add_mandatory_orientation(DTimeVar(x, 0), DTimeVar(y, 0))
            if y in parents[x] and y not in children[x]:
                bk.add_mandatory_orientation(DTimeVar(y, 0), DTimeVar(x, 0))

    for x in nodes:
        if x not in adj[x]:
            for k, l in neq_kl:
                bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(x, -l))

    return bk


tpc = TsPC(0.05, ci_test=FisherZTest, oracle = True, true_ftdag = ft_dag)
tpc.run(data)
tpc.g_hat.draw_graph(time_graph = True)
print(f"w/o BK : {tpc.nb_ci_tests}")
# %%
bk = bk_from_scg(scg,gamma_max)
tpc2 = TsPC(0.05, ci_test=FisherZTest, background_knowledge = bk, oracle = True, true_ftdag = ft_dag)
tpc2.run(data)
tpc2.g_hat.draw_graph(time_graph = True)
print(f"w/ BK : {tpc2.nb_ci_tests}")
# %%



# %% Simple experiment — generate SCM, FT-DAG, SCG, data, and orient ONE prespecified edge

import numpy as np

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.orientation_using_scg import (
    OrientInstantaneousEdge,
    sOrientability,
)
from pyciphod.utils.graphs.separation import d_separated
# %% Hyperparamètres de simulation

SEED = np.random.poisson(1000)
print(SEED)

SEED = 1050

NUM_TS = 3              # nombre de séries temporelles (variables)
P_EDGE = 0.5             # probabilité d'arête
GAMMA_MAX = 1            # lag maximal
ALPHA = 0.05             # seuil des tests CI (non utilisé ici en mode oracle)
N_TIMEPOINTS = 10_000    # longueur des séries générées
BURN_IN = 200

ORACLE = True            # True = d-séparation exacte sur le FT-DAG, False = tests CI sur les données

# Arête instantanée à orienter (noms de variables, ex: "V0", "V1", ...)
EDGE_U = "V1"
EDGE_V = "V2"

# %% Génération du SCM

scm = create_random_linear_dt_dynamic_scm(
    num_ts=NUM_TS,
    p_edge=P_EDGE,
    causally_stationary=True,
    max_delay=GAMMA_MAX,
    allow_instantaneous=True,
    allow_unmeasured_confounding=False,
    weight_low=-1 / (NUM_TS * P_EDGE * GAMMA_MAX),
    weight_high=1 / (NUM_TS * P_EDGE * GAMMA_MAX),
    intercept_low=0.0,
    intercept_high=0.0,
    u_dist=lambda: np.random.normal(0, 1),
    seed=SEED,
)

# %% FT-DAG et SCG induits

ft_dag = scm.induced_ft_dag()
ft_dag.draw_graph()
scg = ft_dag.get_summary_causal_graph()
scg.draw_graph()
gamma_max = scg.get_lag_max()

print("FT-DAG vertices:", sorted(ft_dag.get_vertices()))
print("FT-DAG directed edges:", ft_dag.get_directed_edges())

print("SCG vertices:", sorted(scg.get_vertices()))
print("SCG directed edges:", scg.get_directed_edges())
print("SCG lag_max:", gamma_max)

# %% Génération des données (utile seulement si ORACLE=False)

data = None
if not ORACLE:
    data_full = scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=N_TIMEPOINTS,
        burn_in=BURN_IN,
        include_latent=False,
        time_series_format="ts_lagged",
        seed=SEED,
    )
    data = data_full[[c for c in data_full.columns if 0 <= -c.time <= gamma_max]].copy()
    print(data.shape)

# %% Orientation de l'arête prespécifiée EDGE_U <-> EDGE_V

A = DTimeVar(EDGE_U, 0)
B = DTimeVar(EDGE_V, 0)

orienter = OrientInstantaneousEdge(
    data=data,
    A=A,
    B=B,
    scg=scg,
    alpha=ALPHA,
    ci_test=None,
    oracle=ORACLE,
    true_ftdag=ft_dag,
)

predicted, case = orienter.orient()

s_orientable = (
    sOrientability(scg, A, B).is_s_orientable()
    if EDGE_U in scg.get_adjacencies(EDGE_V)
    else None
)

def ground_truth(ft_dag, A, B):
    if (A,B) in ft_dag.get_directed_edges():
        print(f"True edge : {A.get_name()}t -> {B.get_name()}t")
    elif (B,A) in ft_dag.get_directed_edges():
        print(f"True edge : {A.get_name()}t <- {B.get_name()}t")
    else:
        print("True edge : Not adjacent")
    

print(f"\nArête testée : {EDGE_U} -- {EDGE_V}")
print(f"Orientation prédite : {predicted} (case {case})")
ground_truth(ft_dag, A, B)
print(f"s-orientable : {s_orientable}")
print(f"Nombre de tests CI effectués : {orienter.nb_ci_tests}")

# %%
