"""Experiment: compare PC vs RestPC on random linear DtDynamicSCM

- 100 iterations (configurable)
- In each iteration:
  - generate a random linear DtDynamicSCM WITHOUT unmeasured confounding
  - generate causally-stationary data using the latest mechanisms with 50 timepoints and 1000 samples
  - keep only the last timepoint (optionally subsample to 100 final samples)
  - run PC and RestPC on that cross-sectional dataset
  - obtain the true CPDAG from the SCM via induced_ft_dag() -> CompletedPartiallyDirectedAcyclicDifferenceGraph.construct_from_dag
  - compute F1 on skeleton (undirected adjacency) between estimated CPDAG and true CPDAG
  - compute F1 on cluster-super-unshielded-colliders (CSUC) sets
- Report mean and variance of scores for PC and RestPC

Notes / assumptions:
- The user asked for 1000 samples then later mentioned 100 final samples; by default we generate 1000 samples and then subsample to 100 final samples to match the request. You can change `SUBSAMPLE_FINAL` to None to keep all samples.
"""

from typing import Set, Tuple, FrozenSet, Iterable
import numpy as np
import pandas as pd
import random

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm, DtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftadmg
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag, FtDirectedAcyclicGraph
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest

# Helper functions
def skeleton_edges_as_undirected_set(graph) -> Set[frozenset]:
    """Return set of undirected unordered pairs for any adjacency in `graph`.
    Graph is expected to implement `get_vertices()` and `get_adjacencies()`.
    """
    verts = list(graph.get_vertices())
    edges = set()
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            u = verts[i]
            v = verts[j]
            if graph.is_adjacent(u, v):
                edges.add(frozenset((u, v)))
    return edges


def f1_from_sets(pred_set: Set, true_set: Set) -> float:
    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def canonical_csuc_set(csucs: Iterable[Tuple[FrozenSet, FrozenSet, FrozenSet]]):
    """Return a canonical set form for CSUC triplets so they are comparable across graphs."""
    # elements are tuples of frozensets already; they are hashable. Return set of tuples.
    return set(csucs)




import random


def generate_ft_dag_with_one_cluster_super_unshielded_collider(
    nb_ts: int,
    p_edge: float,
    seed: int,
    max_delay: int = 2,
    allow_instantaneous: bool = True,
):
    """
    Construct a structurally causally stationary FT-DAG containing a guaranteed
    cluster-super-unshielded collider (CSUC).

    Guaranteed CSUC:
        X = {V0}, Y = {V1}, Z = {V2}

    with stationary collider pattern:
        V0[t] -> V1[t + lag_c]
        V2[t] -> V1[t + lag_c]

    The construction *guarantees* that no alternative active path exists between
    V0 and V2 in the summary graph by isolating V0 and V2 from all other series
    except via their edges into V1.

    Structural causal stationarity:
        if Vi[t] -> Vj[t + lag] exists for one valid t, then it exists for all
        valid t with the same lag.
    """
    if nb_ts < 3:
        raise ValueError("nb_ts must be at least 3.")
    if not (0.0 <= p_edge <= 1.0):
        raise ValueError("p_edge must be between 0 and 1.")
    if max_delay < 1:
        raise ValueError("max_delay must be at least 1.")

    rng = random.Random(seed)

    # Keep graph size manageable while still using max_delay.
    n_slices = min(max(3, max_delay + 1), 8)
    max_lag = min(max_delay, n_slices - 1)

    ft = FtDirectedAcyclicGraph()

    nodes = {
        (i, t): DTimeVar(f"V{i}", t)
        for i in range(nb_ts)
        for t in range(n_slices)
    }

    for node in nodes.values():
        ft.add_vertex(node)

    def add_stationary_edge_pattern(src_i: int, dst_i: int, lag: int) -> None:
        """
        Add stationary pattern Vi[t] -> Vj[t+lag] for all valid t.
        """
        if lag < 0:
            raise ValueError("lag must be non-negative")
        if lag == 0 and not allow_instantaneous:
            return
        if lag == 0 and src_i == dst_i:
            return

        for t in range(n_slices - lag):
            ft.add_directed_edge(nodes[(src_i, t)], nodes[(dst_i, t + lag)])

    # ------------------------------------------------------------------
    # 1) Add self-lag edges for every series: Vi[t] -> Vi[t+1]
    # ------------------------------------------------------------------
    for i in range(nb_ts):
        add_stationary_edge_pattern(i, i, 1)

    # ------------------------------------------------------------------
    # 2) Add the guaranteed stationary collider V0 -> V1 <- V2
    # ------------------------------------------------------------------
    collider_lag = 0 if allow_instantaneous else 1
    add_stationary_edge_pattern(0, 1, collider_lag)
    add_stationary_edge_pattern(2, 1, collider_lag)

    # ------------------------------------------------------------------
    # 3) Add random stationary background structure ONLY among V3, V4, ...
    #    This prevents any alternative path between V0 and V2.
    # ------------------------------------------------------------------
    background_series = list(range(3, nb_ts))
    admissible_lags = range(0 if allow_instantaneous else 1, max_lag + 1)

    for src_i in background_series:
        for dst_i in background_series:
            for lag in admissible_lags:
                # For instantaneous edges, impose an order to preserve acyclicity.
                if lag == 0:
                    if src_i == dst_i:
                        continue
                    if src_i >= dst_i:
                        continue

                if rng.random() < p_edge:
                    add_stationary_edge_pattern(src_i, dst_i, lag)

    # Optional: allow additional incoming edges into Y from background nodes,
    # while still preventing X-Z alternative paths.
    #
    # These do NOT create a V0--...--V2 path as long as V0 and V2 remain
    # isolated from the background.
    for src_i in background_series:
        for lag in admissible_lags:
            if lag == 0 and not allow_instantaneous:
                continue
            if rng.random() < p_edge:
                add_stationary_edge_pattern(src_i, 1, lag)

    csuc = (
        frozenset({"V0"}),
        frozenset({"V1"}),
        frozenset({"V2"}),
    )

    return ft, {csuc}




# Experiment parameters
NUM_ITERS = 10
N_TIMEPOINTS = 10
N_SAMPLES = 500
BURN_IN = 5
# SUBSAMPLE_FINAL = 1000  # If not None, subsample final timepoint to this many samples
SEED_BASE = 42


def run_single_iteration(seed: int):
    rng = np.random.default_rng(seed)

    g, true_csucs = generate_ft_dag_with_one_cluster_super_unshielded_collider(nb_ts=10, p_edge=0.1, seed=seed)

    # create random SCM without unmeasured confounding
    # g = create_random_ft_dag(num_ts=10, p_edge=0.1, allow_instantaneous=True, seed=seed)

    vertices = list(g.get_vertices())
    for vertex_i in vertices:
        for vertex_j in vertices:
            name_i = f"{vertex_i.name}"
            time_i = vertex_i.time
            name_j = f"{vertex_j.name}"
            time_j = vertex_j.time
            if name_i == name_j:
                if time_i == time_j - 1:
                    if (vertex_i, vertex_j) not in g.get_directed_edges():
                        g.add_directed_edge(vertex_i, vertex_j)

    scm = create_random_linear_dt_dynamic_scm_from_ftadmg(
        ftadmg=g,
        causally_stationary=True,
        u_dist=np.random.normal,
        seed=seed,
    )


    # generate causally-stationary data using latest mechanisms
    # DtDynamicSCM.generate_causally_stationary_data_from_latest_mechanisms
    df = scm.generate_causally_stationary_data_from_latest_mechanisms(
        n_timepoints=N_TIMEPOINTS,
        n_samples=N_SAMPLES,
        burn_in=BURN_IN,
        include_latent=False,
        seed=seed,
        reindex_time=True,
    )

    # keep only last timepoint: columns are DTimeVar objects indexed 1..N_TIMEPOINTS
    # we select columns with time index == N_TIMEPOINTS
    last_cols = [c for c in df.columns if isinstance(c, DTimeVar) and c.time == N_TIMEPOINTS]
    df_last = df[last_cols]

    # optionally subsample rows
    # if subsample_final is not None and subsample_final < len(df_last):
    #     df_last = df_last.sample(n=subsample_final, random_state=seed).reset_index(drop=True)
    new_names = [f"{col.name}" for col in df_last.columns]
    df_last.columns = new_names

    print("data generattion complete, running PC and RestPC...")


    # run PC and RestPC
    pc = PC(sparsity=0.05, ci_test=LinearRegressionCoefficientTTest)
    pc.run(df_last)
    restpc = RestPC(sparsity=0.05, ci_test=LinearRegressionCoefficientTTest)
    restpc.run(df_last)

    # True CPDAG from SCM
    true_dag = scm.induced_ft_dag()
    # TODO CPDAG of SCG not of DAG
    true_scg = true_dag.get_summary_causal_graph()
    true_cpdag = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    true_cpdag.construct_from_dag(true_scg)
    true_scg.draw_graph()

    # compute skeleton F1
    true_edges = skeleton_edges_as_undirected_set(true_cpdag)
    pc_edges = skeleton_edges_as_undirected_set(pc.g_hat)
    restpc_edges = skeleton_edges_as_undirected_set(restpc.g_hat)

    f1_pc = f1_from_sets(pc_edges, true_edges)
    f1_restpc = f1_from_sets(restpc_edges, true_edges)

    # compute CSUC F1 (set equality style)
    # true_csucs = canonical_csuc_set(true_scg.get_all_cluster_super_unshielded_colliders())
    pc_csucs = canonical_csuc_set(pc.g_hat.get_all_cluster_super_unshielded_colliders())
    restpc_csucs = canonical_csuc_set(restpc.g_hat.get_all_cluster_super_unshielded_colliders())

    print(f"True CSUCs: {len(true_csucs)}")
    print(f"PC CSUCs: {len(pc_csucs)}, RestPC CSUCs: {len(restpc_csucs)}")

    csuc_f1_pc = f1_from_sets(pc_csucs, true_csucs)
    csuc_f1_restpc = f1_from_sets(restpc_csucs, true_csucs)

    return {
        'f1_pc': f1_pc,
        'f1_restpc': f1_restpc,
        'csuc_f1_pc': csuc_f1_pc,
        'csuc_f1_restpc': csuc_f1_restpc,
    }


def run_experiment(num_iters: int = NUM_ITERS):
    results = []
    for i in range(num_iters):
        print(f"Running iteration {i+1}/{num_iters} with seed {SEED_BASE + i}...")
        seed = SEED_BASE + i
        res = run_single_iteration(seed)
        results.append(res)
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{num_iters} iterations")
    df_res = pd.DataFrame(results)

    summary = {}
    for col in df_res.columns:
        summary[col + '_mean'] = float(df_res[col].mean())
        summary[col + '_var'] = float(df_res[col].var(ddof=0))

    print("\nExperiment summary:")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")

    return df_res, summary


if __name__ == '__main__':
    # Run the full experiment
    df_results, summary = run_experiment(NUM_ITERS)
    # Save results
    print(df_results)
    # df_results.to_csv('pc_vs_restpc_experiment_results.csv', index=False)
    # print('\nSaved detailed results to pc_vs_restpc_experiment_results.csv')
