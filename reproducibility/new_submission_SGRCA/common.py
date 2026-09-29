from __future__ import annotations

import contextlib
import io
import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from pyciphod.utils.graphs.temporal_graphs import FtDirectedAcyclicGraph
from pyciphod.utils.time_series.data_format import DTimeVar

from pyciphod.root_cause_analysis.graph_based import SGRCA
from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc


# ============================================================================
# Algorithm adapters
# ============================================================================
# RCD hyperparameters. Note that RCD's `gamma` is its variable-chunk size; it is
# unrelated to the temporal max lag above.
RCD_BINS = 5
RCD_CHUNK_SIZE = 5
RCD_LOCALIZED = False
# Suppress the very verbose prints in the provided MICROCause implementation.
SUPPRESS_BASELINE_OUTPUT = True

def run_sgrca(
    SGRCA,
    *,
    scg,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
) -> List[str]:
    """Return SGRCA's non-lineage root-cause predictions."""
    sgrca = SGRCA(
        g=scg,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=algorithm_max_lag,
        sig_threshold=sig_threshold,
    )

    root_causes = sgrca.run(data)

    prediction: Set[str] = set()
    for component_result in root_causes.values():
        # Do not give SGRCA the planted anomaly category.
        prediction.update(component_result.get("effect_defying", []))
        prediction.update(component_result.get("noise_defying", []))

    return list(prediction)


def run_ftdagrca(
    FTDAGRCA,
    *,
    ftdag,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
) -> List[str]:
    """
    Return FT-DAG-RCA's non-lineage root-cause predictions.
    """

    ftdagrca = FTDAGRCA(
        g=ftdag,
        anomalous_vertices=list(
            anomalous_vertices
        ),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=algorithm_max_lag,
        sig_threshold=sig_threshold,
    )

    root_causes = ftdagrca.run(
        data
    )

    prediction: Set[str] = set()

    for component_result in root_causes.values():

        # Exactly the same evaluation output as SGRCA:
        # lineage-defying roots are not included because the benchmark
        # evaluates the subsequent non-lineage localization stage.
        prediction.update(
            component_result.get(
                "effect_defying",
                [],
            )
        )

        prediction.update(
            component_result.get(
                "noise_defying",
                [],
            )
        )

    return list(prediction)

def run_easyrca(
    EasyRCA,
    *,
    scg,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
) -> List[str]:
    """
    Return EasyRCA's non-lineage root-cause predictions.

    EasyRCA searches lineage-, time-, and effect-defying causes,
    but does not search for noise-defying causes.
    """

    easyrca = EasyRCA(
        g=scg,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=(anomaly_start),
        anomaly_length=(anomaly_size),
        max_delay=(algorithm_max_lag),
        sig_threshold=(sig_threshold),
    )

    root_causes = easyrca.run(data)

    prediction: Set[str] = set()

    for component_result in (root_causes.values()):

        # Same stage-specific protocol as SGRCA:
        # the lineage root is assumed to have already
        # been handled.
        prediction.update(component_result.get( "effect_defying",[],))

    return list(prediction)


def run_sgrca_star(
    sgrca_star_cls,
    *,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
    discovery_sparsity: float = 0.05,
    max_sepset_size=None,
) -> List[str]:
    """
    Run SGRCA*.

    The graph is learned internally from the normal regime using TsPC.

    Return the non-lineage data-driven root-cause predictions.
    """

    sgrca_star = sgrca_star_cls(
        data=data,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=algorithm_max_lag,
        sig_threshold=sig_threshold,
        discovery_sparsity=discovery_sparsity,
        max_sepset_size=max_sepset_size,
    )

    root_causes = sgrca_star.run(data)

    prediction: Set[str] = set()

    for component_result in root_causes.values():
        prediction.update(
            component_result.get(
                "effect_defying",
                [],
            )
        )

        prediction.update(
            component_result.get(
                "noise_defying",
                [],
            )
        )

    return list(prediction)


def run_ftdagrca_star(
    ftdagrca_star_cls,
    *,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
    discovery_sparsity: float = 0.05,
    max_sepset_size=None,
) -> List[str]:
    """
    Run FT-DAG-RCA*.

    The FT-CPDAG is learned internally from the normal regime using TsPC.

    Variables whose complete parent sets are identified are tested as in
    FT-DAG-RCA. Variables whose parent sets remain unoriented are returned
    as potential root causes.
    """

    ftdagrca_star = ftdagrca_star_cls(
        data=data,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=algorithm_max_lag,
        sig_threshold=sig_threshold,
        discovery_sparsity=discovery_sparsity,
        max_sepset_size=max_sepset_size,
    )

    root_causes = ftdagrca_star.run(data)

    prediction: Set[str] = set()

    for component_result in root_causes.values():
        prediction.update(
            component_result.get(
                "effect_defying",
                [],
            )
        )

        prediction.update(
            component_result.get(
                "noise_defying",
                [],
            )
        )

        prediction.update(
            component_result.get(
                "potential_root_causes",
                [],
            )
        )

    return list(prediction)


def run_easyrca_star(
    easyrca_star_cls,
    *,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
    discovery_sparsity: float = 0.05,
    max_sepset_size=None,
) -> List[str]:
    """
    Run EasyRCA*.

    The graph is learned internally from the normal regime using TsPC.

    EasyRCA does not search for noise-defying root causes, so only its
    effect-defying predictions are returned.
    """

    easyrca_star = easyrca_star_cls(
        data=data,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=algorithm_max_lag,
        sig_threshold=sig_threshold,
        discovery_sparsity=discovery_sparsity,
        max_sepset_size=max_sepset_size,
    )

    root_causes = easyrca_star.run(data)

    prediction: Set[str] = set()

    for component_result in root_causes.values():
        prediction.update(
            component_result.get(
                "effect_defying",
                [],
            )
        )

    return list(prediction)

def run_rcd(
    rcd_top_k_rc,
    *,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    n_true_root_causes: int,
    seed: int,
) -> List[str]:
    """Run RCD and return its top-k root-cause variables."""
    normal_df = data.loc[
        data["time"] < anomaly_start,
        list(series_names),
    ].copy()

    anomalous_df = data.loc[
        (data["time"] >= anomaly_start)
        & (data["time"] < anomaly_start + anomaly_size),
        list(series_names),
    ].copy()

    # RCD uses numpy's global RNG in create_chunks().
    np.random.seed(seed)

    result = rcd_top_k_rc(
        normal_df,
        anomalous_df,
        k=n_true_root_causes,
        bins=RCD_BINS,
        gamma=RCD_CHUNK_SIZE,
        localized=RCD_LOCALIZED,
        verbose=False,
    )

    return list(dict.fromkeys(result["root_cause"]))


def run_microcause(
    micro_cause,
    *,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    n_true_root_causes: int,
    algorithm_max_lag: int,
    sig_threshold: float,
    seed: int,
) -> List[str]:
    """Run MICROCause and keep the first k returned root causes."""
    ts_data = data.loc[:, list(series_names)].copy()

    anomalies_start_time = {
        name: anomaly_start
        for name in series_names
    }

    # MICROCause's random walk uses both random and numpy's global RNG.
    random.seed(seed)
    np.random.seed(seed)

    kwargs = dict(
        data=ts_data,
        anomalous_nodes=list(series_names),
        anomalies_start_time=anomalies_start_time,
        anomaly_length=anomaly_size,
        gamma_max=algorithm_max_lag,
        sig_threshold=sig_threshold,
    )

    if SUPPRESS_BASELINE_OUTPUT:
        with contextlib.redirect_stdout(io.StringIO()):
            rc_list = micro_cause(**kwargs)
    else:
        rc_list = micro_cause(**kwargs)

    # The provided MICROCause implementation returns up to two nodes. Because
    # this experiment has one true changed mechanism by default, truncate to k
    # for a comparable top-k localization evaluation.
    return list(rc_list)[:n_true_root_causes]

def run_circa(
    circa_cls,
    *,
    scg,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    n_true_root_causes: int,
    algorithm_max_lag: int,
    descendant_adjustment: bool = False,
) -> List[str]:
    """
    Run CIRCA using the supplied causal graph and return root causes
    whose raw  score exceeds a fixed threshold.
    For the current simulation:
      - cross-series causal effects are instantaneous;
      - each series has lag-1 self-dependence.

    Therefore CIRCA models Y_t using:
        Pa(Y)_t and Y_{t-1}.
    """

    # CIRCA must only receive the actual time-series variables.
    ts_data = data.loc[:, list(series_names)].copy()

    circa = circa_cls(
        parent_max_lag=0,
        include_self_lag=True,
        self_max_lag=algorithm_max_lag,
        descendant_adjustment=descendant_adjustment,
    )

    circa.fit(
        graph=scg,
        data=ts_data,
        anomaly_start=anomaly_start,
        anomaly_size=anomaly_size,
    )

    # return circa.predict_top_k(
    #     n_true_root_causes
    # )
    prediction = circa.predict_threshold(
            threshold=4,
    )
    return list(prediction)

def generate_random_instantaneous_ftdag(
    num_series: int,
    p_cross_edge: float,
    seed: int,
    true_max_lag: int = 1,
):
    if true_max_lag != 1:
        raise ValueError(
            "This generator assumes first-order self-dependence."
        )

    rng = np.random.default_rng(seed)

    series_names = [
        f"X{i}" for i in range(num_series)
    ]

    topological_order = list(
        rng.permutation(series_names)
    )

    cross_edges = []

    for i in range(num_series):
        for j in range(i + 1, num_series):
            if rng.random() < p_cross_edge:
                cross_edges.append(
                    (
                        topological_order[i],
                        topological_order[j],
                    )
                )

    if not cross_edges:
        i, j = sorted(
            rng.choice(
                num_series,
                size=2,
                replace=False,
            )
        )

        cross_edges.append(
            (
                topological_order[i],
                topological_order[j],
            )
        )

    ftdag = FtDirectedAcyclicGraph()

    for name in series_names:
        ftdag.add_vertex(
            DTimeVar(name, 0)
        )
        ftdag.add_vertex(
            DTimeVar(name, 1)
        )

    for name in series_names:
        ftdag.add_directed_edge(
            DTimeVar(name, 0),
            DTimeVar(name, 1),
        )

    for source, target in cross_edges:
        for t in (0, 1):
            ftdag.add_directed_edge(
                DTimeVar(source, t),
                DTimeVar(target, t),
            )

    return (
        ftdag,
        series_names,
        topological_order,
        cross_edges,
    )


def sample_signed_coefficient(
    rng: np.random.Generator,
    abs_low: float,
    abs_high: float,
) -> float:
    magnitude = float(
        rng.uniform(abs_low, abs_high)
    )

    sign = -1.0 if rng.random() < 0.5 else 1.0

    return sign * magnitude


def root_cause_metrics(
    truth: Iterable[str],
    prediction: Iterable[str],
) -> Dict[str, float]:
    truth = set(truth)
    prediction = set(prediction)

    tp = len(truth & prediction)
    fp = len(prediction - truth)
    fn = len(truth - prediction)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": float(tp > 0),
        "exact": float(truth == prediction),
        "n_predicted": float(len(prediction)),
    }


def summarize_results(
    raw_results: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        raw_results
        .groupby(
            ["algorithm", "anomaly_size"],
            as_index=False,
        )
        .agg(
            n_runs=("f1", "count"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            hit_rate=("hit", "mean"),
            exact_rate=("exact", "mean"),
            mean_n_predicted=("n_predicted", "mean"),
            runtime_mean=("runtime_seconds", "mean"),
            runtime_std=("runtime_seconds", "std"),
            failure_rate=(
                "status",
                lambda x: (x == "error").mean(),
            ),
        )
    )

    return summary