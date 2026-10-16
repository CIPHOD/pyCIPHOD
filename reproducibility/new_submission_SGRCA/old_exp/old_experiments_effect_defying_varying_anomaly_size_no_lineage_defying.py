from __future__ import annotations

import contextlib
import io
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyciphod.utils.graphs.temporal_graphs import FtDirectedAcyclicGraph
from pyciphod.utils.time_series.data_format import DTimeVar

from pyciphod.root_cause_analysis.graph_based import SGRCA
from baseline.tspc_compare import TsPCCompare
from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc, BINS


from common import (
    generate_random_instantaneous_ftdag,
    sample_signed_coefficient,
    root_cause_metrics,
)

# ============================================================================
# Hyperparameters
# ============================================================================

ANOMALY_SIZES = [30, 60, 125, 250, 500, 1000, 2000]

# Monte Carlo design
N_REPETITIONS = 20
N_SERIES = 10
P_CROSS_EDGE = 0.35
BASE_SEED = 42

# Temporal structure
TRUE_MAX_LAG = 1

# User-given max lag used by SGRCA (and MICROCause for a fair temporal search).
# Change this value for a max-lag sensitivity analysis.
ALGORITHM_MAX_LAG = 1

# Data generation
BURN_IN = 500
NORMAL_SIZE = 10 * max(ANOMALY_SIZES)
NOISE_STD = 1.0

SELF_WEIGHT_LOW = 0.20
SELF_WEIGHT_HIGH = 0.50

CROSS_WEIGHT_ABS_LOW = 0.40
CROSS_WEIGHT_ABS_HIGH = 0.90

# During the anomalous regime, beta_anomalous = EFFECT_MULTIPLIER * beta_normal.
EFFECT_MULTIPLIER = 2.5

# Statistical hyperparameters
SIG_THRESHOLD = 0.05

# RCD hyperparameters. Note that RCD's `gamma` is its variable-chunk size; it is
# unrelated to the temporal max lag above.
RCD_BINS = 5
RCD_CHUNK_SIZE = 5
RCD_LOCALIZED = False

# Algorithms to run. More methods can be added later through another adapter.
METHODS = ("SGRCA", "RCD", "MICROCause")

# Suppress the very verbose prints in the provided MICROCause implementation.
SUPPRESS_BASELINE_OUTPUT = True


# ============================================================================
# FtDAG generation
# ============================================================================


# def generate_random_instantaneous_ftdag(
#     num_series: int,
#     p_cross_edge: float,
#     seed: int,
# ):
#     """
#     Generate a random FtDAG with exactly the temporal structure of this study.
#
#     Structure
#     ---------
#     * Every X has X_{t-1} -> X_t.
#     * Every cross-series relation is instantaneous: X_t -> Y_t.
#     * Cross-series edges follow one random topological order, hence there is no
#       instantaneous directed cycle.
#     * Two time slices (0 and 1) are sufficient to encode the stationary
#       lag-0 / lag-1 template used here.
#
#     Returns
#     -------
#     ftdag : FtDirectedAcyclicGraph
#     series_names : list[str]
#     topological_order : list[str]
#     cross_edges : list[tuple[str, str]]
#     """
#     if TRUE_MAX_LAG != 1:
#         raise ValueError(
#             "This simulation is defined with true first-order self-dependence; "
#             "TRUE_MAX_LAG must be 1."
#         )
#
#     if num_series < 2:
#         raise ValueError("num_series must be at least 2.")
#
#     rng = np.random.default_rng(seed)
#
#     series_names = [f"X{i}" for i in range(num_series)]
#     topological_order = list(rng.permutation(series_names))
#
#     cross_edges: List[Tuple[str, str]] = []
#
#     for i in range(num_series):
#         for j in range(i + 1, num_series):
#             if rng.random() < p_cross_edge:
#                 cross_edges.append(
#                     (topological_order[i], topological_order[j])
#                 )
#
#     # We need at least one cross-series edge to create an effect-defying change.
#     if not cross_edges:
#         i, j = sorted(rng.choice(num_series, size=2, replace=False))
#         cross_edges.append(
#             (topological_order[i], topological_order[j])
#         )
#
#     ftdag = FtDirectedAcyclicGraph()
#
#     # Add two time slices.
#     for name in series_names:
#         ftdag.add_vertex(DTimeVar(name, 0))
#         ftdag.add_vertex(DTimeVar(name, 1))
#
#     # First-order self-dependence: X_0 -> X_1.
#     for name in series_names:
#         ftdag.add_directed_edge(
#             DTimeVar(name, 0),
#             DTimeVar(name, 1),
#         )
#
#     # Instantaneous cross-series relations at both slices.
#     for source, target in cross_edges:
#         ftdag.add_directed_edge(
#             DTimeVar(source, 0),
#             DTimeVar(target, 0),
#         )
#         ftdag.add_directed_edge(
#             DTimeVar(source, 1),
#             DTimeVar(target, 1),
#         )
#
#     return ftdag, series_names, topological_order, cross_edges


# ============================================================================
# Piecewise-linear data generation
# ============================================================================


def _sample_signed_coefficient(
    rng: np.random.Generator,
    abs_low: float,
    abs_high: float,
) -> float:
    magnitude = float(rng.uniform(abs_low, abs_high))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * magnitude


def simulate_effect_change_from_ftdag(
    *,
    series_names: Sequence[str],
    topological_order: Sequence[str],
    cross_edges: Sequence[Tuple[str, str]],
    normal_size: int,
    max_anomaly_size: int,
    burn_in: int,
    seed: int,
):
    """
    Simulate from the stationary FtDAG template with one temporary effect change.

    For each series V:

        V_t = phi_V V_{t-1}
              + sum_{U -> V} beta_{UV,r} U_t
              + N^V_t,

    where the cross-series graph is instantaneous and acyclic. Exactly one
    beta_{UV} changes at the anomaly onset. Noise distributions and all other
    mechanisms stay unchanged.

    Returns
    -------
    data : pandas.DataFrame
        Columns: time, X0, ..., Xp.
    anomaly_start : int
        Start in the returned (post-burn-in) dataframe.
    changed_edge : tuple[str, str]
    true_root_causes : set[str]
        Target of the changed direct effect.
    parameter_info : dict
    """
    rng = np.random.default_rng(seed)

    n_series = len(series_names)
    index_of = {name: i for i, name in enumerate(series_names)}

    self_coefficients = {
        name: float(rng.uniform(SELF_WEIGHT_LOW, SELF_WEIGHT_HIGH))
        for name in series_names
    }

    cross_coefficients = {
        edge: _sample_signed_coefficient(
            rng,
            CROSS_WEIGHT_ABS_LOW,
            CROSS_WEIGHT_ABS_HIGH,
        )
        for edge in cross_edges
    }

    changed_edge = cross_edges[int(rng.integers(len(cross_edges)))]
    beta_normal = cross_coefficients[changed_edge]
    beta_anomalous = EFFECT_MULTIPLIER * beta_normal

    parents: Dict[str, List[str]] = {name: [] for name in series_names}
    for source, target in cross_edges:
        parents[target].append(source)

    n_retained = normal_size + max_anomaly_size
    total_length = burn_in + n_retained
    anomaly_start_absolute = burn_in + normal_size

    values = np.zeros((total_length, n_series), dtype=float)

    for t in range(total_length):
        for target in topological_order:
            target_idx = index_of[target]

            value = 0.0

            # Lag-1 self-dependence.
            if t > 0:
                value += (
                    self_coefficients[target]
                    * values[t - 1, target_idx]
                )

            # Instantaneous cross-series parents have already been evaluated at t
            # because topological_order is respected.
            for source in parents[target]:
                beta = cross_coefficients[(source, target)]

                if (
                    (source, target) == changed_edge
                    and t >= anomaly_start_absolute
                ):
                    beta = beta_anomalous

                value += beta * values[t, index_of[source]]

            value += float(rng.normal(0.0, NOISE_STD))
            values[t, target_idx] = value

    retained = values[burn_in:]

    data = pd.DataFrame(retained, columns=series_names)
    data.insert(0, "time", np.arange(n_retained, dtype=int))

    anomaly_start = normal_size
    true_root_causes = {changed_edge[1]}

    parameter_info = {
        "changed_source": changed_edge[0],
        "changed_target": changed_edge[1],
        "beta_normal": beta_normal,
        "beta_anomalous": beta_anomalous,
        "self_coefficients": self_coefficients,
        "cross_coefficients": cross_coefficients,
    }

    return (
        data,
        anomaly_start,
        changed_edge,
        true_root_causes,
        parameter_info,
    )


# ============================================================================
# Algorithm adapters
# ============================================================================


def run_sgrca(
    SGRCA,
    *,
    scg,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
) -> Set[str]:
    """Return only SGRCA's effect-defying macro-nodes."""
    sgrca = SGRCA(
        g=scg,
        anomalous_vertices=list(series_names),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=ALGORITHM_MAX_LAG,
        sig_threshold=SIG_THRESHOLD,
    )

    root_causes = sgrca.run(data)

    prediction: Set[str] = set()
    for component_result in root_causes.values():
        prediction.update(component_result.get("effect_defying", []))
        prediction.update(component_result.get("noise_defying", []))


    return prediction


def run_rcd(
    rcd_top_k_rc,
    *,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    n_true_root_causes: int,
    seed: int,
) -> Set[str]:
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

    return set(result["root_cause"])


def run_microcause(
    micro_cause,
    *,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    n_true_root_causes: int,
    seed: int,
) -> Set[str]:
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
        gamma_max=ALGORITHM_MAX_LAG,
        sig_threshold=SIG_THRESHOLD,
    )

    if SUPPRESS_BASELINE_OUTPUT:
        with contextlib.redirect_stdout(io.StringIO()):
            rc_list = micro_cause(**kwargs)
    else:
        rc_list = micro_cause(**kwargs)

    # The provided MICROCause implementation returns up to two nodes. Because
    # this experiment has one true changed mechanism by default, truncate to k
    # for a comparable top-k localization evaluation.
    return set(list(rc_list)[:n_true_root_causes])


# # ============================================================================
# # Metrics
# # ============================================================================
#
#
# def root_cause_metrics(
#     truth: Iterable[str],
#     prediction: Iterable[str],
# ) -> Dict[str, float]:
#     truth = set(truth)
#     prediction = set(prediction)
#
#     tp = len(truth & prediction)
#     fp = len(prediction - truth)
#     fn = len(truth - prediction)
#
#     precision = tp / (tp + fp) if (tp + fp) else 0.0
#     recall = tp / (tp + fn) if (tp + fn) else 0.0
#     f1 = (
#         2.0 * precision * recall / (precision + recall)
#         if (precision + recall)
#         else 0.0
#     )
#
#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "hit": float(tp > 0),
#         "exact": float(truth == prediction),
#         "n_predicted": float(len(prediction)),
#     }


# ============================================================================
# Experiment
# ============================================================================


def run_experiment() -> pd.DataFrame:
    algorithms = {
        "SGRCA": SGRCA,
        "RCD": top_k_rc,
        "MICROCause": micro_cause,
    }
    rows = []

    if NORMAL_SIZE < 10 * max(ANOMALY_SIZES):
        warnings.warn(
            "NORMAL_SIZE is smaller than 10 * max(ANOMALY_SIZES). "
            "SGRCA will therefore use fewer than 10 normal batches for at "
            "least one anomaly size."
        )

    for repetition in range(N_REPETITIONS):
        graph_seed = BASE_SEED + 10_000 * repetition
        data_seed = graph_seed + 1

        (
            ftdag,
            series_names,
            topological_order,
            cross_edges,
        ) = generate_random_instantaneous_ftdag(
            num_series=N_SERIES,
            p_cross_edge=P_CROSS_EDGE,
            seed=graph_seed,
        )

        scg = ftdag.get_summary_causal_graph()

        (
            full_data,
            anomaly_start,
            changed_edge,
            true_root_causes,
            parameter_info,
        ) = simulate_effect_change_from_ftdag(
            series_names=series_names,
            topological_order=topological_order,
            cross_edges=cross_edges,
            normal_size=NORMAL_SIZE,
            max_anomaly_size=max(ANOMALY_SIZES),
            burn_in=BURN_IN,
            seed=data_seed,
        )

        for anomaly_size in ANOMALY_SIZES:
            # Nested anomalous windows: the normal prefix, graph, coefficients,
            # and initial anomalous observations are identical across sizes.
            data = full_data.iloc[
                : anomaly_start + anomaly_size
            ].copy()

            for algorithm_name in METHODS:
                method_seed = (
                    BASE_SEED
                    + repetition * 100_000
                    + anomaly_size * 10
                    + METHODS.index(algorithm_name)
                )

                start_time = time.perf_counter()
                error = ""

                try:
                    if algorithm_name == "SGRCA":
                        prediction = run_sgrca(
                            algorithms["SGRCA"],
                            scg=scg,
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                        )

                    elif algorithm_name == "RCD":
                        prediction = run_rcd(
                            algorithms["RCD"],
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            n_true_root_causes=len(true_root_causes),
                            seed=method_seed,
                        )

                    elif algorithm_name == "MICROCause":
                        prediction = run_microcause(
                            algorithms["MICROCause"],
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            n_true_root_causes=len(true_root_causes),
                            seed=method_seed,
                        )

                    else:
                        raise ValueError(
                            f"Unknown algorithm: {algorithm_name}"
                        )

                    metrics = root_cause_metrics(
                        true_root_causes,
                        prediction,
                    )
                    status = "ok"

                except Exception as exc:
                    prediction = set()
                    metrics = {
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "hit": np.nan,
                        "exact": np.nan,
                        "n_predicted": np.nan,
                    }
                    status = "error"
                    error = repr(exc)

                runtime = time.perf_counter() - start_time

                rows.append(
                    {
                        "repetition": repetition,
                        "graph_seed": graph_seed,
                        "data_seed": data_seed,
                        "algorithm": algorithm_name,
                        "anomaly_size": anomaly_size,
                        "normal_size": NORMAL_SIZE,
                        "true_max_lag": TRUE_MAX_LAG,
                        "algorithm_max_lag": ALGORITHM_MAX_LAG,
                        "n_series": N_SERIES,
                        "n_cross_edges": len(cross_edges),
                        "changed_source": changed_edge[0],
                        "changed_target": changed_edge[1],
                        "beta_normal": parameter_info["beta_normal"],
                        "beta_anomalous": parameter_info["beta_anomalous"],
                        "true_root_causes": ",".join(
                            sorted(true_root_causes)
                        ),
                        "predicted_root_causes": ",".join(
                            sorted(prediction)
                        ),
                        "runtime_seconds": runtime,
                        "status": status,
                        "error": error,
                        **metrics,
                    }
                )

        print(
            f"Completed repetition {repetition + 1}/{N_REPETITIONS} "
            f"| changed edge {changed_edge[0]} -> {changed_edge[1]}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# Summaries and plots
# ============================================================================


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    valid = raw_results.loc[raw_results["status"] == "ok"].copy()

    summary = (
        valid.groupby(["algorithm", "anomaly_size"], as_index=False)
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
        )
    )

    return summary


def _plot_metric(
    summary: pd.DataFrame,
    *,
    y_column: str,
    ylabel: str,
    output_file: Path,
    ylim: Tuple[float, float] | None = None,
):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for algorithm in METHODS:
        d = summary.loc[
            summary["algorithm"] == algorithm
        ].sort_values("anomaly_size")

        if d.empty:
            continue

        ax.plot(
            d["anomaly_size"],
            d[y_column],
            marker="o",
            label=algorithm,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(ANOMALY_SIZES)
    ax.set_xticklabels([str(x) for x in ANOMALY_SIZES])
    ax.set_xlabel("Anomaly size")
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    plt.close(fig)



# ============================================================================
# Main
# ============================================================================


def main():
    warnings.filterwarnings("ignore")

    if ALGORITHM_MAX_LAG < 0:
        raise ValueError("ALGORITHM_MAX_LAG must be non-negative.")

    print("Effect-defying anomaly-size simulation")
    print(f"Anomaly sizes:       {ANOMALY_SIZES}")
    print(f"Repetitions:         {N_REPETITIONS}")
    print(f"Number of series:    {N_SERIES}")
    print(f"Normal sample size:  {NORMAL_SIZE}")
    print(f"True max lag:        {TRUE_MAX_LAG}")
    print(f"Algorithm max lag:   {ALGORITHM_MAX_LAG}")
    print(f"Algorithms:          {METHODS}")

    raw_results = run_experiment()

    summary = summarize_results(raw_results)

    # ---------------------------------------------------------
    # Results for each anomaly size
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS BY ANOMALY SIZE")
    print("=" * 80)

    print(
        summary[
            [
                "algorithm",
                "anomaly_size",
                "f1_mean",
                "f1_std",
                "runtime_mean",
                "runtime_std",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )




if __name__ == "__main__":
    main()
