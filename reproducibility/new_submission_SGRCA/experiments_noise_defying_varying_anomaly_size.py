from __future__ import annotations

import contextlib
import io
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from pyciphod.utils.graphs.temporal_graphs import FtDirectedAcyclicGraph
from pyciphod.utils.time_series.data_format import DTimeVar

from pyciphod.root_cause_analysis.graph_based import SGRCA, FTDAGRCA, EasyRCA
from pyciphod.root_cause_analysis.data_based import SGRCAStar, EasyRCAStar, FTDAGRCAStar

from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc
from baseline.circa import CIRCA

from common import (
    run_sgrca, run_sgrca_star,
    run_easyrca, run_easyrca_star,
    run_ftdagrca, run_ftdagrca_star,
    run_rcd, run_microcause, run_circa,
    generate_random_instantaneous_ftdag,
    sample_signed_coefficient,
    root_cause_metrics,
    summarize_results
)

# ============================================================================
# Hyperparameters
# ============================================================================

ANOMALY_SIZES = [30, 60, 125, 250, 500, 1000, 2000]

# Monte Carlo design
N_REPETITIONS = 20
N_SERIES = 5
P_CROSS_EDGE = 0.35
BASE_SEED = 42

# Temporal structure
TRUE_MAX_LAG = 1

# User-given maximum lag used by SGRCA and MICROCause.
ALGORITHM_MAX_LAG = 1

# Data generation
BURN_IN = 500
NORMAL_SIZE = 10 * max(ANOMALY_SIZES)

# Keep these structural ranges aligned with the effect-defying experiment.
SELF_WEIGHT_LOW = 0.20
SELF_WEIGHT_HIGH = 0.50

CROSS_WEIGHT_ABS_LOW = 0.40
CROSS_WEIGHT_ABS_HIGH = 0.90

NOISE_MEAN = 0.0
NORMAL_NOISE_STD = 1.0
LINEAGE_NOISE_CHANGE_MEAN_OR_VAR = 0  # 0: variance change, 1: mean change
ANOMALOUS_NOISE_MULTIPLIER = 3.0
LINEAGE_EVAL = True  # True for Yes False for No

# Statistical hyperparameters
SIG_THRESHOLD = 0.05

# Algorithms to run.
METHODS = ("SGRCA", "SGRCA*", "EasyRCA", "EasyRCA*", "FT-DAG-RCA", "FT-DAG-RCA*", "CIRCA", "RCD-1", "RCD-all", "MICROCause", "MICROCause-1")

# METHODS = ("SGRCA", "CIRCA")


def _cross_parents(scg, node) -> Set[str]:
    """Macro parents excluding the self-loop."""
    return set(scg.get_parents(node)) - {node}


def _candidate_noise_root_pairs(scg) -> List[Tuple[str, str]]:
    """
    Return X -> Y pairs for which X has no cross-series parent.

    X is therefore handled by SGRCA's lineage-defying stage, whereas Y has
    at least the cross-series parent X and is the noise-defying root evaluated
    in this experiment.
    """
    candidates: List[Tuple[str, str]] = []

    for source, target in scg.get_directed_edges():
        if source == target:
            continue

        if not _cross_parents(scg, source):
            candidates.append((source, target))

    return candidates


# ============================================================================
# Piecewise-linear data generation
# ============================================================================


def simulate_two_noise_changes_from_ftdag(
    *,
    series_names: Sequence[str],
    topological_order: Sequence[str],
    cross_edges: Sequence[Tuple[str, str]],
    candidate_pairs: Sequence[Tuple[str, str]],
    normal_size: int,
    max_anomaly_size: int,
    burn_in: int,
    seed: int,
):
    """
    Simulate one continuous trajectory with two temporary noise changes.

    For each series V,

        V_t = phi_V V_{t-1}
              + sum_{U -> V} beta_{UV} U_t
              + N^V_t.

    The graph and all structural coefficients stay unchanged throughout the
    simulation.

    * lineage_root: has no cross-series parent and is handled by SGRCA as
      lineage-defying;
    * tested_noise_root: a child of lineage_root and the noise-defying root
      evaluated in this experiment.

    The anomalous regime is generated on the same trajectory as the normal
    regime, exactly as in the effect-defying experiment.

    Returns
    -------
    data : pandas.DataFrame
        Columns: time, X0, ..., Xp.
    anomaly_start : int
    all_noise_root_causes : set[str]
        Both variables whose noise mechanisms change.
    evaluated_noise_root_causes : set[str]
        Only the non-lineage noise-defying root.
    parameter_info : dict
    """
    if not candidate_pairs:
        raise ValueError("candidate_pairs must contain at least one X -> Y pair.")

    rng = np.random.default_rng(seed)

    lineage_root, tested_noise_root = candidate_pairs[
        int(rng.integers(len(candidate_pairs)))
    ]

    n_series = len(series_names)
    index_of = {name: i for i, name in enumerate(series_names)}

    # Same structural parameterization as the effect-defying experiment.
    self_coefficients = {
        name: float(rng.uniform(SELF_WEIGHT_LOW, SELF_WEIGHT_HIGH))
        for name in series_names
    }

    cross_coefficients = {
        edge: sample_signed_coefficient(
            rng,
            CROSS_WEIGHT_ABS_LOW,
            CROSS_WEIGHT_ABS_HIGH,
        )
        for edge in cross_edges
    }

    parents: Dict[str, List[str]] = {name: [] for name in series_names}
    for source, target in cross_edges:
        parents[target].append(source)

    n_retained = normal_size + max_anomaly_size
    total_length = burn_in + n_retained
    anomaly_start_absolute = burn_in + normal_size

    values = np.zeros((total_length, n_series), dtype=float)

    for t in range(total_length):
        anomalous = t >= anomaly_start_absolute

        for target in topological_order:
            target_idx = index_of[target]

            value = 0.0

            # Lag-1 self-dependence.
            if t > 0:
                value += (
                    self_coefficients[target]
                    * values[t - 1, target_idx]
                )

            # Instantaneous parents are already available at time t because
            # targets are evaluated according to topological_order.
            for source in parents[target]:
                value += (
                    cross_coefficients[(source, target)]
                    * values[t, index_of[source]]
                )

            sigma = NORMAL_NOISE_STD
            mean = NOISE_MEAN
            if anomalous and target == lineage_root:
                if LINEAGE_NOISE_CHANGE_MEAN_OR_VAR == 0:
                    sigma *= ANOMALOUS_NOISE_MULTIPLIER
                else:
                    mean += ANOMALOUS_NOISE_MULTIPLIER

            elif anomalous and target == tested_noise_root:
                    sigma *= ANOMALOUS_NOISE_MULTIPLIER

            value += float(rng.normal(mean, sigma))
            values[t, target_idx] = value

    retained = values[burn_in:]

    data = pd.DataFrame(retained, columns=series_names)
    data.insert(0, "time", np.arange(n_retained, dtype=int))

    anomaly_start = normal_size

    all_noise_root_causes = {
        lineage_root,
        tested_noise_root,
    }

    # This benchmark evaluates the root that reaches SGRCA's noise-defying
    # stage; the lineage root is intentionally excluded from the ground truth.
    if LINEAGE_EVAL:
        evaluated_noise_root_causes = all_noise_root_causes
    else:
        evaluated_noise_root_causes = {
            tested_noise_root,
        }

    parameter_info = {
        "lineage_noise_root": lineage_root,
        "tested_noise_root": tested_noise_root,
        "normal_noise_std": NORMAL_NOISE_STD,
        "anomalous_noise_std": (
            NORMAL_NOISE_STD * ANOMALOUS_NOISE_MULTIPLIER
        ),
        "self_coefficients": self_coefficients,
        "cross_coefficients": cross_coefficients,
    }

    return (
        data,
        anomaly_start,
        all_noise_root_causes,
        evaluated_noise_root_causes,
        parameter_info,
    )



# ============================================================================
# Experiment
# ============================================================================


def run_experiment() -> pd.DataFrame:
    algorithms = {
        "SGRCA": SGRCA,
        "SGRCA*": SGRCAStar,
        "EasyRCA": EasyRCA,
        "EasyRCA*": EasyRCAStar,
        "FT-DAG-RCA": FTDAGRCA,
        "FT-DAG-RCA*": FTDAGRCAStar,
        "CIRCA": CIRCA,
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

        candidate_pairs = _candidate_noise_root_pairs(scg)
        if not candidate_pairs:
            # This should not occur for a non-empty DAG generated from a
            # topological order, but keep the check explicit.
            raise RuntimeError(
                "Generated cross-series DAG contains no candidate "
                "lineage-root -> tested-noise-root pair."
            )

        (
            full_data,
            anomaly_start,
            all_noise_root_causes,
            evaluated_noise_root_causes,
            parameter_info,
        ) = simulate_two_noise_changes_from_ftdag(
            series_names=series_names,
            topological_order=topological_order,
            cross_edges=cross_edges,
            candidate_pairs=candidate_pairs,
            normal_size=NORMAL_SIZE,
            max_anomaly_size=max(ANOMALY_SIZES),
            burn_in=BURN_IN,
            seed=data_seed,
        )

        lineage_noise_root = parameter_info["lineage_noise_root"]
        tested_noise_root = parameter_info["tested_noise_root"]

        # Sanity checks defining the benchmark.
        assert not _cross_parents(scg, lineage_noise_root)
        assert _cross_parents(scg, tested_noise_root)
        assert (lineage_noise_root, tested_noise_root) in set(cross_edges)

        # Variables structurally affected by either changed innovation.
        anomalous_vertices = (
            set(scg.get_descendants(lineage_noise_root))
            | set(scg.get_descendants(tested_noise_root))
            | {lineage_noise_root, tested_noise_root}
        )

        for anomaly_size in ANOMALY_SIZES:
            # Nested anomalous windows: the normal prefix, graph, coefficients,
            # and initial anomalous observations are identical across sizes.
            data = full_data.iloc[
                : anomaly_start + anomaly_size
            ].copy()

            for algorithm_name in METHODS:
                if algorithm_name.startswith("RCD"):
                    method_seed = (
                            BASE_SEED
                            + repetition * 100_000
                            + anomaly_size * 10
                            + 1
                    )
                else:
                    method_seed = (
                            BASE_SEED
                            + repetition * 100_000
                            + anomaly_size * 10
                            + METHODS.index(algorithm_name)
                    )
                # method_seed = (
                #     BASE_SEED
                #     + repetition * 100_000
                #     + anomaly_size * 10
                #     + METHODS.index(algorithm_name)
                # )

                start_time = time.perf_counter()
                error = ""

                try:
                    if algorithm_name == "SGRCA":
                        prediction = run_sgrca(
                            algorithms["SGRCA"],
                            scg=scg,
                            data=data,
                            anomalous_vertices=anomalous_vertices,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                        )

                    elif algorithm_name == "EasyRCA":
                        prediction = run_easyrca(
                            algorithms["EasyRCA"],
                            scg=scg,
                            data=data,
                            anomalous_vertices=anomalous_vertices,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                        )
                        prediction = set(prediction)

                    elif algorithm_name == "FT-DAG-RCA":
                        prediction = run_ftdagrca(
                            algorithms["FT-DAG-RCA"],
                            ftdag=ftdag,
                            data=data,
                            anomalous_vertices=anomalous_vertices,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                        )
                        prediction = set(prediction)

                    elif algorithm_name == "SGRCA*":
                        prediction = run_sgrca_star(
                            algorithms["SGRCA*"],
                            data=data,
                            anomalous_vertices=anomalous_vertices,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                        )

                        prediction = set(prediction)
                        if not LINEAGE_EVAL:
                            prediction.discard(lineage_noise_root)

                    elif algorithm_name == "FT-DAG-RCA*":
                        prediction = run_ftdagrca_star(
                            algorithms["FT-DAG-RCA*"],
                            data=data,
                            anomalous_vertices=anomalous_vertices,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                        )

                        prediction = set(prediction)
                        if not LINEAGE_EVAL:
                            prediction.discard(lineage_noise_root)

                    elif algorithm_name == "EasyRCA*":
                        prediction = run_easyrca_star(
                            algorithms["EasyRCA*"],
                            data=data,
                            anomalous_vertices=anomalous_vertices,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                        )

                        prediction = set(prediction)
                        if not LINEAGE_EVAL:
                            prediction.discard(lineage_noise_root)

                    elif algorithm_name in {"RCD-all", "RCD-1"}:
                        prediction = run_rcd(
                            algorithms["RCD"],
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            n_true_root_causes=N_SERIES,
                            seed=method_seed,
                        )
                        if not LINEAGE_EVAL:
                            prediction = [node for node in prediction if node != lineage_noise_root]
                        if algorithm_name == "RCD-1":
                            if not LINEAGE_EVAL:
                                prediction = set(prediction[:1])
                            else:
                                prediction = set(prediction[:2])
                        else:
                            prediction = set(prediction)

                    # elif algorithm_name == "MICROCause":
                    elif algorithm_name in {"MICROCause", "MICROCause-1"}:
                        prediction = run_microcause(
                            algorithms["MICROCause"],
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            n_true_root_causes=N_SERIES,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            sig_threshold=SIG_THRESHOLD,
                            seed=method_seed,
                        )
                        if not LINEAGE_EVAL:
                            prediction = [node for node in prediction if node != lineage_noise_root]
                        if algorithm_name == "MICROCause-1":
                            if not LINEAGE_EVAL:
                                prediction = set(prediction[:1])
                            else:
                                prediction = set(prediction[:2])

                    elif algorithm_name == "CIRCA":
                        prediction = run_circa(
                            algorithms["CIRCA"],
                            scg=scg,
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            n_true_root_causes=(
                                2 if LINEAGE_EVAL else 1
                            ),
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            descendant_adjustment=False,
                        )

                        prediction = set(prediction)

                        if not LINEAGE_EVAL:
                            prediction.discard(lineage_noise_root)

                    else:
                        raise ValueError(
                            f"Unknown algorithm: {algorithm_name}"
                        )

                    metrics = root_cause_metrics(
                        evaluated_noise_root_causes,
                        prediction,
                    )
                    status = "ok"

                # except Exception as exc:
                #     prediction = set()
                #     metrics = {
                #         "precision": np.nan,
                #         "recall": np.nan,
                #         "f1": np.nan,
                #         "hit": np.nan,
                #         "exact": np.nan,
                #         "n_predicted": np.nan,
                #     }
                #     status = "error"
                #     error = repr(exc)
                except Exception as exc:
                    prediction = set()
                    metrics = root_cause_metrics(
                        evaluated_noise_root_causes,
                        prediction,
                    )
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
                        "n_scg_edges": len(scg.get_directed_edges()),
                        "lineage_noise_root": lineage_noise_root,
                        "tested_noise_root": tested_noise_root,
                        "all_noise_root_causes": ",".join(
                            sorted(all_noise_root_causes)
                        ),
                        "evaluated_noise_root_causes": ",".join(
                            sorted(evaluated_noise_root_causes)
                        ),
                        "predicted_root_causes": ",".join(
                            sorted(prediction)
                        ),
                        "normal_noise_std": parameter_info[
                            "normal_noise_std"
                        ],
                        "anomalous_noise_std": parameter_info[
                            "anomalous_noise_std"
                        ],
                        "runtime_seconds": runtime,
                        "status": status,
                        "error": error,
                        **metrics,
                    }
                )

        print(
            f"Completed repetition {repetition + 1}/{N_REPETITIONS} "
            f"| lineage noise root: {lineage_noise_root} "
            f"| tested noise root: {tested_noise_root}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# Summaries
# ============================================================================


# def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
#     valid = raw_results.loc[raw_results["status"] == "ok"].copy()
#
#     return (
#         valid.groupby(["algorithm", "anomaly_size"], as_index=False)
#         .agg(
#             n_runs=("f1", "count"),
#             precision_mean=("precision", "mean"),
#             precision_std=("precision", "std"),
#             recall_mean=("recall", "mean"),
#             recall_std=("recall", "std"),
#             f1_mean=("f1", "mean"),
#             f1_std=("f1", "std"),
#             hit_rate=("hit", "mean"),
#             exact_rate=("exact", "mean"),
#             mean_n_predicted=("n_predicted", "mean"),
#             runtime_mean=("runtime_seconds", "mean"),
#             runtime_std=("runtime_seconds", "std"),
#         )
#     )


# ============================================================================
# Main
# ============================================================================


def main():
    warnings.filterwarnings("ignore")

    if ALGORITHM_MAX_LAG < 0:
        raise ValueError("ALGORITHM_MAX_LAG must be non-negative.")

    print("Noise-defying anomaly-size simulation")
    print(f"Anomaly sizes:             {ANOMALY_SIZES}")
    print(f"Repetitions:               {N_REPETITIONS}")
    print(f"Number of series:          {N_SERIES}")
    print(f"Normal sample size:        {NORMAL_SIZE}")
    print(f"True max lag:              {TRUE_MAX_LAG}")
    print(f"Algorithm max lag:         {ALGORITHM_MAX_LAG}")
    print(f"Noise multiplier:          {ANOMALOUS_NOISE_MULTIPLIER}")
    print(f"Algorithms:                {METHODS}")
    if LINEAGE_EVAL:
        print(
            "Evaluation target: lineage + "
            "non-lineage noise-defying roots"
        )
    else:
        print(
            "Evaluation target: non-lineage "
            "noise-defying root only"
        )

    raw_results = run_experiment()
    summary = summarize_results(raw_results)

    print("\n" + "=" * 88)
    print("RESULTS BY ANOMALY SIZE")
    if LINEAGE_EVAL:
        print(
            "F1 evaluates BOTH planted "
            "noise-defying root causes."
        )
    else:
        print(
            "F1 evaluates ONLY the non-lineage "
            "noise-defying root cause."
        )
        print("=" * 88)

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
