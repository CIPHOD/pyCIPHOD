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

from pyciphod.root_cause_analysis.graph_based import SGRCA, FTDAGRCA, EasyRCA
from pyciphod.root_cause_analysis.data_based import SGRCAStar, EasyRCAStar, FTDAGRCAStar

from baseline.tspc_compare import TsPCCompare
from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc, BINS
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
BASE_SEED = 43

# Temporal structure
TRUE_MAX_LAG = 1

# User-given max lag used by SGRCA (and MICROCause for a fair temporal search).
# Change this value for a max-lag sensitivity analysis.
ALGORITHM_MAX_LAG = 1

# Data generation
BURN_IN = 500
NORMAL_SIZE = 10 * max(ANOMALY_SIZES)
NOISE_MEAN = 0.0
NOISE_STD = 1.0
LINEAGE_NOISE_CHANGE_MEAN_OR_VAR = 0  # 0 for var 1 for mean
LINEAGE_NOISE_MULTIPLIER = 3.0
SELF_EFFECT_MULTIPLIER = 1.7
LINEAGE_EVAL = True  # True for Yes False for No

SELF_WEIGHT_LOW = 0.20
SELF_WEIGHT_HIGH = 0.50
CROSS_WEIGHT_ABS_LOW = 0.40
CROSS_WEIGHT_ABS_HIGH = 0.90

# SELF_WEIGHT_LOW = 0.20
# SELF_WEIGHT_HIGH = 0.50
# CROSS_WEIGHT_ABS_LOW = 0.20
# CROSS_WEIGHT_ABS_HIGH = 0.50

# During the anomalous regime, beta_anomalous = EFFECT_MULTIPLIER * beta_normal.
EFFECT_MULTIPLIER = 2.5

# Statistical hyperparameters
SIG_THRESHOLD = 0.05


# Algorithms to run. More methods can be added later through another adapter.
METHODS = ("SGRCA", "SGRCA*", "EasyRCA", "EasyRCA*", "FT-DAG-RCA", "FT-DAG-RCA*",  "CIRCA", "RCD-all", "RCD-1", "MICROCause", "MICROCause-1")

# METHODS = ("SGRCA", "CIRCA")


def _cross_parents(scg, node) -> Set[str]:
    """Macro parents excluding the self-loop."""
    return set(scg.get_parents(node)) - {node}


def _candidate_effect_root_pairs(scg) -> List[Tuple[str, str]]:
    """
    Candidate X -> Y pairs such that X has no cross-series parent.

    X will be made lineage-defying through a noise change, while Y will be
    made an additional non-lineage effect-defying root through a change in
    the coefficient of X -> Y.
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


def simulate_effect_change_from_ftdag(
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
    Simulate one continuous trajectory with two planted root causes.

    Select X -> Y such that X has no cross-series parent. At the anomaly onset:

    * X receives a noise change and is therefore the lineage-defying root;
    * the coefficient of X -> Y changes, making Y an additional non-lineage
      effect-defying root cause.

    All other structural coefficients and all other noise distributions remain
    unchanged. The reported benchmark evaluates only Y, analogously to the
    noise-defying experiment, which evaluates only its additional non-lineage
    root cause.
    """
    if not candidate_pairs:
        raise ValueError("candidate_pairs must contain at least one X -> Y pair.")

    rng = np.random.default_rng(seed)

    lineage_root, tested_effect_root = candidate_pairs[
        int(rng.integers(len(candidate_pairs)))
    ]
    changed_edge = (lineage_root, tested_effect_root)

    n_series = len(series_names)
    index_of = {name: i for i, name in enumerate(series_names)}

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
        anomalous = t >= anomaly_start_absolute

        for target in topological_order:
            target_idx = index_of[target]
            value = 0.0

            # Lag-1 self-dependence.
            if t > 0:
                self_beta = self_coefficients[target]

                # Structural change X_{t-1} -> X_t for the lineage root.
                if anomalous and target == lineage_root:
                    anomalous_self_coefficient = (SELF_EFFECT_MULTIPLIER * self_beta)
                    self_beta = anomalous_self_coefficient

                value += (
                    self_beta
                    * values[t - 1, target_idx]
                )

            # Instantaneous cross-series parents are already evaluated at t
            # because the topological order is respected.
            for source in parents[target]:
                beta = cross_coefficients[(source, target)]

                if anomalous and (source, target) == changed_edge:
                    beta = beta_anomalous

                value += beta * values[t, index_of[source]]

            # Only X receives a noise change. Y's noise stays unchanged.
            sigma = NOISE_STD
            mean = NOISE_MEAN
            # if anomalous and target == lineage_root:
            #     if LINEAGE_NOISE_CHANGE_MEAN_OR_VAR == 0:
            #         sigma *= LINEAGE_NOISE_MULTIPLIER
            #     else:
            #         mean += LINEAGE_NOISE_MULTIPLIER

            value += float(rng.normal(mean, sigma))
            values[t, target_idx] = value

    retained = values[burn_in:]

    data = pd.DataFrame(retained, columns=series_names)
    data.insert(0, "time", np.arange(n_retained, dtype=int))

    anomaly_start = normal_size

    all_root_causes = {lineage_root, tested_effect_root}
    if LINEAGE_EVAL:
        evaluated_effect_root_causes = all_root_causes
    else:
        evaluated_effect_root_causes = {tested_effect_root}

    parameter_info = {
        "lineage_root": lineage_root,
        "tested_effect_root": tested_effect_root,
        "changed_source": changed_edge[0],
        "changed_target": changed_edge[1],
        "beta_normal": beta_normal,
        "beta_anomalous": beta_anomalous,
        "normal_noise_std": NOISE_STD,
        # "lineage_anomalous_noise_std": (
        #     NOISE_STD * LINEAGE_NOISE_MULTIPLIER
        # ),
        "self_coefficients": self_coefficients,
        "cross_coefficients": cross_coefficients,
    }

    return (
        data,
        anomaly_start,
        changed_edge,
        all_root_causes,
        evaluated_effect_root_causes,
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

        candidate_pairs = _candidate_effect_root_pairs(scg)
        if not candidate_pairs:
            raise RuntimeError(
                "Generated graph contains no candidate "
                "lineage-root -> effect-root pair."
            )

        (
            full_data,
            anomaly_start,
            changed_edge,
            all_root_causes,
            evaluated_effect_root_causes,
            parameter_info,
        ) = simulate_effect_change_from_ftdag(
            series_names=series_names,
            topological_order=topological_order,
            cross_edges=cross_edges,
            candidate_pairs=candidate_pairs,
            normal_size=NORMAL_SIZE,
            max_anomaly_size=max(ANOMALY_SIZES),
            burn_in=BURN_IN,
            seed=data_seed,
        )

        lineage_root = parameter_info["lineage_root"]
        tested_effect_root = parameter_info["tested_effect_root"]

        # Sanity checks defining the experiment.
        assert not _cross_parents(scg, lineage_root)
        assert _cross_parents(scg, tested_effect_root)
        assert (lineage_root, tested_effect_root) in set(cross_edges)

        # Nodes structurally affected by the two planted mechanism changes.
        anomalous_vertices = (
            set(scg.get_descendants(lineage_root))
            | set(scg.get_descendants(tested_effect_root))
            | {lineage_root, tested_effect_root}
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
                        prediction = set(prediction)

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
                            prediction.discard(lineage_root)

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
                            prediction.discard(lineage_root)

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
                            prediction.discard(lineage_root)

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
                            prediction.discard(lineage_root)

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
                            prediction = [node for node in prediction if node != lineage_root]
                        if algorithm_name == "RCD-1":
                            if LINEAGE_EVAL:
                                prediction = set(prediction[:2])
                            else:
                                prediction = set(prediction[:1])
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
                            sig_threshold = SIG_THRESHOLD,
                            seed=method_seed,
                        )
                        if not LINEAGE_EVAL:
                            prediction = [node for node in prediction if node != lineage_root]
                        if algorithm_name == "MICROCause-1":
                            if not LINEAGE_EVAL:
                                prediction = set(prediction[:1])
                            else:
                                prediction = set(prediction[:2])
                    else:
                        raise ValueError(
                            f"Unknown algorithm: {algorithm_name}"
                        )

                    metrics = root_cause_metrics(
                        evaluated_effect_root_causes,
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
                        evaluated_effect_root_causes,
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
                        "lineage_root": lineage_root,
                        "tested_effect_root": tested_effect_root,
                        "changed_source": changed_edge[0],
                        "changed_target": changed_edge[1],
                        "beta_normal": parameter_info["beta_normal"],
                        "beta_anomalous": parameter_info["beta_anomalous"],
                        "normal_noise_std": parameter_info["normal_noise_std"],
                        # "lineage_anomalous_noise_std": parameter_info[
                        #     "lineage_anomalous_noise_std"
                        # ],
                        "all_root_causes": ",".join(
                            sorted(all_root_causes)
                        ),
                        "evaluated_effect_root_causes": ",".join(
                            sorted(evaluated_effect_root_causes)
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
            f"| lineage root: {lineage_root} "
            f"| tested effect root: {tested_effect_root} "
            f"| changed edge {changed_edge[0]} -> {changed_edge[1]}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# Summaries and plots
# ============================================================================


# def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
#     valid = raw_results.loc[raw_results["status"] == "ok"].copy()
#
#     summary = (
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
#
#     return summary


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
    print(f"Lineage noise multiplier: {LINEAGE_NOISE_MULTIPLIER}")
    if LINEAGE_EVAL:
        print("Evaluation target:   lineage + effect-defying roots")
    else:
        print("Evaluation target:   non-lineage effect-defying root only")

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
