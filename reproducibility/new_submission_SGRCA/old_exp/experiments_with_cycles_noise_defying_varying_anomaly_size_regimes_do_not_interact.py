from __future__ import annotations

import time
import warnings
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag
from pyciphod.utils.scms.dynamic_scm import (
    LinearDtDynamicSCM,
    create_random_linear_dt_dynamic_scm_from_ftdag,
)

from pyciphod.root_cause_analysis.graph_based import SGRCA, FTDAGRCA, EasyRCA
from pyciphod.root_cause_analysis.data_based import (
    SGRCAStar,
    EasyRCAStar,
    FTDAGRCAStar,
)
from pyciphod.root_cause_analysis.graph_based import SGRCAPlus

from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc

from common import (
    run_sgrca,
    run_sgrca_star,
    run_easyrca,
    run_easyrca_star,
    run_ftdagrca,
    run_ftdagrca_star,
    run_rcd,
    run_microcause,
    root_cause_metrics,
    summarize_results,
)


# ============================================================================
# Hyperparameters
# ============================================================================

ANOMALY_SIZES = [30, 60, 125, 250, 500, 1000, 2000]

# Monte Carlo design
N_REPETITIONS = 20
N_SERIES = 5
P_EDGE = 0.35
BASE_SEED = 42
MAX_GRAPH_ATTEMPTS = 500

# If False, the randomly generated SCG is allowed to be either cyclic or
# acyclic. Set this to True if you want a benchmark containing only cyclic SCGs.
REQUIRE_CYCLIC_SCG = False

# Temporal structure
TRUE_MAX_LAG = 1
ALGORITHM_MAX_LAG = 1

# Data generation
BURN_IN = 500
NORMAL_SIZE = 10 * max(ANOMALY_SIZES)

# WEIGHT_LOW = -0.40
# WEIGHT_HIGH = 0.40
WEIGHT_LOW = -0.90
WEIGHT_HIGH = 0.90
INTERCEPT_LOW = 0.0
INTERCEPT_HIGH = 0.0

NOISE_MEAN = 0.0
NORMAL_NOISE_STD = 1.0
LINEAGE_NOISE_CHANGE_MEAN_OR_VAR = 0  # 0: variance, 1: mean
ANOMALOUS_NOISE_MULTIPLIER = 3.0

SIG_THRESHOLD = 0.05

METHODS = (
    "SGRCA",
    "SGRCA+",
    "SGRCA*",
    "EasyRCA",
    "EasyRCA*",
    "FT-DAG-RCA",
    "FT-DAG-RCA*",
    "RCD-1",
    "RCD-all",
    "MICROCause",
)


# ============================================================================
# Graph / SCM helpers
# ============================================================================


def _cross_parents(scg, node) -> Set[str]:
    """Macro parents excluding a possible self-loop."""
    return set(scg.get_parents(node)) - {node}


def _scg_has_nontrivial_cycle(scg) -> bool:
    return any(
        len(set(cycle)) > 1
        for cycle in scg.get_all_simple_cycles()
    )


def _n_nontrivial_scg_cycles(scg) -> int:
    return sum(
        len(set(cycle)) > 1
        for cycle in scg.get_all_simple_cycles()
    )


def _macro_cross_edges(scg) -> Set[Tuple[str, str]]:
    return {
        (source, target)
        for source, target in scg.get_directed_edges()
        if source != target
    }


def _candidate_noise_root_pairs(scg) -> List[Tuple[str, str]]:
    """
    Candidate X -> Y macro relations such that X has no cross-series parent.

    X is therefore lineage-defying when its innovation changes, while Y is the
    additional non-lineage noise-defying root evaluated by this benchmark.
    """
    candidates: List[Tuple[str, str]] = []

    for source, target in scg.get_directed_edges():
        if source == target:
            continue

        if not _cross_parents(scg, source):
            candidates.append((source, target))

    return sorted(set(candidates))


def _make_noise_map(
    u_variables,
    series_names: Sequence[str],
    std_by_series: Dict[str, float],
    mean_by_series: Dict[str, float],
    seed: int,
):
    seed_rng = np.random.default_rng(seed)
    noise_map = {}

    for u in sorted(u_variables, key=lambda z: (z.name, z.time)):
        sigma = NORMAL_NOISE_STD
        mean = NOISE_MEAN

        for series in series_names:
            if u.name == f"U_{series}":
                sigma = std_by_series[series]
                mean = mean_by_series[series]
                break

        local_rng = np.random.default_rng(
            int(seed_rng.integers(0, 2**32 - 1))
        )

        def draw(size=None, *, _rng=local_rng, _sigma=sigma, _mean=mean):
            return _rng.normal(_mean, _sigma, size=size)

        noise_map[u] = draw

    return noise_map


def _normalize_ts_dataframe(
    data: pd.DataFrame,
    series_names: Sequence[str],
) -> pd.DataFrame:
    data = data.copy()

    if "time" not in data.columns:
        if data.index.name == "time":
            data = data.reset_index()
        else:
            data.insert(0, "time", np.arange(len(data), dtype=int))

    missing = set(series_names) - set(data.columns)
    if missing:
        raise ValueError(
            f"Generated dataframe is missing columns: {sorted(missing)}"
        )

    data = data.loc[:, ["time", *series_names]].copy()
    data.index = np.arange(len(data))
    return data


def _generate_random_noise_problem(base_graph_seed: int):
    """
    Generate a fully random stationary FT-DAG and random linear dynamic SCM.

    The induced SCG is not constrained to be acyclic, so it may contain macro
    feedback cycles produced by temporally ordered FT-DAG relations.
    """
    for attempt in range(MAX_GRAPH_ATTEMPTS):
        graph_seed = base_graph_seed + attempt
        scm_seed = graph_seed + 1

        ftdag = create_random_ft_dag(
            num_ts=N_SERIES,
            p_edge=P_EDGE,
            causally_stationary=True,
            max_delay=TRUE_MAX_LAG,
            allow_instantaneous=True,
            seed=graph_seed,
        )
        # Adding self loops in every time series
        for node_i in ftdag.get_vertices():
            for node_j in ftdag.get_vertices():
                if node_i.name == node_j.name:
                    if node_i.time < node_j.time:
                        ftdag.add_directed_edge(node_i, node_j)

        scg = ftdag.get_summary_causal_graph()

        if REQUIRE_CYCLIC_SCG and not _scg_has_nontrivial_cycle(scg):
            continue

        candidate_pairs = _candidate_noise_root_pairs(scg)
        if not candidate_pairs:
            continue

        series_names = sorted(scg.get_vertices(), key=str)

        template_scm, coefficients, intercepts = (
            create_random_linear_dt_dynamic_scm_from_ftdag(
                ftdag=ftdag,
                weight_low=WEIGHT_LOW,
                weight_high=WEIGHT_HIGH,
                intercept_low=INTERCEPT_LOW,
                intercept_high=INTERCEPT_HIGH,
                u_dist=None,
                seed=scm_seed,
            )
        )

        return (
            ftdag,
            scg,
            series_names,
            template_scm,
            coefficients,
            intercepts,
            candidate_pairs,
            graph_seed,
            scm_seed,
        )

    raise RuntimeError(
        "Could not generate a random FT-DAG with a valid "
        "lineage-root -> noise-root pair after "
        f"{MAX_GRAPH_ATTEMPTS} attempts."
    )


# ============================================================================
# Piecewise-linear data generation
# ============================================================================


def simulate_two_noise_changes_from_ftdag(
    *,
    scg,
    series_names: Sequence[str],
    template_scm,
    coefficients,
    intercepts,
    candidate_pairs: Sequence[Tuple[str, str]],
    normal_size: int,
    max_anomaly_size: int,
    burn_in: int,
    seed: int,
):
    """
    Plant two innovation changes while keeping the general random FT-DAG and
    all structural coefficients fixed.

    * lineage_root has no cross-series parent and is handled by SGRCA as
      lineage-defying;
    * tested_noise_root is a child of lineage_root and is the non-lineage
      noise-defying root evaluated in this experiment.
    """
    if not candidate_pairs:
        raise ValueError("candidate_pairs must not be empty.")

    rng = np.random.default_rng(seed)
    lineage_root, tested_noise_root = candidate_pairs[
        int(rng.integers(len(candidate_pairs)))
    ]

    normal_std = {
        name: NORMAL_NOISE_STD
        for name in series_names
    }
    normal_mean = {
        name: NOISE_MEAN
        for name in series_names
    }

    anomalous_std = dict(normal_std)
    anomalous_mean = dict(normal_mean)

    if LINEAGE_NOISE_CHANGE_MEAN_OR_VAR == 0:
        anomalous_std[lineage_root] *= ANOMALOUS_NOISE_MULTIPLIER
    else:
        anomalous_mean[lineage_root] += ANOMALOUS_NOISE_MULTIPLIER

    # Y always receives a variance change, as in the previous benchmark.
    anomalous_std[tested_noise_root] *= ANOMALOUS_NOISE_MULTIPLIER

    normal_scm = LinearDtDynamicSCM(
        v=template_scm.v,
        u=template_scm.u,
        coefficients=coefficients,
        intercepts=intercepts,
        u_dist=_make_noise_map(
            template_scm.u,
            series_names,
            normal_std,
            normal_mean,
            seed + 1,
        ),
    )

    anomalous_scm = LinearDtDynamicSCM(
        v=template_scm.v,
        u=template_scm.u,
        coefficients=coefficients,
        intercepts=intercepts,
        u_dist=_make_noise_map(
            template_scm.u,
            series_names,
            anomalous_std,
            anomalous_mean,
            seed + 2,
        ),
    )

    normal_data = normal_scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=normal_size,
        burn_in=burn_in,
        include_latent=False,
        time_series_format="ts",
        seed=seed + 3,
    )

    anomalous_data = anomalous_scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=max_anomaly_size,
        burn_in=burn_in,
        include_latent=False,
        time_series_format="ts",
        seed=seed + 4,
    )

    normal_data = _normalize_ts_dataframe(normal_data, series_names)
    anomalous_data = _normalize_ts_dataframe(anomalous_data, series_names)

    normal_data["time"] = np.arange(normal_size, dtype=int)
    anomalous_data["time"] = np.arange(
        normal_size,
        normal_size + max_anomaly_size,
        dtype=int,
    )

    full_data = pd.concat(
        [normal_data, anomalous_data],
        ignore_index=True,
    )

    anomaly_start = normal_size

    all_noise_root_causes = {
        lineage_root,
        tested_noise_root,
    }
    evaluated_noise_root_causes = {
        tested_noise_root,
    }

    parameter_info = {
        "lineage_noise_root": lineage_root,
        "tested_noise_root": tested_noise_root,
        "normal_noise_std": NORMAL_NOISE_STD,
        "anomalous_noise_std": anomalous_std[tested_noise_root],
        "scg_is_cyclic": _scg_has_nontrivial_cycle(scg),
        "n_nontrivial_scg_cycles": _n_nontrivial_scg_cycles(scg),
    }

    return (
        full_data,
        anomaly_start,
        all_noise_root_causes,
        evaluated_noise_root_causes,
        parameter_info,
    )


# ============================================================================
# SGRCA+ adapter
# ============================================================================


def run_sgrca_plus(
    sgrca_plus_cls,
    *,
    scg,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
    algorithm_max_lag: int,
    sig_threshold: float,
) -> Set[str]:
    model = sgrca_plus_cls(
        g=scg,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=algorithm_max_lag,
        sig_threshold=sig_threshold,
    )

    root_causes = model.run(data)

    prediction: Set[str] = set()
    for component_result in root_causes.values():
        prediction.update(component_result.get("effect_defying", []))
        prediction.update(component_result.get("noise_defying", []))

    return prediction


# ============================================================================
# Experiment
# ============================================================================


def run_experiment() -> pd.DataFrame:
    algorithms = {
        "SGRCA": SGRCA,
        "SGRCA+": SGRCAPlus,
        "SGRCA*": SGRCAStar,
        "EasyRCA": EasyRCA,
        "EasyRCA*": EasyRCAStar,
        "FT-DAG-RCA": FTDAGRCA,
        "FT-DAG-RCA*": FTDAGRCAStar,
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
        base_graph_seed = BASE_SEED + 10_000 * repetition

        (
            ftdag,
            scg,
            series_names,
            template_scm,
            coefficients,
            intercepts,
            candidate_pairs,
            graph_seed,
            scm_seed,
        ) = _generate_random_noise_problem(base_graph_seed)

        data_seed = graph_seed + 2

        (
            full_data,
            anomaly_start,
            all_noise_root_causes,
            evaluated_noise_root_causes,
            parameter_info,
        ) = simulate_two_noise_changes_from_ftdag(
            scg=scg,
            series_names=series_names,
            template_scm=template_scm,
            coefficients=coefficients,
            intercepts=intercepts,
            candidate_pairs=candidate_pairs,
            normal_size=NORMAL_SIZE,
            max_anomaly_size=max(ANOMALY_SIZES),
            burn_in=BURN_IN,
            seed=data_seed,
        )

        lineage_noise_root = parameter_info["lineage_noise_root"]
        tested_noise_root = parameter_info["tested_noise_root"]

        assert not _cross_parents(scg, lineage_noise_root)
        assert lineage_noise_root in _cross_parents(scg, tested_noise_root)
        assert (lineage_noise_root, tested_noise_root) in set(candidate_pairs)

        anomalous_vertices = (
            set(scg.get_descendants(lineage_noise_root))
            | set(scg.get_descendants(tested_noise_root))
            | {lineage_noise_root, tested_noise_root}
        )

        macro_cross_edges = _macro_cross_edges(scg)

        for anomaly_size in ANOMALY_SIZES:
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

                    elif algorithm_name == "SGRCA+":
                        prediction = run_sgrca_plus(
                            algorithms["SGRCA+"],
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
                        prediction = [
                            node
                            for node in prediction
                            if node != lineage_noise_root
                        ]
                        if algorithm_name == "RCD-1":
                            prediction = set(prediction[:1])
                        else:
                            prediction = set(prediction)

                    elif algorithm_name == "MICROCause":
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
                        prediction = [
                            node
                            for node in prediction
                            if node != lineage_noise_root
                        ]
                        prediction = set(prediction[:1])

                    else:
                        raise ValueError(
                            f"Unknown algorithm: {algorithm_name}"
                        )

                    metrics = root_cause_metrics(
                        evaluated_noise_root_causes,
                        prediction,
                    )
                    status = "ok"

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
                        "scm_seed": scm_seed,
                        "data_seed": data_seed,
                        "algorithm": algorithm_name,
                        "anomaly_size": anomaly_size,
                        "normal_size": NORMAL_SIZE,
                        "true_max_lag": TRUE_MAX_LAG,
                        "algorithm_max_lag": ALGORITHM_MAX_LAG,
                        "n_series": N_SERIES,
                        "n_cross_edges": len(macro_cross_edges),
                        "n_ft_directed_edges": len(ftdag.get_directed_edges()),
                        "n_scg_edges": len(scg.get_directed_edges()),
                        "scg_is_cyclic": parameter_info["scg_is_cyclic"],
                        "n_nontrivial_scg_cycles": parameter_info[
                            "n_nontrivial_scg_cycles"
                        ],
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
            f"| cyclic SCG: {parameter_info['scg_is_cyclic']} "
            f"| lineage noise root: {lineage_noise_root} "
            f"| tested noise root: {tested_noise_root}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================


def main():
    warnings.filterwarnings("ignore")

    if ALGORITHM_MAX_LAG < TRUE_MAX_LAG:
        warnings.warn(
            "ALGORITHM_MAX_LAG is smaller than TRUE_MAX_LAG. Some true "
            "temporal direct effects are outside the algorithms' search window."
        )

    print("Noise-defying anomaly-size simulation: random FT-DAGs")
    print(f"Anomaly sizes:             {ANOMALY_SIZES}")
    print(f"Repetitions:               {N_REPETITIONS}")
    print(f"Number of series:          {N_SERIES}")
    print(f"Normal sample size:        {NORMAL_SIZE}")
    print(f"True max lag:              {TRUE_MAX_LAG}")
    print(f"Algorithm max lag:         {ALGORITHM_MAX_LAG}")
    print(f"Require cyclic SCG:        {REQUIRE_CYCLIC_SCG}")
    print(f"Noise multiplier:          {ANOMALOUS_NOISE_MULTIPLIER}")
    print(f"Algorithms:                {METHODS}")
    print("Evaluation target:         non-lineage noise-defying root only")

    raw_results = run_experiment()
    summary = summarize_results(raw_results)

    print("\n" + "=" * 88)
    print("RESULTS BY ANOMALY SIZE")
    print("F1 evaluates ONLY the non-lineage noise-defying root cause.")
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

    graph_rows = raw_results.drop_duplicates("repetition")
    cyclic_rate = float(graph_rows["scg_is_cyclic"].mean())
    print(f"\nFraction of generated SCGs that are cyclic: {cyclic_rate:.3f}")


if __name__ == "__main__":
    main()
