from __future__ import annotations

import time
import warnings
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag

from pyciphod.root_cause_analysis.graph_based import SGRCA, FTDAGRCA, EasyRCA
from pyciphod.root_cause_analysis.data_based import (
    SGRCAStar,
    EasyRCAStar,
    FTDAGRCAStar,
)
from pyciphod.root_cause_analysis.graph_based import SGRCAPlus

from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc
from baseline.circa import CIRCA

from common import (
    run_sgrca,
    run_sgrca_star,
    run_easyrca,
    run_easyrca_star,
    run_ftdagrca,
    run_ftdagrca_star,
    run_rcd,
    run_microcause,
    run_circa,
    sample_signed_coefficient,
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

# Same structural coefficient ranges as in the no-cycle benchmark.
SELF_WEIGHT_LOW = 0.20
SELF_WEIGHT_HIGH = 0.50

CROSS_WEIGHT_ABS_LOW = 0.40
CROSS_WEIGHT_ABS_HIGH = 0.90
MAX_COEFFICIENT_ATTEMPTS = 500

NOISE_MEAN = 0.0
NORMAL_NOISE_STD = 1.0
LINEAGE_NOISE_CHANGE_MEAN_OR_VAR = 0  # 0: variance, 1: mean
ANOMALOUS_NOISE_MULTIPLIER = 3.0
LINEAGE_EVAL = True


SIG_THRESHOLD = 0.05

METHODS = (
    "SGRCA",
    "SGRCA+",
    "SGRCA*",
    "EasyRCA",
    "EasyRCA*",
    "FT-DAG-RCA",
    "FT-DAG-RCA*",
    "CIRCA",
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


def _temporal_relations(ftdag) -> List[Tuple[str, str, int]]:
    """Return the unique stationary relations (source, target, lag)."""
    relations = set()

    for source, target in ftdag.get_directed_edges():
        gamma = target.time - source.time
        if gamma < 0:
            raise ValueError(
                f"Invalid future-to-past FT edge: {source!r} -> {target!r}."
            )
        relations.add((source.name, target.name, gamma))

    return sorted(relations)


def _instantaneous_topological_order(
    series_names: Sequence[str],
    temporal_relations: Sequence[Tuple[str, str, int]],
) -> List[str]:
    """Topological order of the lag-0 part of the FT-DAG."""
    children = {name: set() for name in series_names}
    indegree = {name: 0 for name in series_names}

    for source, target, gamma in temporal_relations:
        if gamma != 0:
            continue

        if source == target:
            raise ValueError("The FT-DAG contains an instantaneous self-loop.")

        if target not in children[source]:
            children[source].add(target)
            indegree[target] += 1

    available = sorted(
        name for name in series_names
        if indegree[name] == 0
    )
    order: List[str] = []

    while available:
        source = available.pop(0)
        order.append(source)

        for target in sorted(children[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
                available.sort()

    if len(order) != len(series_names):
        raise ValueError("The instantaneous part of the FT-DAG is cyclic.")

    return order


def _stability_radius(
    series_names: Sequence[str],
    temporal_relations: Sequence[Tuple[str, str, int]],
    coefficients: Dict[Tuple[str, str, int], float],
) -> float:
    """Spectral radius of the reduced VAR representation."""
    n_series = len(series_names)
    index_of = {name: i for i, name in enumerate(series_names)}

    matrices = [
        np.zeros((n_series, n_series), dtype=float)
        for _ in range(TRUE_MAX_LAG + 1)
    ]

    for source, target, gamma in temporal_relations:
        if gamma > TRUE_MAX_LAG:
            raise ValueError(
                f"FT relation {(source, target, gamma)!r} exceeds TRUE_MAX_LAG."
            )

        matrices[gamma][
            index_of[target],
            index_of[source],
        ] = coefficients[(source, target, gamma)]

    instantaneous_inverse = np.linalg.inv(
        np.eye(n_series) - matrices[0]
    )

    reduced_lag_matrices = [
        instantaneous_inverse @ matrices[gamma]
        for gamma in range(1, TRUE_MAX_LAG + 1)
    ]

    if not reduced_lag_matrices:
        return 0.0

    companion = np.zeros(
        (n_series * TRUE_MAX_LAG, n_series * TRUE_MAX_LAG),
        dtype=float,
    )
    companion[:n_series, :] = np.hstack(reduced_lag_matrices)

    if TRUE_MAX_LAG > 1:
        companion[n_series:, :-n_series] = np.eye(
            n_series * (TRUE_MAX_LAG - 1)
        )

    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _sample_stable_coefficients(
    rng: np.random.Generator,
    series_names: Sequence[str],
    temporal_relations: Sequence[Tuple[str, str, int]],
):
    """Sample coefficients as in the no-cycle benchmark and keep a stable SCM."""
    for _ in range(MAX_COEFFICIENT_ATTEMPTS):
        coefficients: Dict[Tuple[str, str, int], float] = {}

        for source, target, gamma in temporal_relations:
            relation = (source, target, gamma)

            if source == target:
                coefficients[relation] = float(
                    rng.uniform(SELF_WEIGHT_LOW, SELF_WEIGHT_HIGH)
                )
            else:
                coefficients[relation] = sample_signed_coefficient(
                    rng,
                    CROSS_WEIGHT_ABS_LOW,
                    CROSS_WEIGHT_ABS_HIGH,
                )

        stability_radius = _stability_radius(
            series_names,
            temporal_relations,
            coefficients,
        )

        if stability_radius < 1.0:
            return coefficients, stability_radius

    raise RuntimeError(
        "Could not sample stable coefficients for the generated FT-DAG after "
        f"{MAX_COEFFICIENT_ATTEMPTS} attempts."
    )


def _generate_random_noise_problem(base_graph_seed: int):
    """
    Generate a fully random stationary FT-DAG.

    The induced SCG is not constrained to be acyclic, so it may contain macro
    feedback cycles produced by temporally ordered FT-DAG relations.
    """
    for attempt in range(MAX_GRAPH_ATTEMPTS):
        graph_seed = base_graph_seed + attempt

        ftdag = create_random_ft_dag(
            num_ts=N_SERIES,
            p_edge=P_EDGE,
            causally_stationary=True,
            max_delay=TRUE_MAX_LAG,
            allow_instantaneous=True,
            seed=graph_seed,
        )

        # Adding lag-1 self loops in every time series.
        vertices = set(ftdag.get_vertices())
        directed_edges = set(ftdag.get_directed_edges())

        for node_i in vertices:
            for node_j in vertices:
                if node_i.name == node_j.name:
                    if node_j.time - node_i.time == 1:
                        if (node_i, node_j) not in directed_edges:
                            ftdag.add_directed_edge(node_i, node_j)
                            directed_edges.add((node_i, node_j))

        scg = ftdag.get_summary_causal_graph()

        if REQUIRE_CYCLIC_SCG and not _scg_has_nontrivial_cycle(scg):
            continue

        candidate_pairs = _candidate_noise_root_pairs(scg)
        if not candidate_pairs:
            continue

        series_names = sorted(scg.get_vertices(), key=str)
        temporal_relations = _temporal_relations(ftdag)
        topological_order = _instantaneous_topological_order(
            series_names,
            temporal_relations,
        )

        return (
            ftdag,
            scg,
            series_names,
            temporal_relations,
            topological_order,
            candidate_pairs,
            graph_seed,
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
    temporal_relations: Sequence[Tuple[str, str, int]],
    topological_order: Sequence[str],
    candidate_pairs: Sequence[Tuple[str, str]],
    normal_size: int,
    max_anomaly_size: int,
    burn_in: int,
    seed: int,
):
    """
    Simulate one continuous trajectory with two temporary noise changes.

    For each series V,

        V_t = sum_{(U,V,gamma)} beta_{UV,gamma} U_{t-gamma}
              + N^V_t.

    The FT-DAG and all structural coefficients stay unchanged throughout the
    simulation. Coefficients are sampled with the same ranges as in the
    no-cycle benchmark.

    * lineage_root has no cross-series parent and is handled by SGRCA as
      lineage-defying;
    * tested_noise_root is a child of lineage_root and is the non-lineage
      noise-defying root evaluated in this experiment.

    The anomalous regime is generated on the same trajectory as the normal
    regime, exactly as in the no-cycle benchmark.
    """
    if not candidate_pairs:
        raise ValueError("candidate_pairs must not be empty.")

    rng = np.random.default_rng(seed)
    lineage_root, tested_noise_root = candidate_pairs[
        int(rng.integers(len(candidate_pairs)))
    ]

    coefficients, stability_radius = _sample_stable_coefficients(
        rng,
        series_names,
        temporal_relations,
    )

    n_series = len(series_names)
    index_of = {name: i for i, name in enumerate(series_names)}

    parents: Dict[str, List[Tuple[str, int]]] = {
        name: []
        for name in series_names
    }
    for source, target, gamma in temporal_relations:
        parents[target].append((source, gamma))

    n_retained = normal_size + max_anomaly_size
    total_length = burn_in + n_retained
    anomaly_start_absolute = burn_in + normal_size

    values = np.zeros((total_length, n_series), dtype=float)

    for t in range(total_length):
        anomalous = t >= anomaly_start_absolute

        for target in topological_order:
            target_idx = index_of[target]
            value = 0.0

            for source, gamma in parents[target]:
                source_time = t - gamma
                if source_time < 0:
                    continue

                value += (
                    coefficients[(source, target, gamma)]
                    * values[source_time, index_of[source]]
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

    full_data = pd.DataFrame(retained, columns=series_names)
    full_data.insert(0, "time", np.arange(n_retained, dtype=int))

    anomaly_start = normal_size

    all_noise_root_causes = {
        lineage_root,
        tested_noise_root,
    }

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
        "coefficients": coefficients,
        "stability_radius": stability_radius,
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
        base_graph_seed = BASE_SEED + 10_000 * repetition

        (
            ftdag,
            scg,
            series_names,
            temporal_relations,
            topological_order,
            candidate_pairs,
            graph_seed,
        ) = _generate_random_noise_problem(base_graph_seed)

        data_seed = graph_seed + 1

        (
            full_data,
            anomaly_start,
            all_noise_root_causes,
            evaluated_noise_root_causes,
            parameter_info,
        ) = simulate_two_noise_changes_from_ftdag(
            scg=scg,
            series_names=series_names,
            temporal_relations=temporal_relations,
            topological_order=topological_order,
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
                        # On cyclic SCGs EasyRCA is outside its applicability
                        # domain and may raise. The benchmark convention below
                        # counts such a failure as an empty prediction / F1 = 0.
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
                            prediction = [
                                node
                                for node in prediction
                                if node != lineage_noise_root
                            ]

                        if algorithm_name == "RCD-1":
                            if LINEAGE_EVAL:
                                prediction = set(prediction[:2])
                            else:
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
                        if not LINEAGE_EVAL:
                            prediction = [
                                node
                                for node in prediction
                                if node != lineage_noise_root
                            ]
                        if algorithm_name == "MICROCause-1":
                            if LINEAGE_EVAL:
                                prediction = set(prediction[:2])
                            else:
                                prediction = set(prediction[:1])
                        else:
                            prediction = set(prediction)

                    elif algorithm_name == "CIRCA":
                        prediction = run_circa(
                            algorithms["CIRCA"],
                            scg=scg,
                            data=data,
                            series_names=series_names,
                            anomaly_start=anomaly_start,
                            anomaly_size=anomaly_size,
                            n_true_root_causes=N_SERIES,
                            algorithm_max_lag=ALGORITHM_MAX_LAG,
                            descendant_adjustment=False,
                        )

                        if not LINEAGE_EVAL:
                            prediction = [
                                node
                                for node in prediction
                                if node != lineage_noise_root
                            ]

                        if LINEAGE_EVAL:
                            prediction = set(prediction[:2])
                        else:
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
