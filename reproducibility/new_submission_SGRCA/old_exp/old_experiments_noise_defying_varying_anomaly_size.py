from __future__ import annotations

import contextlib
import io
import random
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from pyciphod.root_cause_analysis.graph_based import SGRCA
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag
from pyciphod.utils.scms.dynamic_scm import (
    LinearDtDynamicSCM,
    create_random_linear_dt_dynamic_scm_from_ftadmg,
)
from pyciphod.utils.time_series.data_format import DTimeVar

from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc


# ============================================================================
# Hyperparameters
# ============================================================================

ANOMALY_SIZES = [30, 60, 125, 250, 500, 1000, 2000]

N_REPETITIONS = 20
N_SERIES = 5
P_EDGE = 0.35
BASE_SEED = 42

TRUE_MAX_LAG = 1
ALGORITHM_MAX_LAG = 1

BURN_IN = 500
NORMAL_SIZE = 10 * max(ANOMALY_SIZES)

WEIGHT_LOW = 0.20
WEIGHT_HIGH = 0.50
INTERCEPT_LOW = 0.0
INTERCEPT_HIGH = 0.0

NORMAL_NOISE_STD = 1.0
ANOMALOUS_NOISE_MULTIPLIER = 3.0

SIG_THRESHOLD = 0.05

RCD_BINS = 5
RCD_CHUNK_SIZE = 5
RCD_LOCALIZED = False

METHODS = ("SGRCA", "RCD", "MICROCause")
SUPPRESS_BASELINE_OUTPUT = True

OUTPUT_DIR = Path("results_noise_defying_varying_anomaly_size")


# ============================================================================
# FtDAG generation
# ============================================================================


def _macro_cross_edges(ftdag) -> Set[Tuple[str, str]]:
    return {
        (src.name, tgt.name)
        for src, tgt in ftdag.get_directed_edges()
        if src.name != tgt.name
    }


def _times_by_series(ftdag) -> Dict[str, Set[int]]:
    out = {}
    for node in ftdag.get_vertices():
        out.setdefault(node.name, set()).add(node.time)
    return out


def _force_first_order_self_dependence(ftdag) -> None:
    vertices = set(ftdag.get_vertices())
    times_by_series = _times_by_series(ftdag)

    for name, times in times_by_series.items():
        for t in sorted(times):
            src = DTimeVar(name, t - 1)
            tgt = DTimeVar(name, t)
            if src in vertices and tgt in vertices:
                if (src, tgt) not in ftdag.get_directed_edges():
                    ftdag.add_directed_edge(src, tgt)


def _force_instantaneous_realization_of_each_scg_edge(ftdag) -> None:
    """If X at any lag causes Y, also force X_t -> Y_t."""
    vertices = set(ftdag.get_vertices())
    times_by_series = _times_by_series(ftdag)

    for source, target in sorted(_macro_cross_edges(ftdag)):
        common_times = times_by_series[source] & times_by_series[target]

        for t in sorted(common_times):
            src = DTimeVar(source, t)
            tgt = DTimeVar(target, t)
            if src in vertices and tgt in vertices:
                if (src, tgt) not in ftdag.get_directed_edges():
                    ftdag.add_directed_edge(src, tgt)


def _cross_parents(scg, node) -> Set[str]:
    return set(scg.get_parents(node)) - {node}


def _candidate_noise_root_pairs(scg) -> list[Tuple[str, str]]:
    """
    Candidate X -> Y pairs such that X is lineage-defying and Y is not.
    """
    candidates = []

    for source, target in scg.get_directed_edges():
        if source == target:
            continue

        if not _cross_parents(scg, source):
            candidates.append((source, target))

    return candidates


def generate_random_ftdag_for_noise_experiment(
    num_series: int,
    p_edge: float,
    seed: int,
    max_attempts: int = 200,
):
    if TRUE_MAX_LAG != 1:
        raise ValueError("This experiment assumes TRUE_MAX_LAG == 1.")

    for attempt in range(max_attempts):
        actual_seed = seed + attempt

        try:
            ftdag = create_random_ft_dag(
                num_ts=num_series,
                p_edge=p_edge,
                causally_stationary=True,
                max_delay=TRUE_MAX_LAG,
                allow_instantaneous=True,
                seed=actual_seed,
            )

            _force_first_order_self_dependence(ftdag)
            _force_instantaneous_realization_of_each_scg_edge(ftdag)

        except ValueError:
            # Forcing all macro-relations to be instantaneous can reveal an
            # instantaneous cycle. In that case regenerate another FtDAG.
            continue

        scg = ftdag.get_summary_causal_graph()
        candidates = _candidate_noise_root_pairs(scg)

        if candidates:
            return ftdag, scg, candidates, actual_seed

    raise RuntimeError(
        "Could not generate a valid FtDAG after "
        f"{max_attempts} attempts."
    )


# ============================================================================
# Dynamic SCM and time-series generation
# ============================================================================


def _make_noise_map(
    u_variables,
    series_names: Sequence[str],
    std_by_series: Dict[str, float],
    seed: int,
):
    """Create node-specific exogenous distributions for LinearDtDynamicSCM."""
    seed_rng = np.random.default_rng(seed)
    noise_map = {}

    for u in sorted(u_variables, key=lambda z: (z.name, z.time)):
        sigma = NORMAL_NOISE_STD

        for series in series_names:
            if u.name == f"U_{series}":
                sigma = std_by_series[series]
                break

        local_rng = np.random.default_rng(
            int(seed_rng.integers(0, 2**32 - 1))
        )

        def draw(size=None, *, _rng=local_rng, _sigma=sigma):
            return _rng.normal(0.0, _sigma, size=size)

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


def simulate_two_noise_changes_from_ftdag(
    ftdag,
    scg,
    candidate_pairs: Sequence[Tuple[str, str]],
    normal_size: int,
    max_anomaly_size: int,
    burn_in: int,
    seed: int,
):
    """
    Change exactly two noise mechanisms during the anomalous regime.

    Select X -> Y where X has no cross-series parent.

    * X: noise changes, but SGRCA handles X as lineage-defying.
    * Y: noise changes and Y is not lineage-defying, so this is the
         noise-defying root cause evaluated in this experiment.

    Structural coefficients do not change.
    """
    rng = np.random.default_rng(seed)

    lineage_root, tested_noise_root = candidate_pairs[
        int(rng.integers(len(candidate_pairs)))
    ]

    series_names = sorted(scg.get_vertices(), key=str)

    template_scm, coefficients, intercepts = (
        create_random_linear_dt_dynamic_scm_from_ftadmg(
            ftadmg=ftdag,
            weight_low=WEIGHT_LOW,
            weight_high=WEIGHT_HIGH,
            intercept_low=INTERCEPT_LOW,
            intercept_high=INTERCEPT_HIGH,
            u_dist=None,
            causally_stationary=True,
            seed=seed + 1,
        )
    )

    normal_std = {name: NORMAL_NOISE_STD for name in series_names}
    anomalous_std = dict(normal_std)

    anomalous_std[lineage_root] *= ANOMALOUS_NOISE_MULTIPLIER
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
            seed + 2,
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
            seed + 3,
        ),
    )

    normal_data = normal_scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=normal_size,
        burn_in=burn_in,
        include_latent=False,
        time_series_format="ts",
        seed=seed + 4,
    )

    anomalous_data = anomalous_scm.generate_causally_stationary_time_series_from_latest_mechanisms(
        n_timepoints=max_anomaly_size,
        burn_in=burn_in,
        include_latent=False,
        time_series_format="ts",
        seed=seed + 5,
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

    all_noise_root_causes = {lineage_root, tested_noise_root}

    # IMPORTANT: this simulation focuses only on the noise-defying root that
    # survives SGRCA's earlier lineage-defying stage.
    evaluated_noise_root_causes = {tested_noise_root}

    parameter_info = {
        "lineage_noise_root": lineage_root,
        "tested_noise_root": tested_noise_root,
        "normal_noise_std": NORMAL_NOISE_STD,
        "anomalous_noise_std": (
            NORMAL_NOISE_STD * ANOMALOUS_NOISE_MULTIPLIER
        ),
    }

    return (
        full_data,
        anomaly_start,
        all_noise_root_causes,
        evaluated_noise_root_causes,
        parameter_info,
    )


# ============================================================================
# Algorithm adapters
# ============================================================================


def run_sgrca(
    scg,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomalous_vertices: Set[str],
    anomaly_start: int,
    anomaly_size: int,
) -> Set[str]:
    """Return only SGRCA's noise-defying predictions."""
    sgrca = SGRCA(
        g=scg,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=anomaly_start,
        anomaly_length=anomaly_size,
        max_delay=ALGORITHM_MAX_LAG,
        sig_threshold=SIG_THRESHOLD,
    )

    root_causes = sgrca.run(data)

    prediction = set()
    for component_result in root_causes.values():
        print(component_result)
        prediction.update(component_result.get("effect_defying", []))
        prediction.update(component_result.get("noise_defying", []))

    return prediction


def run_rcd(
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    k: int,
    seed: int,
) -> Set[str]:
    normal_df = data.loc[
        data["time"] < anomaly_start,
        list(series_names),
    ].copy()

    anomalous_df = data.loc[
        (data["time"] >= anomaly_start)
        & (data["time"] < anomaly_start + anomaly_size),
        list(series_names),
    ].copy()

    np.random.seed(seed)

    result = top_k_rc(
        normal_df,
        anomalous_df,
        k=k,
        bins=RCD_BINS,
        gamma=RCD_CHUNK_SIZE,
        localized=RCD_LOCALIZED,
        verbose=False,
    )

    return set(result["root_cause"])


def run_microcause(
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomaly_start: int,
    anomaly_size: int,
    k: int,
    seed: int,
) -> Set[str]:
    ts_data = data.loc[:, list(series_names)].copy()

    anomalies_start_time = {
        name: anomaly_start
        for name in series_names
    }

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

    return set(list(rc_list)[:k])


# ============================================================================
# Metrics
# ============================================================================


def root_cause_metrics(
    truth: Iterable[str],
    prediction: Iterable[str],
) -> Dict[str, float]:
    truth = set(truth)
    prediction = set(prediction)

    tp = len(truth & prediction)
    fp = len(prediction - truth)
    fn = len(truth - prediction)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
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


# ============================================================================
# Experiment
# ============================================================================


def run_experiment() -> pd.DataFrame:
    rows = []

    if NORMAL_SIZE < 10 * max(ANOMALY_SIZES):
        warnings.warn(
            "NORMAL_SIZE is smaller than 10 * max(ANOMALY_SIZES)."
        )

    for repetition in range(N_REPETITIONS):
        requested_graph_seed = BASE_SEED + 10_000 * repetition
        data_seed = requested_graph_seed + 1

        ftdag, scg, candidate_pairs, graph_seed = (
            generate_random_ftdag_for_noise_experiment(
                num_series=N_SERIES,
                p_edge=P_EDGE,
                seed=requested_graph_seed,
            )
        )

        series_names = sorted(scg.get_vertices(), key=str)

        (
            full_data,
            anomaly_start,
            all_noise_root_causes,
            evaluated_noise_root_causes,
            parameter_info,
        ) = simulate_two_noise_changes_from_ftdag(
            ftdag=ftdag,
            scg=scg,
            candidate_pairs=candidate_pairs,
            normal_size=NORMAL_SIZE,
            max_anomaly_size=max(ANOMALY_SIZES),
            burn_in=BURN_IN,
            seed=data_seed,
        )

        lineage_noise_root = parameter_info["lineage_noise_root"]
        tested_noise_root = parameter_info["tested_noise_root"]

        # Sanity checks defining the experiment.
        assert not _cross_parents(scg, lineage_noise_root)
        assert _cross_parents(scg, tested_noise_root)

        # Nodes structurally affected by the two noise changes.
        anomalous_vertices = (
                set(scg.get_descendants(lineage_noise_root))
                | set(scg.get_descendants(tested_noise_root))
                | {lineage_noise_root, tested_noise_root}
        )

        for anomaly_size in ANOMALY_SIZES:
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
                            scg,
                            data,
                            series_names,
                            anomalous_vertices,
                            anomaly_start,
                            anomaly_size,
                        )

                    elif algorithm_name == "RCD":
                        prediction = run_rcd(
                            data,
                            series_names,
                            anomaly_start,
                            anomaly_size,
                            k=len(evaluated_noise_root_causes),
                            seed=method_seed,
                        )

                    elif algorithm_name == "MICROCause":
                        prediction = run_microcause(
                            data,
                            series_names,
                            anomaly_start,
                            anomaly_size,
                            k=len(evaluated_noise_root_causes),
                            seed=method_seed,
                        )

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


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    valid = raw_results.loc[raw_results["status"] == "ok"].copy()

    return (
        valid
        .groupby(["algorithm", "anomaly_size"], as_index=False)
        .agg(
            n_runs=("f1", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            runtime_mean=("runtime_seconds", "mean"),
            runtime_std=("runtime_seconds", "std"),
        )
    )


def summarize_overall(raw_results: pd.DataFrame) -> pd.DataFrame:
    valid = raw_results.loc[raw_results["status"] == "ok"].copy()

    return (
        valid
        .groupby("algorithm")
        .agg(
            n_runs=("f1", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            runtime_mean=("runtime_seconds", "mean"),
            runtime_std=("runtime_seconds", "std"),
            total_runtime=("runtime_seconds", "sum"),
        )
        .reset_index()
    )


# ============================================================================
# Main
# ============================================================================


def main():
    warnings.filterwarnings("ignore")

    if ALGORITHM_MAX_LAG < 0:
        raise ValueError("ALGORITHM_MAX_LAG must be non-negative.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Noise-defying anomaly-size simulation")
    print(f"Anomaly sizes:             {ANOMALY_SIZES}")
    print(f"Repetitions:               {N_REPETITIONS}")
    print(f"Number of series:          {N_SERIES}")
    print(f"Normal sample size:        {NORMAL_SIZE}")
    print(f"True max lag:              {TRUE_MAX_LAG}")
    print(f"Algorithm max lag:         {ALGORITHM_MAX_LAG}")
    print(f"Noise multiplier:          {ANOMALOUS_NOISE_MULTIPLIER}")
    print(f"Algorithms:                {METHODS}")
    print("Evaluation target:         non-lineage noise-defying root only")

    raw_results = run_experiment()
    summary = summarize_results(raw_results)
    overall = summarize_overall(raw_results)

    raw_results.to_csv(OUTPUT_DIR / "raw_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_by_anomaly_size.csv", index=False)
    overall.to_csv(OUTPUT_DIR / "summary_overall.csv", index=False)

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
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n" + "=" * 88)
    print("OVERALL RESULTS")
    print("=" * 88)
    print(
        overall[
            [
                "algorithm",
                "f1_mean",
                "f1_std",
                "runtime_mean",
                "runtime_std",
                "total_runtime",
                "n_runs",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
