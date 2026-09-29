from __future__ import annotations

import contextlib
import io
import random
import time
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np
import pandas as pd

from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph

from pyciphod.root_cause_analysis.graph_based import (
    SGRCA,
    EasyRCA,
    SGRCAPlus
)
from pyciphod.root_cause_analysis.data_based import (
    SGRCAStar,
    EasyRCAStar,
    FTDAGRCAStar,
)

from baseline.microcause import micro_cause
from baseline.rcd import top_k_rc
from baseline.circa import CIRCA

import matplotlib.pyplot as plt

# Reuse the adapters that are already used by the simulation files for the
# fully data-driven starred methods.
from common import (
    run_sgrca_star,
    run_easyrca_star,
    run_ftdagrca_star,
)


# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = "../../datasets/nephrology_data.csv"

ANOMALY_START = 44
ANOMALY_END = 54

# ANOMALY_START = 34
# ANOMALY_END = 43


# Inclusive interval [ANOMALY_START, ANOMALY_END].
ANOMALY_LENGTH = ANOMALY_END - ANOMALY_START + 1
NORMAL_WINDOW_SIZE = 10 * ANOMALY_LENGTH

ALGORITHM_MAX_LAG = 1
SIG_THRESHOLD = 0.05
# SIG_THRESHOLD = 0.01
# SIG_THRESHOLD = 0.000001
# SIG_THRESHOLD = 0.0000001
SEED = 42

# RCD hyperparameters used in the simulation experiments.
RCD_BINS = 5
RCD_CHUNK_SIZE = 5
RCD_LOCALIZED = False

# The real-data input below gives an SCG, not a true FT-DAG.

# Suppress verbose internal output from baselines and print only the final
# root-cause summary.
SUPPRESS_BASELINE_OUTPUT = True


# ============================================================================
# Graph
# ============================================================================


def build_summary_causal_graph(list_nodes: Sequence[str]) -> SummaryCausalGraph:
    """Build the supplied monitoring-system SCG."""
    gt = SummaryCausalGraph()
    gt.add_vertices(list_nodes)

    directed_edges = [
        ("Tacrolimus", "Creatinine"),
        ("Tacrolimus", "Leukocytes"),
        ("Tacrolimus", "Platelets_THROMBOCYTES"), # new
        # ("CRP", "Creatinine"),
        # ("CRP", "Leukocytes"),
        # ("Protein", "Creatinine"),
        ("Chloride", "Creatinine"),
        ("Chloride", "Sodium"),
        # ("Urinary protein", "Creatinine"),
        # ("Urinary protein", "Protein"),
        ("Hematocrit", "Hemoglobin"),
        # ("Platelets_THROMBOCYTES", "Creatinine"),
        ("Sodium", "Creatinine"),
        ("Sodium", "Chloride"),
        ("Creatinine", "Calcium"),
        ("Creatinine", "Phosphorus"),
        ("Creatinine", "Urea"),
        ("Creatinine", "Bicarbonate"),
        ("Creatinine", "Hemoglobin"),
        ("Hemoglobin", "Hematocrit"),
        ("Calcium", "Phosphorus"),
        ("Urea", "Creatinine"),
        ("Urea", "Bicarbonate"),
        ("Phosphorus", "Calcium"),
        ("Bicarbonate", "Potassium"),
    ]

    # directed_edges = [
    #     # Immunosuppression / drug exposure
    #     ("Tacrolimus", "Creatinine"),
    #     ("Tacrolimus", "Leukocytes"),
    #     ("Tacrolimus", "Platelets_THROMBOCYTES"),
    #
    #     # Renal dysfunction proxy -> downstream manifestations
    #     ("Creatinine", "Urea"),
    #     ("Creatinine", "Potassium"),
    #     ("Creatinine", "Bicarbonate"),
    #     ("Creatinine", "Phosphorus"),
    #     ("Creatinine", "Calcium"),
    #     ("Creatinine", "Hemoglobin"),
    #
    #     # Blood-count relation
    #     ("Hemoglobin", "Hematocrit"),
    #
    #     # Additional
    #     ("Urea", "Creatinine"),
    #     ("Sodium", "Creatinine"),
    #     ("Chloride", "Creatinine"),
    #     ("Sodium", "Chloride"),
    #     ("Chloride", "Sodium"),
    # ]

    missing_nodes = {
        node
        for edge in directed_edges
        for node in edge
        if node not in set(list_nodes)
    }
    if missing_nodes:
        raise ValueError(
            "The following nodes used by the supplied graph are absent from "
            f"the CSV columns: {sorted(missing_nodes)}"
        )

    for source, target in directed_edges:
        gt.add_directed_edge(source, target)

    # Same self-loop assumption as in the supplied starter code.
    for node in list_nodes:
        gt.add_directed_edge(node, node)

    return gt

# ============================================================================
# Data preparation
# ============================================================================


def load_real_data() -> tuple[pd.DataFrame, List[str]]:
        """
        Load the monitoring data and keep only:
        - the most recent NORMAL_WINDOW_SIZE normal points
        - the anomalous interval
        """

        data = pd.read_csv(DATA_PATH,sep=";",header=0,index_col=0,decimal=",")
        data = data.transpose()
        data.drop(columns=["Urinary creatinine", "Spot urine microalbumin"], inplace=True)
        data.drop(columns=["CRP"], inplace=True)
        data.drop(columns=["Protein"], inplace=True)
        data.drop(columns=["Urinary protein"], inplace=True)
        # data.drop(columns=["Platelets_THROMBOCYTES"], inplace=True)

        list_nodes = list(data.columns)

        data.index.name = "time"
        data.index = pd.to_numeric(data.index)
        data = data.apply(pd.to_numeric, errors="coerce").astype(float)
        print(data)


        ########## plotting the data
        # data_to_plot = data.loc[:ANOMALY_END]
        # axes = data_to_plot.plot(
        #     subplots=True,
        #     figsize=(12, 8),
        #     sharex=True,
        #     legend=False
        # )
        # for ax in axes:
        #     ax.axvline(x=ANOMALY_START, linestyle="--", linewidth=2,  color="red")
        # for ax in axes:
        #     ax.set_ylabel("")
        #
        # axes[-1].set_xlabel("Time")
        #
        # plt.tight_layout()
        # plt.show()
        ##########


        if not (data.index < ANOMALY_START).any():
                raise ValueError("No normal observations exist before ANOMALY_START.")
        observed_anomaly = data.loc[(data.index >= ANOMALY_START)  & (data.index <= ANOMALY_END)]
        if observed_anomaly.empty:
                raise ValueError("The requested anomalous interval contains no observations.")

        normal_start = ANOMALY_START - NORMAL_WINDOW_SIZE

        data = data.loc[
        (data.index >= normal_start)
        & (data.index <= ANOMALY_END)
        ].copy()

        print(f"Normal window starts at: {normal_start}")
        print(f"Data used: {data.index.min()} -> {data.index.max()}")
        print(f"Number of rows used: {len(data)}")
        return data, list_nodes


# ============================================================================
# Result helpers
# ============================================================================


ROOT_CAUSE_TYPES = (
    "lineage_defying",
    "time_defying",
    "effect_defying",
    "noise_defying",
)


def flatten_graph_based_root_causes(root_causes: Any) -> Set[str]:
    """
    Union all SGRCA-family root-cause categories.

    The real-data comparison intentionally does not distinguish lineage,
    time, effect, and noise root causes.
    """
    prediction: Set[str] = set()

    if root_causes is None:
        return prediction

    if isinstance(root_causes, dict):
        for component_result in root_causes.values():
            if isinstance(component_result, dict):
                for root_type in ROOT_CAUSE_TYPES:
                    prediction.update(component_result.get(root_type, []))
            elif isinstance(component_result, (set, list, tuple)):
                prediction.update(component_result)
        return prediction

    if isinstance(root_causes, (set, list, tuple)):
        prediction.update(root_causes)
        return prediction

    raise TypeError(
        "Unexpected root-cause output type: "
        f"{type(root_causes).__name__}"
    )


def normalize_prediction(prediction: Any) -> Set[str]:
    """Normalize a generic adapter/baseline output to a set of node names."""
    if prediction is None:
        return set()
    if isinstance(prediction, set):
        return prediction
    if isinstance(prediction, (list, tuple)):
        return set(prediction)
    if isinstance(prediction, dict):
        return flatten_graph_based_root_causes(prediction)
    return {str(prediction)}


# ============================================================================
# Algorithm runners
# ============================================================================


def run_graph_based_model(
    model_cls,
    *,
    scg: SummaryCausalGraph,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
) -> tuple[Set[str], Any]:
    """Run SGRCA, or EasyRCA and union every root-cause type."""
    model = model_cls(
        g=scg,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=ANOMALY_START,
        anomaly_length=ANOMALY_LENGTH,
        max_delay=ALGORITHM_MAX_LAG,
        sig_threshold=SIG_THRESHOLD,
    )

    raw_result = model.run(data)
    print(raw_result)
    print("Possible root causes: ", model.possible_root_causes)
    prediction = flatten_graph_based_root_causes(raw_result)
    return prediction, model


def run_data_based_model(
    model_cls,
    *,
    data: pd.DataFrame,
    anomalous_vertices: Set[str],
) -> tuple[Set[str], Any]:
    """Run SGRCA, or EasyRCA and union every root-cause type."""
    model = model_cls(
        data = data,
        anomalous_vertices=list(anomalous_vertices),
        anomalies_start_time=ANOMALY_START,
        anomaly_length=ANOMALY_LENGTH,
        max_delay=ALGORITHM_MAX_LAG,
        sig_threshold=SIG_THRESHOLD,
    )

    raw_result = model.run(data)
    print(raw_result)
    prediction = flatten_graph_based_root_causes(raw_result)
    return prediction, model

def run_rcd_real(
    *,
    data: pd.DataFrame,
    series_names: Sequence[str],
    k: int,
) -> List[str]:
    """Run RCD and preserve its returned ranking."""
    normal_df = data.loc[
        data.index < ANOMALY_START,
        list(series_names),
    ].copy()

    anomalous_df = data.loc[
        (data.index >= ANOMALY_START)
        & (data.index < ANOMALY_START + ANOMALY_LENGTH),
        list(series_names),
    ].copy()

    np.random.seed(SEED)

    result = top_k_rc(
        normal_df,
        anomalous_df,
        k=k,
        bins=RCD_BINS,
        gamma=RCD_CHUNK_SIZE,
        localized=RCD_LOCALIZED,
        verbose=False,
    )
    print(result["root_cause"])
    return list(result["root_cause"])


def run_microcause_real(
    *,
    data: pd.DataFrame,
    series_names: Sequence[str],
    anomalous_vertices: Set[str],
) -> List[str]:
    """Run MICROCause and keep every root cause returned by the method."""
    ts_data = data.loc[:, list(series_names)].copy()

    # anomalies_start_time = {
    #     name: ANOMALY_START
    #     for name in series_names
    # }
    anomalies_start_time = {
        name: ANOMALY_START
        for name in anomalous_vertices
    }

    random.seed(SEED)
    np.random.seed(SEED)

    kwargs = dict(
        data=ts_data,
        anomalous_nodes=list(anomalous_vertices),
        anomalies_start_time=anomalies_start_time,
        anomaly_length=ANOMALY_LENGTH,
        gamma_max=ALGORITHM_MAX_LAG,
        sig_threshold=SIG_THRESHOLD,
    )

    if SUPPRESS_BASELINE_OUTPUT:
        with contextlib.redirect_stdout(io.StringIO()):
            rc_list = micro_cause(**kwargs)
    else:
        rc_list = micro_cause(**kwargs)
    print(rc_list)
    return list(rc_list)

def run_circa_real(
    *,
    scg: SummaryCausalGraph,
    data: pd.DataFrame,
    series_names: Sequence[str],
    descendant_adjustment: bool = True,
) -> List[str]:
    """
    Run CIRCA with the supplied SCG and return the complete ranking.
    """

    # CIRCA should only receive the actual time-series columns.
    ts_data = data.loc[:, list(series_names)].copy()

    # IMPORTANT:
    # CIRCA's anomaly_start is a ROW POSITION in the implementation,
    # whereas ANOMALY_START here is the original time/index value.
    anomaly_start_pos = int(
        np.flatnonzero(ts_data.index >= ANOMALY_START)[0]
    )

    circa = CIRCA(
        parent_max_lag=0,
        include_self_lag=True,
        self_max_lag=ALGORITHM_MAX_LAG,
        descendant_adjustment=descendant_adjustment,
    )

    circa.fit(
        graph=scg,
        data=ts_data,
        anomaly_start=anomaly_start_pos,
        anomaly_size=ANOMALY_LENGTH,
    )

    ranking = circa.rank()

    print(ranking)

    return ranking


# ============================================================================
# Main comparison
# ============================================================================


def main() -> None:
    data, list_nodes = load_real_data()
    scg = build_summary_causal_graph(list_nodes)

    # The starter code assigns the same anomaly onset to every graph variable,
    # so every observed series is treated as anomalous.
    anomalous_vertices = set(list_nodes)
    anomalous_vertices.remove("Tacrolimus")
    # anomalous_vertices = set(["Creatinine", "Urea", "Hemoglobin", "Phosphorus", "Potassium", "Bicarbonate"])
    anomalies_start_time = {node: ANOMALY_START for node in anomalous_vertices}
    print(anomalies_start_time)
    # Kept explicitly because it is useful if you later replace the common
    # onset by variable-specific anomaly onsets.
    assert set(anomalies_start_time) == anomalous_vertices

    print("=" * 100)
    print("REAL-DATA ROOT CAUSE LOCALIZATION")
    print("=" * 100)
    print(f"Dataset:          {DATA_PATH}")
    print(f"Variables:        {len(list_nodes)}")
    print(f"Anomaly interval: [{ANOMALY_START}, {ANOMALY_END}]")
    print(f"Anomaly length:   {ANOMALY_LENGTH}")
    print(f"Maximum lag:      {ALGORITHM_MAX_LAG}")
    print(f"Significance:     {SIG_THRESHOLD}")
    print()
    print(
        "For SGRCA-family methods, the printed prediction is the UNION of "
        "lineage-, time-, effect-, and noise-defying root causes."
    )
    print()

    results: Dict[str, Dict[str, Any]] = {}

    def execute(name: str, function) -> None:
            print(f"\nRunning {name} ...", flush=True)

            start = time.perf_counter()

            try:
                    prediction = function()
                    runtime = time.perf_counter() - start

                    if isinstance(prediction, list):
                        stored_prediction = prediction
                    else:
                        stored_prediction = normalize_prediction(prediction)

                    results[name] = {
                        "status": "ok",
                        "root_causes": stored_prediction,
                        "runtime": runtime,
                        "error": "",
                    }

                    print(
                            f"Finished {name} in {runtime:.2f} seconds",
                            flush=True,
                    )

            except Exception as exc:
                    runtime = time.perf_counter() - start

                    results[name] = {
                            "status": "error",
                            "root_causes": set(),
                            "runtime": runtime,
                            "error": repr(exc),
                    }

                    print(
                            f"{name} failed after {runtime:.2f} seconds: {exc}",
                            flush=True,
                    )
    # ------------------------------------------------------------------------
    # Graph-given methods
    # ------------------------------------------------------------------------

    execute(
        "SGRCA",
        lambda: run_graph_based_model(
            SGRCA,
            scg=scg,
            data=data,
            anomalous_vertices=anomalous_vertices,
        )[0],
    )

    execute(
        "EasyRCA",
        lambda: run_graph_based_model(
            EasyRCA,
            scg=scg,
            data=data,
            anomalous_vertices=anomalous_vertices,
        )[0],
    )

    execute(
        "CIRCA",
        lambda: run_circa_real(
            scg=scg,
            data=data,
            series_names=list_nodes,
            descendant_adjustment=False,
        ),
    )


    # ------------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------------

    execute(
        "RCD-1",
        lambda: run_rcd_real(
            data=data,
            series_names=list_nodes,
            k=1,
        ),
    )

    # There is no known number of true root causes in real data, so RCD-all
    # asks for up to all observed variables and prints the complete returned
    # ranking/set instead of choosing an arbitrary ground-truth k.
    execute(
        "RCD-all",
        lambda: run_rcd_real(
            data=data,
            series_names=list_nodes,
            k=len(list_nodes),
        ),
    )

    execute(
        "MICROCause",
        lambda: run_microcause_real(
            data=data,
            series_names=list_nodes,
            anomalous_vertices=anomalous_vertices,
        ),
    )

    execute(
        "FTDAGRCAStar",
        lambda: run_data_based_model(
            FTDAGRCAStar,
            data=data,
            anomalous_vertices=anomalous_vertices,
        )[0],
    )
    execute(
        "SGRCAPlus",
        lambda: run_graph_based_model(
            SGRCAPlus,
            scg=scg,
            data=data,
            anomalous_vertices=anomalous_vertices,
        )[0],
    )

    # ------------------------------------------------------------------------
    # Fully data-driven starred methods
    # ------------------------------------------------------------------------
    execute(
        "SGRCAStar",
        lambda: run_data_based_model(
            SGRCAStar,
            data=data,
            anomalous_vertices=anomalous_vertices,
        )[0],
    )
    # execute(
    #     "SGRCA*",
    #     lambda: run_sgrca_star(
    #         SGRCAStar,
    #         data=data,
    #         anomalous_vertices=anomalous_vertices,
    #         anomaly_start=ANOMALY_START,
    #         anomaly_size=ANOMALY_LENGTH,
    #         algorithm_max_lag=ALGORITHM_MAX_LAG,
    #         sig_threshold=SIG_THRESHOLD,
    #     ),
    # )

    # execute(
    #     "EasyRCAStar",
    #     lambda: run_data_based_model(
    #         EasyRCAStar,
    #         data=data,
    #         anomalous_vertices=anomalous_vertices,
    #     )[0],
    # )

    # execute(
    #     "EasyRCA*",
    #     lambda: run_easyrca_star(
    #         EasyRCAStar,
    #         data=data,
    #         anomalous_vertices=anomalous_vertices,
    #         anomaly_start=ANOMALY_START,
    #         anomaly_size=ANOMALY_LENGTH,
    #         algorithm_max_lag=ALGORITHM_MAX_LAG,
    #         sig_threshold=SIG_THRESHOLD,
    #     ),
    # )


    # execute(
    #     "FT-DAG-RCA*",
    #     lambda: run_ftdagrca_star(
    #         FTDAGRCAStar,
    #         data=data,
    #         anomalous_vertices=anomalous_vertices,
    #         anomaly_start=ANOMALY_START,
    #         anomaly_size=ANOMALY_LENGTH,
    #         algorithm_max_lag=ALGORITHM_MAX_LAG,
    #         sig_threshold=SIG_THRESHOLD,
    #     ),
    # )



    # ------------------------------------------------------------------------
    # Clean output
    # ------------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("ROOT CAUSES DETECTED BY EACH ALGORITHM")
    print("=" * 100)

    order = (
        "SGRCA",
        "SGRCAPlus",
        "SGRCAStar",
        # "EasyRCA",
        # "EasyRCAStar",
        "FTDAGRCAStar",
        "RCD-1",
        "RCD-all",
        "MICROCause",
    )

    for name in order:
        result = results[name]
        status = result["status"]

        if status == "ok":
            roots = sorted(result["root_causes"])
            print(f"{name:<12}: {roots}")
            print(f"{'':<12}  runtime = {result['runtime']:.4f} s")

        elif status == "skipped":
            print(f"{name:<12}: SKIPPED")
            print(f"{'':<12}  {result['error']}")

        else:
            print(f"{name:<12}: ERROR")
            print(f"{'':<12}  {result['error']}")

    print("=" * 100)


if __name__ == "__main__":
    main()
