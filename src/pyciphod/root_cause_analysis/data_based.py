from __future__ import annotations

from typing import Optional

import pandas as pd

from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC

from pyciphod.utils.graphs.temporal_graphs import (
    FtDirectedAcyclicGraph,
)

from pyciphod.utils.time_series.data_format import (
    DTimeVar,
    ts_to_lagged_df,
)

from pyciphod.root_cause_analysis.graph_based import (
    SGRCA,
    FTDAGRCA,
    EasyRCA,
)


# =============================================================================
# Causal discovery
# =============================================================================


def learn_ftcpdag(
    *,
    data: pd.DataFrame,
    anomalies_start_time,
    max_delay: int,
    sparsity: float = 0.05,
    max_sepset_size: Optional[int] = None,
):
    """
    Learn an FT-CPDAG from the normal regime using TsPC.

    Only observations strictly before the beginning of the anomalous
    regime are used for causal discovery.
    """

    if isinstance(anomalies_start_time, dict):
        normal_end = min(
            anomalies_start_time.values()
        )
    else:
        normal_end = anomalies_start_time

    # -------------------------------------------------------------------------
    # Extract the normal regime.
    # -------------------------------------------------------------------------

    if "time" in data.columns:

        normal_data = data.loc[data["time"] < normal_end].drop(columns=["time"]).copy()

    elif data.index.name == "time":
        normal_data = data.loc[
            data.index < normal_end
        ].copy()

    else:
        raise ValueError(
            "Data must contain a 'time' column or "
            "have an index named 'time'."
        )

    if normal_data.empty:
        raise ValueError(
            "The normal regime contains no observations."
        )

    # -------------------------------------------------------------------------
    # Construct the lag-augmented dataframe required by TsPC.
    # -------------------------------------------------------------------------

    lagged_normal_data = ts_to_lagged_df(
        normal_data,
        window_size=max_delay + 1,
        dropna=True,
    )

    # -------------------------------------------------------------------------
    # Temporal PC.
    # -------------------------------------------------------------------------

    tspc = TsPC(sparsity=sparsity,)

    ftcpdag = tspc.run(data=lagged_normal_data,max_sepset_size=max_sepset_size,)

    return ftcpdag


# =============================================================================
# FT-CPDAG -> SCG
# =============================================================================


def ftcpdag_to_scg(
    ftcpdag,
):
    """
    Convert an FT-CPDAG into a partially specified SCG.

    Directed temporal edges are first copied into an FT-DAG and projected
    to an SCG using the existing `get_summary_causal_graph()` method.

    Remaining undirected temporal edges are represented by uncertain
    macro-level edges in the SCG.

    Thus,

        X_s -> Y_t

    produces

        X -> Y,

    whereas

        X_t - Y_t

    produces an uncertain SCG adjacency

        X o-o Y.

    An undirected CPDAG edge is NOT interpreted as latent confounding.
    """

    # -------------------------------------------------------------------------
    # Directed part of the learned FT-CPDAG.
    # -------------------------------------------------------------------------

    directed_ftdag = FtDirectedAcyclicGraph()

    for node in ftcpdag.get_vertices():
        directed_ftdag.add_vertex(
            node
        )

    for source, target in (
        ftcpdag.get_directed_edges()
    ):
        directed_ftdag.add_directed_edge(
            source,
            target,
        )

    # Use the existing FT-DAG -> SCG conversion.
    scg = (
        directed_ftdag
        .get_summary_causal_graph()
    )

    # -------------------------------------------------------------------------
    # Preserve unresolved orientations as uncertain SCG edges.
    # -------------------------------------------------------------------------

    added_uncertain_edges = set()

    for x, y in (
        ftcpdag.get_undirected_edges()
    ):

        source = x.name
        target = y.name

        # An undirected edge between two realizations of the same
        # series should normally have been oriented by temporal order.
        if source == target:
            continue

        key = frozenset(
            {source, target}
        )

        if key in added_uncertain_edges:
            continue
        scg.add_directed_edge(source, target)
        scg.add_directed_edge(target,source)

        # SCG do not support uncertain edges, so we add both directed edges to represent the uncertainty.
        # scg.add_uncertain_edge(
        #     source,
        #     target,
        #     edge_type="o-o",
        # )

        added_uncertain_edges.add(
            key
        )

    return scg


# =============================================================================
# SGRCA*
# =============================================================================


class SGRCAStar(SGRCA):
    """
    SGRCA*.

    Unlike SGRCA, the SCG is not given to the algorithm.
    An FT-CPDAG is first learned from normal-regime data with TsPC and
    then projected to a partially specified SCG.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        anomalous_vertices,
        anomalies_start_time,
        anomaly_length=200,
        max_delay=1,
        sig_threshold=0.05,
        discovery_sparsity=0.05,
        max_sepset_size=None,
    ):

        # ---------------------------------------------------------------------
        # Learn the causal structure from normal data.
        # ---------------------------------------------------------------------

        self.ftcpdag = learn_ftcpdag(
            data=data,
            anomalies_start_time=(
                anomalies_start_time
            ),
            max_delay=max_delay,
            sparsity=discovery_sparsity,
            max_sepset_size=(
                max_sepset_size
            ),
        )

        self.learned_scg = (
            ftcpdag_to_scg(
                self.ftcpdag
            )
        )

        # ---------------------------------------------------------------------
        # Run ordinary SGRCA on the learned SCG.
        # ---------------------------------------------------------------------

        super().__init__(
            g=self.learned_scg,
            anomalous_vertices=(
                anomalous_vertices
            ),
            anomalies_start_time=(
                anomalies_start_time
            ),
            anomaly_length=(
                anomaly_length
            ),
            max_delay=max_delay,
            sig_threshold=sig_threshold,
        )


# =============================================================================
# EasyRCA*
# =============================================================================


class EasyRCAStar(EasyRCA):
    """
    EasyRCA*.

    The SCG is learned from the normal regime using TsPC rather than being
    supplied to EasyRCA.

    EasyRCA itself retains its usual restriction that the learned SCG must
    be acyclic except possibly for self-loops.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        anomalous_vertices,
        anomalies_start_time,
        anomaly_length=200,
        max_delay=1,
        sig_threshold=0.05,
        discovery_sparsity=0.05,
        max_sepset_size=None,
    ):

        # ---------------------------------------------------------------------
        # Learn the causal structure from normal data.
        # ---------------------------------------------------------------------

        self.ftcpdag = learn_ftcpdag(
            data=data,
            anomalies_start_time=(
                anomalies_start_time
            ),
            max_delay=max_delay,
            sparsity=discovery_sparsity,
            max_sepset_size=(
                max_sepset_size
            ),
        )

        self.learned_scg = (
            ftcpdag_to_scg(
                self.ftcpdag
            )
        )

        # ---------------------------------------------------------------------
        # Run ordinary EasyRCA on the learned SCG.
        #
        # EasyRCA.__init__ performs the acyclicity check.
        # ---------------------------------------------------------------------

        super().__init__(
            g=self.learned_scg,
            anomalous_vertices=(
                anomalous_vertices
            ),
            anomalies_start_time=(
                anomalies_start_time
            ),
            anomaly_length=(
                anomaly_length
            ),
            max_delay=max_delay,
            sig_threshold=sig_threshold,
        )


# =============================================================================
# FT-DAG-RCA*
# =============================================================================


class FTDAGRCAStar(FTDAGRCA):
    """
    FT-DAG-RCA*.

    The temporal graph is learned from normal data using TsPC.

    Unlike oracle FT-DAG-RCA, TsPC may leave some contemporaneous
    adjacencies unoriented. Therefore, for each Y_t:

        * if every edge incident to Y_t is directed, the parent set of Y_t
          is known and FT-DAG-RCA can test its mechanism normally;

        * if at least one edge incident to Y_t remains undirected, the
          parent set of Y_t is not identified and Y is reported as a
          potential root cause.

    No arbitrary orientation of undirected TsPC edges is introduced.
    """

    def __init__(self,*,
        data: pd.DataFrame,
        anomalous_vertices,
        anomalies_start_time,
        anomaly_length=200,
        max_delay=1,
        sig_threshold=0.05,
        discovery_sparsity=0.05,
        max_sepset_size=None,
    ):

        # ---------------------------------------------------------------------
        # Learn the FT-CPDAG from the normal regime.
        # ---------------------------------------------------------------------

        self.ftcpdag = learn_ftcpdag(data=data,anomalies_start_time=anomalies_start_time,
            max_delay=max_delay, sparsity=discovery_sparsity,max_sepset_size=max_sepset_size)

        # ---------------------------------------------------------------------
        # Keep only the directed portion as an FT-DAG.
        #
        # For variables whose parent sets are identifiable, this graph
        # contains exactly the directed parents that FT-DAG-RCA should use.
        # ---------------------------------------------------------------------

        self.ftdag = (self._get_directed_part())

        # ---------------------------------------------------------------------
        # The macro-level graph still preserves unresolved orientations.
        #
        # This is used for anomalous connected components and the graphical
        # stages inherited from SGRCA.
        # ---------------------------------------------------------------------

        self.scg = ftcpdag_to_scg(self.ftcpdag)

        # ---------------------------------------------------------------------
        # We intentionally initialize the SGRCA part directly rather than
        # calling FTDAGRCA.__init__, because FTDAGRCA.__init__ would construct
        # its SCG only from the directed FT-DAG and would therefore lose the
        # uncertain edges learned by TsPC.
        # ---------------------------------------------------------------------

        SGRCA.__init__(self,
            g=self.scg,
            anomalous_vertices=anomalous_vertices,
            anomalies_start_time=anomalies_start_time,
            anomaly_length=anomaly_length,
            max_delay=max_delay,
            sig_threshold=sig_threshold,
        )

        # Used by the inherited FT-DAG-RCA implementation.
        self._exact_parent_cache = {}

        self._validate_ftdag_delay()

        # Additional output category specific to FT-DAG-RCA*.
        for component_result in (
            self.root_causes.values()
        ):
            component_result[
                "potential_root_causes"
            ] = []


    # =========================================================================
    # Directed part of the FT-CPDAG
    # =========================================================================

    def _get_directed_part(self,):
        """
        Return the directed portion of the learned FT-CPDAG as an FT-DAG.
        """

        ftdag = (
            FtDirectedAcyclicGraph()
        )

        for node in (
            self.ftcpdag.get_vertices()
        ):
            ftdag.add_vertex(
                node
            )

        for source, target in (
            self.ftcpdag
            .get_directed_edges()
        ):
            ftdag.add_directed_edge(
                source,
                target,
            )

        return ftdag


    # =========================================================================
    # Parent-set identifiability
    # =========================================================================

    def _mechanism_is_identifiable(self,v,):
        """
        Return True iff the complete parent set of V_t is identified.

        In the FT-CPDAG, the parent set of V_t is known exactly if every
        adjacency incident to V_t is oriented.

        If

            U_t - V_t

        remains undirected, U_t may or may not be a parent of V_t.
        Therefore the structural mechanism of V_t cannot be reconstructed
        uniquely.
        """

        time_v = DTimeVar(v,0,)

        if (time_v not in self.ftcpdag.get_vertices()):
            return False

        for x, y in (self.ftcpdag.get_undirected_edges()):

            if (x == time_v or y == time_v):
                return False
        return True


    # =========================================================================
    # Run
    # =========================================================================

    def run(self,data,):
        """
        Run FT-DAG-RCA*.

        Vertices whose local parent sets are not identified by TsPC are
        returned as potential root causes and are not subjected to the
        effect-defying or noise-defying tests.
        """

        # ---------------------------------------------------------------------
        # Keep the same first stages as SGRCA / FT-DAG-RCA.
        # ---------------------------------------------------------------------

        # self.get_lineage_defying()
        #
        # self.get_time_defying()

        # ---------------------------------------------------------------------
        # Determine which mechanisms are identifiable.
        #
        # A node touched by an undirected FT-CPDAG edge cannot be assigned
        # a definitive RCA category because its complete parent set is
        # unknown.
        # ---------------------------------------------------------------------

        temporarily_blocked = {}

        for (subgraph_id,subgraph,) in (self.maximally_connected_anomalous_components.items()):

            component_result = (
                self.root_causes[
                    subgraph_id
                ]
            )

            temporarily_blocked[
                subgraph_id
            ] = set()

            for v in (subgraph.get_vertices()):

                if (self._mechanism_is_identifiable(v)):
                    continue

                component_result[
                    "potential_root_causes"
                ].append(
                    v
                )

                # -------------------------------------------------------------
                # If an unresolved orientation touches V_t, even its
                # lineage/time classification should not be regarded as
                # definitive.
                # -------------------------------------------------------------

                component_result[
                    "lineage_defying"
                ] = [
                    node
                    for node in component_result[
                        "lineage_defying"
                    ]
                    if node != v
                ]

                component_result[
                    "time_defying"
                ] = [
                    node
                    for node in component_result[
                        "time_defying"
                    ]
                    if node != v
                ]

                # -------------------------------------------------------------
                # FTDAGRCA.get_effect_defying() and get_noise_defying()
                # already skip time-defying nodes.
                #
                # We therefore temporarily put V in that category so the
                # inherited implementations skip it without duplicating the
                # entire FT-DAG-RCA code.
                # -------------------------------------------------------------

                component_result[
                    "time_defying"
                ].append(
                    v
                )

                temporarily_blocked[
                    subgraph_id
                ].add(
                    v
                )

        # ---------------------------------------------------------------------
        # Reuse FT-DAG-RCA's effect and noise tests exactly.
        #
        # For every vertex reaching these stages, all parents of V_t are
        # directed and hence available through self.ftdag.
        # ---------------------------------------------------------------------

        self.get_effect_defying(
            data
        )

        self.get_noise_defying(
            data
        )

        # ---------------------------------------------------------------------
        # Remove the temporary marker. Potential vertices should appear only
        # in `potential_root_causes`, not in `time_defying`.
        # ---------------------------------------------------------------------

        for (
            subgraph_id,
            blocked_vertices,
        ) in temporarily_blocked.items():

            component_result = (
                self.root_causes[
                    subgraph_id
                ]
            )

            component_result[
                "time_defying"
            ] = [
                node
                for node in component_result[
                    "time_defying"
                ]
                if node
                not in blocked_vertices
            ]

        return self.root_causes