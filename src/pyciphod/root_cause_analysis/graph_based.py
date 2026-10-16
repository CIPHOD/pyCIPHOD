from __future__ import annotations

import pandas as pd
import numpy as np

from scipy.stats import energy_distance, permutation_test
from sklearn.linear_model import LinearRegression
from numbers import Number

from pyciphod.causal_reasoning.summary_causal_graph.micro_queries.direct_effect import CDE_is_identifiable, linear_direct_effect_is_identifiable
from pyciphod.utils.stat_tests.dependency_measures import LinearRegressionCoefficient
from pyciphod.utils.stat_tests.outlier_tests import grubb_test
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph


from pyciphod.utils.time_series.data_format import DTimeVar, ts_to_lagged_df, wide_timevar_to_lagged_df

from pyciphod.utils.scms.dynamic_scm import LinearDtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftdag
from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm_from_ftadmg as create_dtdscm
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag

from pyciphod.utils.graphs.temporal_graphs import FtDirectedAcyclicGraph

from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC


class SGRCA:
    def __init__(self, g: SummaryCausalGraph, anomalous_vertices, anomalies_start_time, anomaly_length=200, max_delay=1,
                 sig_threshold=0.05, easy_to_find_root_causes=False):
        self.g = g
        self.anomalous_vertices = anomalous_vertices
        self.possible_root_causes = set(anomalous_vertices)
        self.anomalies_start_time = anomalies_start_time
        self.anomaly_length = anomaly_length
        self.max_delay = max_delay
        self.sig_threshold = sig_threshold
        self.easy_to_find_root_causes = easy_to_find_root_causes

        # Convert scalar parameters to dictionaries
        self.anomalies_start_time = self._normalize_by_variable(
            anomalies_start_time,
            "anomalies_start_time"
        )
        self.anomaly_length = self._normalize_by_variable(
            anomaly_length,
            "anomaly_length"
        )
        # Global information about the anomalous episode
        self.regime_start_time = min(
            self.anomalies_start_time.values()
        )
        self.regime_end_time = max(
            self.anomalies_start_time[v] + self.anomaly_length[v]
            for v in self.anomalous_vertices
        )

        self.maximally_connected_anomalous_components = self._get_maximally_connected_anomalous_components()

        self.root_causes = dict()
        for subgraph_id in self.maximally_connected_anomalous_components.keys():
            self.root_causes[subgraph_id] = {"lineage_defying": [], "time_defying": [], "effect_defying": [],
                                             "noise_defying": []}

        self.estimated_direct_effects_in_anomalous_regime = dict()

    def _normalize_by_variable(self, value, name):
        """
        Convert a scalar or variable-specific dictionary into
        a dictionary indexed by anomalous variable.
        """

        if isinstance(value, Number):
            return {
                v: value
                for v in self.anomalous_vertices
            }

        if isinstance(value, dict):
            missing = set(self.anomalous_vertices) - set(value)

            if missing:
                raise ValueError(
                    f"{name} is missing values for variables: {missing}"
                )

            return {
                v: value[v]
                for v in self.anomalous_vertices
            }

        raise TypeError(
            f"{name} must be either a scalar or a dictionary."
        )

    def _get_maximally_connected_anomalous_components(self):
        # find maximally connected components in the anomalous subgraph
        maximally_connected_components = list()
        anomalous = set(self.anomalous_vertices)
        visited = set()

        for start in anomalous:
            if start in visited:
                continue

            component = set()
            stack = [start]

            while stack:
                v = stack.pop()
                if v in visited:
                    continue

                visited.add(v)
                component.add(v)

                for u in anomalous:
                    if u not in visited and self.g.is_adjacent(v, u):
                        stack.append(u)

            maximally_connected_components.append(component)

        # assign a unique id to each maximally connected component and return as a dict
        mcac_dict = dict()
        for i, component in enumerate(maximally_connected_components):
            g_anom = SummaryCausalGraph()
            for v in component:
                g_anom.add_vertex(v)
                for u in component:
                    if (v, u) in self.g.get_directed_edges():
                        g_anom.add_directed_edge(v, u)
                    if (v, u) in self.g.get_confounded_edges():
                        g_anom.add_confounded_edge(v, u)
                    if (v, u) in self.g.get_uncertain_edges():
                        g_anom.add_uncertain_edge(v, u)
            mcac_dict[i] = g_anom
        return mcac_dict

    def get_lineage_defying(self):
        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.get_vertices():
                if (not subgraph.get_parents(v)) or (subgraph.get_parents(v) == {v}):
                    self.root_causes[subgraph_id]["lineage_defying"].append(v)
                    self.possible_root_causes.discard(v)

    def get_time_defying(self):
        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.get_vertices():
                par = set(subgraph.get_parents(v))
                par.discard(v)
                test_passed = True
                for u in par:
                    if self.anomalies_start_time[v]>=self.anomalies_start_time[u]:
                        test_passed = False
                        break
                if test_passed and len(par) > 0:
                    self.root_causes[subgraph_id]["time_defying"].append(v)
                    self.possible_root_causes.discard(v)

    # def _split_data_by_regime(self, data):
    #     # TODO
    #     # split data into regime before anomaly and regime during anomaly
    #     data_before_anomaly = data[data["time"] < self.anomalies_start_time]
    #     data_during_anomaly = data[(data["time"] >= self.anomalies_start_time) & (data["time"] < self.anomalies_start_time + self.anomaly_length)]
    #     return data_before_anomaly, data_during_anomaly

    def _split_data_by_regime(self, data, v):
        start = self.anomalies_start_time[v]
        length = self.anomaly_length[v]

        if "time" in data.columns:
            time = data["time"]

        elif data.index.name == "time":
            time = data.index

        else:
            raise ValueError(
                "Data must contain a 'time' column or have an index named 'time'."
            )

        before = data[time < start]

        during = data[
            (time >= start)
            & (time < start + length)
            ]

        return before, during

    def _create_normal_batches(
            self,
            data_before_anomaly,
            data_during_anomaly,
            max_batches=10,
    ):
        """
        Split the normal regime into non-overlapping batches having the
        same number of observations as the anomalous regime.

        At most `max_batches` batches are returned.
        """

        batch_size = len(data_during_anomaly)

        if batch_size == 0:
            raise ValueError("The anomalous regime contains no observations.")

        n_possible = len(data_before_anomaly) // batch_size

        if n_possible == 0:
            raise ValueError(
                "The normal regime is shorter than the anomalous regime."
            )

        n_batches = min(max_batches, n_possible)

        # Use the most recent normal observations before the anomaly
        n_required = n_batches * batch_size

        normal_data = data_before_anomaly.iloc[-n_required:]

        batches = [
            normal_data.iloc[
            i * batch_size:(i + 1) * batch_size
            ]
            for i in range(n_batches)
        ]

        return batches

    def get_effect_defying(self, data):
        # data_before_anomaly, data_during_anomaly = self._split_data_by_regime(data)
        # normal_data_batchs = np.array_split(data_before_anomaly, 10)
        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.get_vertices():
                if v not in self.root_causes[subgraph_id]["lineage_defying"] and v not in self.root_causes[subgraph_id]["time_defying"]:
                    data_before_anomaly_v, data_during_anomaly_v = self._split_data_by_regime(data, v)
                    data_before_anomaly_v = ts_to_lagged_df(data_before_anomaly_v, window_size=self.max_delay + 1, dropna=True)
                    data_during_anomaly_v = ts_to_lagged_df(data_during_anomaly_v, window_size=self.max_delay + 1, dropna=True)

                    test_id_v_all = True

                    # data_before_anomaly_v_batchs = np.array_split(data_before_anomaly_v, 10)
                    data_before_anomaly_v_batchs = self._create_normal_batches(data_before_anomaly_v, data_during_anomaly_v, max_batches=10)
                    for u in self.g.get_parents(v):
                        for gamma in range(self.max_delay + 1):
                            if u == v and gamma == 0:
                                continue
                            # # TODO The next condition should be removed after updating the linear_direct_effect_is_identifiable function to handle the case when u == v and gamma > 0
                            # if u == v:
                            #     continue
                            test_id_uv, adj_set_list = linear_direct_effect_is_identifiable(self.g, u, v, gamma=gamma, max_delay=self.max_delay)
                            if test_id_uv:
                                # adj_set = adj_set_list["ferreira_assaad_2024_def12"]
                                adj_set = min(adj_set_list.values(),key=len,)
                                # adj_set = adj_set_list["assaad_ezzejjari_zan_2023_def11"]
                                time_v = DTimeVar(v, 0)
                                time_u = DTimeVar(u, -gamma)
                                lr = LinearRegressionCoefficient(time_u, time_v, cond_list=list(adj_set), drop_na=False)
                                dep_before_list = []
                                for batch in data_before_anomaly_v_batchs:
                                    dep_before = lr.get_dependence(batch)
                                    dep_before_list.append(dep_before)
                                # dep_before = lr.get_dependence(data_before_anomaly)
                                dep_during = lr.get_dependence(data_during_anomaly_v)
                                self.estimated_direct_effects_in_anomalous_regime[(u, v, gamma)] = dep_during
                                dep_list = [dep_during] + dep_before_list
                                grubb_test_res = grubb_test(dep_list, sig_threshold=self.sig_threshold)
                                if grubb_test_res["anomaly_position"] == 0:
                                    # TODO Optimize better the next condition should no be needed
                                    if v not in self.root_causes[subgraph_id]["effect_defying"]:
                                        self.root_causes[subgraph_id]["effect_defying"].append(v)
                                        self.possible_root_causes.discard(v)
                                    break
                            else:
                                test_id_v_all = False
                    if test_id_v_all:
                        self.possible_root_causes.discard(v)

    def get_noise_defying(self, data):
        # data_before_anomaly, data_during_anomaly = self._split_data_by_regime(data)

        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.get_vertices():
                if v not in self.root_causes[subgraph_id]["lineage_defying"] and v not in self.root_causes[subgraph_id][
                    "time_defying"] and v not in self.root_causes[subgraph_id]["effect_defying"]:
                    data_before_anomaly_v, data_during_anomaly_v = self._split_data_by_regime(data, v)
                    data_before_anomaly_v_batchs = self._create_normal_batches(data_before_anomaly_v,
                                                                               data_during_anomaly_v, max_batches=10)
                    data_during_anomaly_v = ts_to_lagged_df(data_during_anomaly_v, window_size=self.max_delay + 1,
                                                            dropna=True)

                    model_anomalous_regime = 0

                    test_mechanism_identifiable = True
                    for u in self.g.get_parents(v):
                        for gamma in range(self.max_delay + 1):
                            if u == v and gamma == 0:
                                continue
                            if (u, v, gamma) not in self.estimated_direct_effects_in_anomalous_regime.keys():
                                test_mechanism_identifiable = False

                    if test_mechanism_identifiable:
                        self.possible_root_causes.discard(v)
                        for u in self.g.get_parents(v):
                            for gamma in range(self.max_delay + 1):
                                if u == v and gamma == 0:
                                    continue

                                time_u = DTimeVar(u, -gamma)
                                beta = self.estimated_direct_effects_in_anomalous_regime[(u, v, gamma)]
                                model_anomalous_regime += beta * data_during_anomaly_v[time_u]

                        time_v = DTimeVar(v, 0)

                        residuals_anomalous = np.asarray(
                            data_during_anomaly_v[time_v] - model_anomalous_regime
                        ).ravel()

                        residuals_normal_list = []

                        for data_before_anomaly_v_batch in data_before_anomaly_v_batchs:
                            data_before_anomaly_v_batch = ts_to_lagged_df(data_before_anomaly_v_batch,window_size=self.max_delay + 1,dropna=True)

                            model_normal_regime = 0

                            for u in self.g.get_parents(v):
                                for gamma in range(self.max_delay + 1):
                                    if u == v and gamma == 0:
                                        continue

                                    time_u = DTimeVar(u, -gamma)
                                    beta = self.estimated_direct_effects_in_anomalous_regime[(u, v, gamma)]
                                    model_normal_regime += beta * data_before_anomaly_v_batch[time_u]

                            residuals_normal = np.asarray(
                                data_before_anomaly_v_batch[time_v] - model_normal_regime
                            ).ravel()

                            residuals_normal_list.append(residuals_normal)

                        normal_distances = []
                        for i in range(len(residuals_normal_list) - 1):
                            normal_distances.append(
                                energy_distance(
                                    residuals_normal_list[i],
                                    residuals_normal_list[i + 1]
                                )
                            )

                        anomaly_distance = energy_distance(
                            residuals_normal_list[-1],
                            residuals_anomalous
                        )

                        result = grubb_test(
                            [anomaly_distance] + normal_distances,
                            sig_threshold=self.sig_threshold
                        )

                        if result["anomaly_position"] == 0:
                            self.root_causes[subgraph_id]["noise_defying"].append(v)

    def get_noise_defying_with_pvalue(self, data):
        # data_before_anomaly, data_during_anomaly = self._split_data_by_regime(data)

        for subgraph_id, subgraph in self.maximally_connected_anomalous_components.items():
            for v in subgraph.get_vertices():
                if v not in self.root_causes[subgraph_id]["lineage_defying"] and v not in self.root_causes[subgraph_id]["time_defying"] and v not in self.root_causes[subgraph_id]["effect_defying"]:
                    data_before_anomaly_v, data_during_anomaly_v = self._split_data_by_regime(data, v)
                    data_before_anomaly_v_batchs = self._create_normal_batches(data_before_anomaly_v, data_during_anomaly_v, max_batches=10)
                    data_before_anomaly_v_batch = data_before_anomaly_v_batchs[-1]  # Use the most recent batch of normal data
                    data_during_anomaly_v = ts_to_lagged_df(data_during_anomaly_v, window_size=self.max_delay + 1, dropna=True)
                    data_before_anomaly_v_batch = ts_to_lagged_df(data_before_anomaly_v_batch, window_size=self.max_delay + 1, dropna=True)

                    model_anomalous_regime = 0
                    model_normal_regime = 0

                    test_mechanism_identifiable = True
                    for u in self.g.get_parents(v):
                        for gamma in range(self.max_delay + 1):
                            if u == v and gamma == 0:
                                continue
                            if (u, v, gamma) not  in self.estimated_direct_effects_in_anomalous_regime.keys():
                                test_mechanism_identifiable = False

                    if test_mechanism_identifiable:
                        self.possible_root_causes.discard(v)
                        for u in self.g.get_parents(v):
                            for gamma in range(self.max_delay + 1):
                                if u == v and gamma == 0:
                                    continue

                                time_u = DTimeVar(u, -gamma)
                                beta = self.estimated_direct_effects_in_anomalous_regime[(u, v, gamma)]
                                model_anomalous_regime += beta * data_during_anomaly_v[time_u]
                                model_normal_regime += beta * data_before_anomaly_v_batch[time_u]

                        time_v = DTimeVar(v, 0)

                        residuals_anomalous = np.asarray(
                            data_during_anomaly_v[time_v] - model_anomalous_regime
                        ).ravel()

                        residuals_normal = np.asarray(
                            data_before_anomaly_v_batch[time_v] - model_normal_regime
                        ).ravel()

                        result = permutation_test(
                            (residuals_normal, residuals_anomalous),
                            statistic=energy_distance,
                            permutation_type="independent",
                            alternative="greater",
                            n_resamples=1000,
                            random_state=42
                        )

                        if result.pvalue < self.sig_threshold:
                            self.root_causes[subgraph_id]["noise_defying"].append(v)

    def _reset_results(self):
        """
        Reset all state produced by a previous run.

        This makes an SGRCA instance safely reusable on another dataset
        with the same graph, anomalous variables, and configuration.
        """

        self.possible_root_causes = set(self.anomalous_vertices)

        self.root_causes = {
            subgraph_id: {
                "lineage_defying": [],
                "time_defying": [],
                "effect_defying": [],
                "noise_defying": [],
            }
            for subgraph_id in self.maximally_connected_anomalous_components
        }

        self.estimated_direct_effects_in_anomalous_regime = {}

    def run(self, data):
        self._reset_results()
        if self.easy_to_find_root_causes:
            self.get_lineage_defying()
            self.get_time_defying()
        self.get_effect_defying(data)
        self.get_noise_defying(data)
        return self.root_causes

class SGRCAPlus(SGRCA):
    """
    SGRCA+.

    SGRCA+ first runs exactly as SGRCA.

    If SGRCA leaves some variables in `self.possible_root_causes`
    because their local mechanism cannot be identified from the SCG,
    SGRCA+ learns an FT-CPDAG from the normal regime using TsPC.

    The supplied SCG is used as background knowledge during temporal
    causal discovery.

    For each remaining possible root cause Y:

        * if all edges incident to Y_t are oriented in the learned
          FT-CPDAG, Pa(Y_t) is identified and the local mechanism can
          be tested directly;

        * otherwise Y remains a possible root cause.

    When Pa(Y_t) is identified, SGRCA+ uses the same effect- and
    noise-defying tests as SGRCA. The only additional ingredient is
    temporal causal discovery for candidates unresolved from the SCG.
    """

    def __init__(self, g, anomalous_vertices, anomalies_start_time, anomaly_length=200, max_delay=1,
                 sig_threshold=0.05, discovery_sparsity=0.05, max_sepset_size=None, easy_to_find_root_causes=False):
        super().__init__(
            g=g,
            anomalous_vertices=anomalous_vertices,
            anomalies_start_time=anomalies_start_time,
            anomaly_length=anomaly_length,
            max_delay=max_delay,
            sig_threshold=sig_threshold,
            easy_to_find_root_causes=easy_to_find_root_causes
        )

        self.discovery_sparsity = discovery_sparsity
        self.max_sepset_size = max_sepset_size

        # Learned only if SGRCA leaves unresolved candidates.
        self.learned_ftcpdag = None

        # Optional bookkeeping:
        # candidates for which discovery allowed us to conclude
        # that there was no local mechanism change.
        self.resolved_non_root_causes = set()

    # =========================================================================
    # Learn FT-CPDAG with the SCG as background knowledge
    # =========================================================================

    def _learn_ftcpdag(self, data):
        """
        Learn an FT-CPDAG from the fully normal regime.

        The SCG is used as background knowledge.

        SCG interpretation
        ------------------
        If X -> Y is absent from the SCG, no temporal edge

            X_{t-gamma} -> Y_t

        is allowed.

        For contemporaneous X_t - Y_t:

            X -> Y only       => X_t -> Y_t
            Y -> X only       => Y_t -> X_t
            X -> Y and Y -> X => orientation left unresolved

        A self-lag X_{t-gamma} -> X_t is allowed only when the
        SCG contains the self-loop X -> X.
        """

        # ---------------------------------------------------------------------
        # Find the common fully normal prefix.
        # ---------------------------------------------------------------------

        if isinstance(self.anomalies_start_time, dict,):
            normal_end = min(self.anomalies_start_time.values())
        else:
            normal_end = self.anomalies_start_time

        if "time" in data.columns:

            normal_data = data.loc[
                data["time"] < normal_end
            ].drop(
                columns=["time"]
            ).copy()

        elif data.index.name == "time":

            normal_data = data.loc[
                data.index < normal_end
            ].copy()

        else:
            raise ValueError(
                "Data must contain a 'time' column or "
                "have an index named 'time'."
            )

        normal_data = ts_to_lagged_df(normal_data, window_size=self.max_delay + 1, dropna=True,)

        scg = self.g

        # ---------------------------------------------------------------------
        # TsPC subclass whose background-knowledge stage actually enforces
        # the information contained in the SCG.
        # ---------------------------------------------------------------------

        class SCGConstrainedTsPC(TsPC):

            def _apply_background_knowledge(tspc_self):
                directed_scg_edges = set(scg.get_directed_edges())

                uncertain_scg_edges = set()

                if hasattr(scg, "get_uncertain_edges",):
                    uncertain_scg_edges = set(scg.get_uncertain_edges())

                def macro_direction_allowed(source, target):
                    if (source, target) in directed_scg_edges:
                        return True

                    # An uncertain SCG adjacency does not
                    # determine the causal direction.
                    if (source, target) in uncertain_scg_edges:
                        return True

                    if (target, source) in uncertain_scg_edges:
                        return True

                    return False

                def remove_undirected(x, y):
                    tspc_self.g_hat.remove_undirected_edge(x, y)

                    # This mirrors TsPC._skeleton(), which
                    # removes both representations.
                    tspc_self.g_hat.remove_undirected_edge(y, x)

                # Work on a copy because some edges will be
                # removed/oriented during iteration.
                undirected_edges = list(tspc_self.g_hat.get_undirected_edges())

                for x, y in undirected_edges:

                    # =========================================================
                    # Same macro-variable
                    # =========================================================

                    if x.name == y.name:

                        # X_{t-gamma} -> X_t is possible only if
                        # the SCG contains X -> X.
                        if (x.name, x.name) not in directed_scg_edges:
                            remove_undirected(x, y,)

                        # Otherwise temporal orientation will orient
                        # the edge correctly later.
                        continue

                    # =========================================================
                    # Different time points
                    # =========================================================

                    if x.time < y.time:

                        # Temporal order forces X -> Y.
                        if not macro_direction_allowed(x.name, y.name,):
                            remove_undirected(x, y,)

                        continue

                    if y.time < x.time:

                        # Temporal order forces Y -> X.
                        if not macro_direction_allowed(y.name, x.name):
                            remove_undirected(x, y,)

                        continue

                    # =========================================================
                    # Contemporaneous edge X_t - Y_t
                    # =========================================================

                    x_to_y = macro_direction_allowed(x.name, y.name)

                    y_to_x = macro_direction_allowed(y.name, x.name)

                    if not x_to_y and not y_to_x:
                        # No macro adjacency allowed.
                        remove_undirected(x, y)

                    elif x_to_y and not y_to_x:

                        # SCG tells us the contemporaneous
                        # direction.
                        remove_undirected(x, y)

                        tspc_self.g_hat.add_directed_edge(x, y)

                    elif y_to_x and not x_to_y:

                        remove_undirected(x, y)

                        tspc_self.g_hat.add_directed_edge(y, x)

                    # If both macro directions are possible,
                    # leave X_t - Y_t unoriented and allow TsPC's
                    # UC / Meek rules to orient it if possible.

        # ---------------------------------------------------------------------
        # Run constrained temporal PC.
        # ---------------------------------------------------------------------

        tspc = SCGConstrainedTsPC(sparsity=self.discovery_sparsity)

        return tspc.run(data=normal_data, max_sepset_size=self.max_sepset_size)

    # =========================================================================
    # Parent-set identifiability in the learned FT-CPDAG
    # =========================================================================

    def _parent_set_is_identified(self, v):
        """
        Return True iff every edge incident to V_t is fully oriented.

        In that case, the directed incoming neighbours of V_t are its
        identified parent set in the learned FT-CPDAG. Any remaining
        undirected or uncertain incident edge leaves Pa(V_t) unresolved.
        """

        time_v = DTimeVar(v, 0)

        if time_v not in self.learned_ftcpdag.get_vertices():
            return False

        for x, y in self.learned_ftcpdag.get_undirected_edges():
            if x == time_v or y == time_v:
                return False

        # Be defensive in case the graph representation can contain
        # partially specified / uncertain edges in addition to the
        # ordinary CPDAG directed and undirected edges.
        if hasattr(self.learned_ftcpdag, "get_uncertain_edges"):
            for x, y in self.learned_ftcpdag.get_uncertain_edges():
                if x == time_v or y == time_v:
                    return False

        return True


    def _get_identified_parents(self, v):
        """
        Return the directed parents of V_t in the learned FT-CPDAG.
        """

        time_v = DTimeVar(v, 0)

        return set(self.learned_ftcpdag.get_parents(time_v))

    # =========================================================================
    # Locate the anomalous component containing V
    # =========================================================================

    def _get_component_id(self, v):
        for (subgraph_id, subgraph, ) in (self.maximally_connected_anomalous_components.items()):
            if v in subgraph.get_vertices():
                return subgraph_id

        raise ValueError(
            f"Variable {v!r} does not belong to any "
            "anomalous connected component."
        )

    # =========================================================================
    # Test one previously unresolved candidate
    # =========================================================================

    def _resolve_candidate(self, data, v):
        """
        Try to resolve one possible root cause.

        Returns
        -------
        "effect"
            V is effect-defying.

        "noise"
            V is noise-defying.

        "not_root"
            The learned parent set is identified and neither mechanism
            change test rejects.

        "unresolved"
            The learned FT-CPDAG does not identify Pa(V_t).
        """

        if not self._parent_set_is_identified(v):
            return "unresolved"

        parents = self._get_identified_parents(v)

        # ---------------------------------------------------------------------
        # Split the data exactly as SGRCA does.
        # ---------------------------------------------------------------------

        data_before, data_during = self._split_data_by_regime(data, v)

        data_before_lagged = (
            ts_to_lagged_df(
                data_before,
                window_size=self.max_delay + 1,
                dropna=True,
            )
        )

        data_during_lagged = (
            ts_to_lagged_df(
                data_during,
                window_size=self.max_delay + 1,
                dropna=True,
            )
        )

        normal_batches = (
            self._create_normal_batches(
                data_before_lagged,
                data_during_lagged,
                max_batches=10,
            )
        )

        time_v = DTimeVar(v, 0)

        anomalous_betas = {}

        # =====================================================================
        # Effect-defying test
        # =====================================================================

        for time_u in parents:

            # The exact other parents form the adjustment set.
            adjustment_set = (
                parents
                - {time_u}
            )

            lr = LinearRegressionCoefficient(
                time_u,
                time_v,
                cond_list=list(
                    adjustment_set
                ),
                drop_na=False,
            )

            dep_before_list = [
                lr.get_dependence(
                    batch
                )
                for batch in normal_batches
            ]

            dep_during = (
                lr.get_dependence(
                    data_during_lagged
                )
            )

            anomalous_betas[
                time_u
            ] = dep_during

            gamma = (
                time_v.time
                - time_u.time
            )

            # Keep SGRCA's bookkeeping compatible.
            self.estimated_direct_effects_in_anomalous_regime[
                (
                    time_u.name,
                    v,
                    gamma,
                )
            ] = dep_during

            grubb_result = grubb_test(
                [dep_during]
                + dep_before_list,
                sig_threshold=(
                    self.sig_threshold
                ),
            )

            if (
                grubb_result[
                    "anomaly_position"
                ]
                == 0
            ):
                return "effect"

        # =====================================================================
        # Noise-defying test
        # =====================================================================

        # Reproduce SGRCA's noise-defying procedure exactly, except that
        # the mechanism is reconstructed from the parent set identified
        # by temporal causal discovery.
        raw_normal_batches = self._create_normal_batches(
            data_before,
            data_during,
            max_batches=10,
        )

        data_during_noise = ts_to_lagged_df(
            data_during,
            window_size=self.max_delay + 1,
            dropna=True,
        )

        model_anomalous = 0

        for time_u in parents:
            beta = anomalous_betas[time_u]
            model_anomalous += beta * data_during_noise[time_u]

        residuals_anomalous = np.asarray(
            data_during_noise[time_v] - model_anomalous
        ).ravel()

        residuals_normal_list = []

        for raw_normal_batch in raw_normal_batches:
            normal_batch = ts_to_lagged_df(
                raw_normal_batch,
                window_size=self.max_delay + 1,
                dropna=True,
            )

            model_normal = 0

            for time_u in parents:
                beta = anomalous_betas[time_u]
                model_normal += beta * normal_batch[time_u]

            residuals_normal = np.asarray(
                normal_batch[time_v] - model_normal
            ).ravel()

            residuals_normal_list.append(residuals_normal)

        normal_distances = []

        for i in range(len(residuals_normal_list) - 1):
            normal_distances.append(
                energy_distance(
                    residuals_normal_list[i],
                    residuals_normal_list[i + 1],
                )
            )

        anomaly_distance = energy_distance(
            residuals_normal_list[-1],
            residuals_anomalous,
        )

        result = grubb_test(
            [anomaly_distance] + normal_distances,
            sig_threshold=self.sig_threshold,
        )

        if result["anomaly_position"] == 0:
            return "noise"

        return "not_root"


    # =========================================================================
    # Refine SGRCA's possible root causes
    # =========================================================================

    def _refine_possible_root_causes(
        self,
        data,
    ):
        """
        Use the learned FT-CPDAG to refine SGRCA's unresolved candidates.
        """

        # Remove candidates SGRCA has already classified by another route.
        already_classified = set()

        for component_result in (
            self.root_causes.values()
        ):
            already_classified.update(
                component_result.get(
                    "lineage_defying",
                    [],
                )
            )

            already_classified.update(
                component_result.get(
                    "time_defying",
                    [],
                )
            )

            already_classified.update(
                component_result.get(
                    "effect_defying",
                    [],
                )
            )

            already_classified.update(
                component_result.get(
                    "noise_defying",
                    [],
                )
            )

        self.possible_root_causes = (
            set(
                self.possible_root_causes
            )
            - already_classified
        )

        # Iterate over a copy because the set is modified below.
        for v in list(
            self.possible_root_causes
        ):

            decision = (
                self._resolve_candidate(
                    data,
                    v,
                )
            )

            # -------------------------------------------------------------
            # Still not identifiable after causal discovery.
            # -------------------------------------------------------------

            if decision == "unresolved":
                continue

            subgraph_id = (
                self._get_component_id(
                    v
                )
            )

            # -------------------------------------------------------------
            # Causal discovery resolved the ambiguity.
            # -------------------------------------------------------------

            if decision == "effect":

                if (
                    v
                    not in self.root_causes[
                        subgraph_id
                    ]["effect_defying"]
                ):
                    self.root_causes[
                        subgraph_id
                    ]["effect_defying"].append(
                        v
                    )

            elif decision == "noise":

                if (
                    v
                    not in self.root_causes[
                        subgraph_id
                    ]["noise_defying"]
                ):
                    self.root_causes[
                        subgraph_id
                    ]["noise_defying"].append(
                        v
                    )

            elif decision == "not_root":

                self.resolved_non_root_causes.add(
                    v
                )

            # In all three cases, V is no longer merely "possible".
            self.possible_root_causes.discard(
                v
            )


    # =========================================================================
    # Run
    # =========================================================================

    def _reset_results(self):
        super()._reset_results()

        self.learned_ftcpdag = None
        self.resolved_non_root_causes = set()

    def run(self, data,):
        """
        Run SGRCA first.

        Temporal causal discovery is performed only if SGRCA leaves
        unresolved possible root causes.
        """

        # ---------------------------------------------------------------------
        # Stage 1: ordinary SGRCA, unchanged.
        # ---------------------------------------------------------------------

        super().run(data)

        # ---------------------------------------------------------------------
        # Nothing unresolved: SGRCA+ is exactly SGRCA.
        # ---------------------------------------------------------------------

        if not self.possible_root_causes:
            return self.root_causes

        # ---------------------------------------------------------------------
        # Stage 2: targeted causal discovery.
        # ---------------------------------------------------------------------

        self.learned_ftcpdag = self._learn_ftcpdag(data)

        # ---------------------------------------------------------------------
        # Stage 3: use the additional temporal information only for the
        # candidates SGRCA could not resolve.
        # ---------------------------------------------------------------------

        self._refine_possible_root_causes(data)

        return self.root_causes


class SGRCAPlusOld(SGRCA):
    """
    SGRCA+.

    SGRCA+ first runs exactly as SGRCA.

    If SGRCA leaves some variables in `self.possible_root_causes`
    because their local mechanism cannot be identified from the SCG,
    SGRCA+ learns an FT-CPDAG from the normal regime using TsPC.

    The supplied SCG is used as background knowledge during temporal
    causal discovery.

    For each remaining possible root cause Y:

        * if all edges incident to Y_t are oriented in the learned
          FT-CPDAG, Pa(Y_t) is identified and the local mechanism can
          be tested directly;

        * otherwise Y remains a possible root cause.

    When Pa(Y_t) is identified, SGRCA+ uses the same Grubbs test and
    energy-distance permutation test as SGRCA.
    """

    def __init__(self, g, anomalous_vertices, anomalies_start_time, anomaly_length=200, max_delay=1,
                 sig_threshold=0.05, discovery_sparsity=0.05, max_sepset_size=None, easy_to_find_root_causes=False):
        super().__init__(
            g=g,
            anomalous_vertices=anomalous_vertices,
            anomalies_start_time=anomalies_start_time,
            anomaly_length=anomaly_length,
            max_delay=max_delay,
            sig_threshold=sig_threshold,
            easy_to_find_root_causes=easy_to_find_root_causes
        )

        self.discovery_sparsity = discovery_sparsity
        self.max_sepset_size = max_sepset_size

        # Learned only if SGRCA leaves unresolved candidates.
        self.learned_ftcpdag = None

        # Optional bookkeeping:
        # candidates for which discovery allowed us to conclude
        # that there was no local mechanism change.
        self.resolved_non_root_causes = set()

    # =========================================================================
    # Learn FT-CPDAG with the SCG as background knowledge
    # =========================================================================

    def _learn_ftcpdag(self, data):
        """
        Learn an FT-CPDAG from the fully normal regime.

        The SCG is used as background knowledge.

        SCG interpretation
        ------------------
        If X -> Y is absent from the SCG, no temporal edge

            X_{t-gamma} -> Y_t

        is allowed.

        For contemporaneous X_t - Y_t:

            X -> Y only       => X_t -> Y_t
            Y -> X only       => Y_t -> X_t
            X -> Y and Y -> X => orientation left unresolved

        A self-lag X_{t-gamma} -> X_t is allowed only when the
        SCG contains the self-loop X -> X.
        """

        # ---------------------------------------------------------------------
        # Find the common fully normal prefix.
        # ---------------------------------------------------------------------

        if isinstance(self.anomalies_start_time, dict,):
            normal_end = min(self.anomalies_start_time.values())
        else:
            normal_end = self.anomalies_start_time

        if "time" in data.columns:

            normal_data = data.loc[
                data["time"] < normal_end
            ].drop(
                columns=["time"]
            ).copy()

        elif data.index.name == "time":

            normal_data = data.loc[
                data.index < normal_end
            ].copy()

        else:
            raise ValueError(
                "Data must contain a 'time' column or "
                "have an index named 'time'."
            )

        normal_data = ts_to_lagged_df(normal_data, window_size=self.max_delay + 1, dropna=True,)

        scg = self.g

        # ---------------------------------------------------------------------
        # TsPC subclass whose background-knowledge stage actually enforces
        # the information contained in the SCG.
        # ---------------------------------------------------------------------

        class SCGConstrainedTsPC(TsPC):

            def _apply_background_knowledge(tspc_self):
                directed_scg_edges = set(scg.get_directed_edges())

                uncertain_scg_edges = set()

                if hasattr(scg, "get_uncertain_edges",):
                    uncertain_scg_edges = set(scg.get_uncertain_edges())

                def macro_direction_allowed(source, target):
                    if (source, target) in directed_scg_edges:
                        return True

                    # An uncertain SCG adjacency does not
                    # determine the causal direction.
                    if (source, target) in uncertain_scg_edges:
                        return True

                    if (target, source) in uncertain_scg_edges:
                        return True

                    return False

                def remove_undirected(x, y):
                    tspc_self.g_hat.remove_undirected_edge(x, y)

                    # This mirrors TsPC._skeleton(), which
                    # removes both representations.
                    tspc_self.g_hat.remove_undirected_edge(y, x)

                # Work on a copy because some edges will be
                # removed/oriented during iteration.
                undirected_edges = list(tspc_self.g_hat.get_undirected_edges())

                for x, y in undirected_edges:

                    # =========================================================
                    # Same macro-variable
                    # =========================================================

                    if x.name == y.name:

                        # X_{t-gamma} -> X_t is possible only if
                        # the SCG contains X -> X.
                        if (x.name, x.name) not in directed_scg_edges:
                            remove_undirected(x, y,)

                        # Otherwise temporal orientation will orient
                        # the edge correctly later.
                        continue

                    # =========================================================
                    # Different time points
                    # =========================================================

                    if x.time < y.time:

                        # Temporal order forces X -> Y.
                        if not macro_direction_allowed(x.name, y.name,):
                            remove_undirected(x, y,)

                        continue

                    if y.time < x.time:

                        # Temporal order forces Y -> X.
                        if not macro_direction_allowed(y.name, x.name):
                            remove_undirected(x, y,)

                        continue

                    # =========================================================
                    # Contemporaneous edge X_t - Y_t
                    # =========================================================

                    x_to_y = macro_direction_allowed(x.name, y.name)

                    y_to_x = macro_direction_allowed(y.name, x.name)

                    if not x_to_y and not y_to_x:
                        # No macro adjacency allowed.
                        remove_undirected(x, y)

                    elif x_to_y and not y_to_x:

                        # SCG tells us the contemporaneous
                        # direction.
                        remove_undirected(x, y)

                        tspc_self.g_hat.add_directed_edge(x, y)

                    elif y_to_x and not x_to_y:

                        remove_undirected(x, y)

                        tspc_self.g_hat.add_directed_edge(y, x)

                    # If both macro directions are possible,
                    # leave X_t - Y_t unoriented and allow TsPC's
                    # UC / Meek rules to orient it if possible.

        # ---------------------------------------------------------------------
        # Run constrained temporal PC.
        # ---------------------------------------------------------------------

        tspc = SCGConstrainedTsPC(sparsity=self.discovery_sparsity)

        return tspc.run(data=normal_data, max_sepset_size=self.max_sepset_size)

    # =========================================================================
    # Parent-set identifiability in the learned FT-CPDAG
    # =========================================================================

    def _parent_set_is_identified(self, v):
        """
        Return True iff every edge incident to V_t is oriented.

        In that case, the directed incoming neighbours of V_t are its
        parent set in the learned FT-CPDAG.
        """

        time_v = DTimeVar(v, 0)

        if (
            time_v
            not in self.learned_ftcpdag.get_vertices()
        ):
            return False

        for x, y in (
            self.learned_ftcpdag
            .get_undirected_edges()
        ):
            if (
                x == time_v
                or y == time_v
            ):
                return False

        return True


    def _get_identified_parents(self, v):
        """
        Return the directed parents of V_t in the learned FT-CPDAG.
        """

        time_v = DTimeVar(v, 0)

        return set(self.learned_ftcpdag.get_parents(time_v))

    # =========================================================================
    # Locate the anomalous component containing V
    # =========================================================================

    def _get_component_id(self, v):
        for (subgraph_id, subgraph, ) in (self.maximally_connected_anomalous_components.items()):
            if v in subgraph.get_vertices():
                return subgraph_id

        raise ValueError(
            f"Variable {v!r} does not belong to any "
            "anomalous connected component."
        )

    # =========================================================================
    # Test one previously unresolved candidate
    # =========================================================================

    def _resolve_candidate(self, data, v):
        """
        Try to resolve one possible root cause.

        Returns
        -------
        "effect"
            V is effect-defying.

        "noise"
            V is noise-defying.

        "not_root"
            The learned parent set is identified and neither mechanism
            change test rejects.

        "unresolved"
            The learned FT-CPDAG does not identify Pa(V_t).
        """

        if not self._parent_set_is_identified(v):
            return "unresolved"

        parents = self._get_identified_parents(v)

        # ---------------------------------------------------------------------
        # Split the data exactly as SGRCA does.
        # ---------------------------------------------------------------------

        data_before, data_during = self._split_data_by_regime(data, v)

        data_before_lagged = (
            ts_to_lagged_df(
                data_before,
                window_size=self.max_delay + 1,
                dropna=True,
            )
        )

        data_during_lagged = (
            ts_to_lagged_df(
                data_during,
                window_size=self.max_delay + 1,
                dropna=True,
            )
        )

        normal_batches = (
            self._create_normal_batches(
                data_before_lagged,
                data_during_lagged,
                max_batches=10,
            )
        )

        time_v = DTimeVar(v, 0)

        anomalous_betas = {}

        # =====================================================================
        # Effect-defying test
        # =====================================================================

        for time_u in parents:

            # The exact other parents form the adjustment set.
            adjustment_set = (
                parents
                - {time_u}
            )

            lr = LinearRegressionCoefficient(
                time_u,
                time_v,
                cond_list=list(
                    adjustment_set
                ),
                drop_na=False,
            )

            dep_before_list = [
                lr.get_dependence(
                    batch
                )
                for batch in normal_batches
            ]

            dep_during = (
                lr.get_dependence(
                    data_during_lagged
                )
            )

            anomalous_betas[
                time_u
            ] = dep_during

            gamma = (
                time_v.time
                - time_u.time
            )

            # Keep SGRCA's bookkeeping compatible.
            self.estimated_direct_effects_in_anomalous_regime[
                (
                    time_u.name,
                    v,
                    gamma,
                )
            ] = dep_during

            grubb_result = grubb_test(
                [dep_during]
                + dep_before_list,
                sig_threshold=(
                    self.sig_threshold
                ),
            )

            if (
                grubb_result[
                    "anomaly_position"
                ]
                == 0
            ):
                return "effect"

        # =====================================================================
        # Noise-defying test
        # =====================================================================

        # Same convention as SGRCA:
        # use the most recent normal batch.
        raw_normal_batches = (
            self._create_normal_batches(
                data_before,
                data_during,
                max_batches=10,
            )
        )

        normal_batch = (
            raw_normal_batches[-1]
        )

        normal_batch = ts_to_lagged_df(
            normal_batch,
            window_size=self.max_delay + 1,
            dropna=True,
        )

        data_during_noise = (
            ts_to_lagged_df(
                data_during,
                window_size=self.max_delay + 1,
                dropna=True,
            )
        )

        model_normal = 0
        model_anomalous = 0

        for time_u in parents:

            beta = anomalous_betas[
                time_u
            ]

            model_normal += (
                beta
                * normal_batch[
                    time_u
                ]
            )

            model_anomalous += (
                beta
                * data_during_noise[
                    time_u
                ]
            )

        residuals_normal = np.asarray(
            normal_batch[
                time_v
            ]
            - model_normal
        ).ravel()

        residuals_anomalous = np.asarray(
            data_during_noise[
                time_v
            ]
            - model_anomalous
        ).ravel()

        result = permutation_test(
            (
                residuals_normal,
                residuals_anomalous,
            ),
            statistic=energy_distance,
            permutation_type="independent",
            alternative="greater",
            n_resamples=1000,
            random_state=42,
        )

        if (
            result.pvalue
            < self.sig_threshold
        ):
            return "noise"

        return "not_root"


    # =========================================================================
    # Refine SGRCA's possible root causes
    # =========================================================================

    def _refine_possible_root_causes(
        self,
        data,
    ):
        """
        Use the learned FT-CPDAG to refine SGRCA's unresolved candidates.
        """

        # Remove candidates SGRCA has already classified by another route.
        already_classified = set()

        for component_result in (
            self.root_causes.values()
        ):
            already_classified.update(
                component_result.get(
                    "lineage_defying",
                    [],
                )
            )

            already_classified.update(
                component_result.get(
                    "time_defying",
                    [],
                )
            )

            already_classified.update(
                component_result.get(
                    "effect_defying",
                    [],
                )
            )

            already_classified.update(
                component_result.get(
                    "noise_defying",
                    [],
                )
            )

        self.possible_root_causes = (
            set(
                self.possible_root_causes
            )
            - already_classified
        )

        # Iterate over a copy because the set is modified below.
        for v in list(
            self.possible_root_causes
        ):

            decision = (
                self._resolve_candidate(
                    data,
                    v,
                )
            )

            # -------------------------------------------------------------
            # Still not identifiable after causal discovery.
            # -------------------------------------------------------------

            if decision == "unresolved":
                continue

            subgraph_id = (
                self._get_component_id(
                    v
                )
            )

            # -------------------------------------------------------------
            # Causal discovery resolved the ambiguity.
            # -------------------------------------------------------------

            if decision == "effect":

                if (
                    v
                    not in self.root_causes[
                        subgraph_id
                    ]["effect_defying"]
                ):
                    self.root_causes[
                        subgraph_id
                    ]["effect_defying"].append(
                        v
                    )

            elif decision == "noise":

                if (
                    v
                    not in self.root_causes[
                        subgraph_id
                    ]["noise_defying"]
                ):
                    self.root_causes[
                        subgraph_id
                    ]["noise_defying"].append(
                        v
                    )

            elif decision == "not_root":

                self.resolved_non_root_causes.add(
                    v
                )

            # In all three cases, V is no longer merely "possible".
            self.possible_root_causes.discard(
                v
            )


    # =========================================================================
    # Run
    # =========================================================================

    def _reset_results(self):
        super()._reset_results()

        self.learned_ftcpdag = None
        self.resolved_non_root_causes = set()

    def run(self, data,):
        """
        Run SGRCA first.

        Temporal causal discovery is performed only if SGRCA leaves
        unresolved possible root causes.
        """

        # ---------------------------------------------------------------------
        # Stage 1: ordinary SGRCA, unchanged.
        # ---------------------------------------------------------------------

        super().run(data)

        # ---------------------------------------------------------------------
        # Nothing unresolved: SGRCA+ is exactly SGRCA.
        # ---------------------------------------------------------------------

        if not self.possible_root_causes:
            return self.root_causes

        # ---------------------------------------------------------------------
        # Stage 2: targeted causal discovery.
        # ---------------------------------------------------------------------

        self.learned_ftcpdag = self._learn_ftcpdag(data)

        # ---------------------------------------------------------------------
        # Stage 3: use the additional temporal information only for the
        # candidates SGRCA could not resolve.
        # ---------------------------------------------------------------------

        self._refine_possible_root_causes(data)

        return self.root_causes


class FTDAGRCA(SGRCA):
    """
    Oracle FT-DAG version of SGRCA.

    The method follows the same RCA procedure as SGRCA, but receives the
    full finite-time DAG instead of only its summary causal graph.

    The SCG induced by the FT-DAG is used only for the parts of SGRCA that
    operate at the macro-variable level:

        * anomalous connected components,
        * lineage-defying root causes,
        * time-defying root causes.

    For effect-defying and noise-defying detection, the exact micro-level
    parents supplied by the FT-DAG are used directly.

    Under the assumed linear, causally sufficient FT-DAG, the coefficient
    associated with a direct edge

        U_{t-gamma} -> V_t

    is estimated by conditioning on the remaining exact parents of V_t.
    Therefore no SCG direct-effect identifiability test is needed.
    """

    def __init__(
        self,
        g: FtDirectedAcyclicGraph,
        anomalous_vertices,
        anomalies_start_time,
        anomaly_length=200,
        max_delay=1,
        sig_threshold=0.05, easy_to_find_root_causes=False
    ):
        # Full temporal graph available to the oracle baseline.
        self.ftdag = g

        # SGRCA's graph-level stages operate on macro variables.
        # Derive the SCG internally; it is NOT supplied to FT-DAG-RCA.
        self.scg = g.get_summary_causal_graph()

        # Reuse SGRCA's initialization exactly:
        #   - normalization of anomaly times/lengths,
        #   - anomalous connected components,
        #   - root-cause dictionaries,
        #   - storage of estimated direct effects.
        super().__init__(
            g=self.scg,
            anomalous_vertices=anomalous_vertices,
            anomalies_start_time=anomalies_start_time,
            anomaly_length=anomaly_length,
            max_delay=max_delay,
            sig_threshold=sig_threshold,
            easy_to_find_root_causes=easy_to_find_root_causes
        )

        self._exact_parent_cache = {}

        self._validate_ftdag_delay()


    # ========================================================================
    # FT-DAG utilities
    # ========================================================================

    def _validate_ftdag_delay(self):
        """
        Check that max_delay is large enough to represent every edge in the
        supplied FT-DAG.
        """
        max_ftdag_delay = 0

        for source, target in self.ftdag.get_directed_edges():
            delay = target.time - source.time

            if delay < 0:
                raise ValueError(
                    "FT-DAG contains an edge from the future to the past: "
                    f"{source} -> {target}."
                )

            max_ftdag_delay = max(
                max_ftdag_delay,
                delay,
            )

        if self.max_delay < max_ftdag_delay:
            raise ValueError(
                "max_delay is smaller than the largest delay present in "
                f"the FT-DAG: max_delay={self.max_delay}, "
                f"FT-DAG maximum delay={max_ftdag_delay}."
            )


    def _get_reference_target(self, v):
        """
        Return the latest finite-time realization of macro-variable v.

        For the stationary two-slice FT-DAGs used in the simulations,
        this corresponds to V_1. Its parents therefore encode the actual
        lag-0 and lag-1 parents of V_t.
        """
        candidates = [
            node
            for node in self.ftdag.get_vertices()
            if node.name == v
        ]

        if not candidates:
            raise ValueError(
                f"Variable {v!r} does not occur in the FT-DAG."
            )

        return max(
            candidates,
            key=lambda node: node.time,
        )


    def _get_exact_parent_relations(self, v):
        """
        Return the exact temporal parents of V_t as pairs

            (u, gamma)

        corresponding to actual FT-DAG edges

            U_{t-gamma} -> V_t.
        """
        if v in self._exact_parent_cache:
            return self._exact_parent_cache[v]

        target = self._get_reference_target(v)

        relations = []

        parents = sorted(
            self.ftdag.get_parents(target),
            key=lambda node: (node.name, node.time),
        )

        for parent in parents:
            gamma = target.time - parent.time

            if gamma < 0:
                raise ValueError(
                    f"Invalid temporal edge {parent} -> {target}."
                )

            if gamma > self.max_delay:
                raise ValueError(
                    f"Parent {parent} of {target} has delay {gamma}, "
                    f"larger than max_delay={self.max_delay}."
                )

            # A contemporaneous self-edge V_t -> V_t cannot be part
            # of a DAG, but keep the same safeguard as SGRCA.
            if parent.name == v and gamma == 0:
                continue

            relations.append(
                (parent.name, gamma)
            )

        self._exact_parent_cache[v] = relations

        return relations


    def _get_exact_adjustment_set(
        self,
        v,
        u,
        gamma,
    ):
        """
        Return the other exact parents of V_t.

        To estimate the coefficient of

            U_{t-gamma} -> V_t,

        condition on

            Pa(V_t) \\ {U_{t-gamma}}.

        In the linear causally sufficient FT-DAG this identifies the
        corresponding structural direct-effect coefficient.
        """
        adjustment_set = []

        for parent_name, parent_gamma in (
            self._get_exact_parent_relations(v)
        ):
            if (
                parent_name == u
                and parent_gamma == gamma
            ):
                continue

            adjustment_set.append(
                DTimeVar(
                    parent_name,
                    -parent_gamma,
                )
            )

        return adjustment_set


    # ========================================================================
    # Effect-defying root causes
    # ========================================================================

    def get_effect_defying(self, data):
        """
        Same effect-defying test as SGRCA, except that the exact FT-DAG
        parent relations are known and therefore no identifiability test
        over SCG-compatible FT-DAGs is required.
        """

        for (
            subgraph_id,
            subgraph,
        ) in self.maximally_connected_anomalous_components.items():

            for v in subgraph.get_vertices():

                if (
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["lineage_defying"]
                    or
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["time_defying"]
                ):
                    continue

                (
                    data_before_anomaly_v,
                    data_during_anomaly_v,
                ) = self._split_data_by_regime(
                    data,
                    v,
                )

                # Keep exactly the same lagged-data construction
                # used by SGRCA.
                data_before_anomaly_v = (
                    ts_to_lagged_df(
                        data_before_anomaly_v,
                        window_size=self.max_delay + 1,
                        dropna=True,
                    )
                )

                data_during_anomaly_v = (
                    ts_to_lagged_df(
                        data_during_anomaly_v,
                        window_size=self.max_delay + 1,
                        dropna=True,
                    )
                )

                data_before_anomaly_v_batches = (
                    self._create_normal_batches(
                        data_before_anomaly_v,
                        data_during_anomaly_v,
                        max_batches=10,
                    )
                )

                # ------------------------------------------------------------
                # Difference from SGRCA:
                #
                # SGRCA loops over SCG parents and candidate delays and then
                # asks whether each micro direct effect is identifiable.
                #
                # FT-DAG-RCA directly loops over the TRUE temporal parents.
                # ------------------------------------------------------------

                effect_change_found = False

                for (
                    u,
                    gamma,
                ) in self._get_exact_parent_relations(v):

                    adjustment_set = (
                        self._get_exact_adjustment_set(
                            v,
                            u,
                            gamma,
                        )
                    )

                    time_v = DTimeVar(
                        v,
                        0,
                    )

                    time_u = DTimeVar(
                        u,
                        -gamma,
                    )

                    lr = LinearRegressionCoefficient(
                        time_u,
                        time_v,
                        cond_list=list(
                            adjustment_set
                        ),
                        drop_na=False,
                    )

                    dep_before_list = []

                    for batch in (
                        data_before_anomaly_v_batches
                    ):
                        dep_before = (
                            lr.get_dependence(
                                batch
                            )
                        )

                        dep_before_list.append(
                            dep_before
                        )

                    dep_during = (
                        lr.get_dependence(
                            data_during_anomaly_v
                        )
                    )

                    # Keep the same dictionary structure as SGRCA.
                    self.estimated_direct_effects_in_anomalous_regime[
                        (
                            u,
                            v,
                            gamma,
                        )
                    ] = dep_during

                    # Same ordering as SGRCA:
                    # anomalous estimate first, then normal estimates.
                    dep_list = (
                        [dep_during]
                        + dep_before_list
                    )

                    grubb_test_res = grubb_test(
                        dep_list,
                        sig_threshold=(
                            self.sig_threshold
                        ),
                    )

                    # Same criterion as SGRCA.
                    if (
                        grubb_test_res[
                            "anomaly_position"
                        ]
                        == 0
                    ):
                        self.root_causes[
                            subgraph_id
                        ]["effect_defying"].append(
                            v
                        )

                        effect_change_found = True
                        break

                if effect_change_found:
                    continue


    # ========================================================================
    # Noise-defying root causes
    # ========================================================================

    def get_noise_defying(self, data):
        """
        Same noise-defying procedure as SGRCA, but reconstruct the mechanism
        using exactly the parents present in the FT-DAG.
        """

        for (
                subgraph_id,
                subgraph,
        ) in self.maximally_connected_anomalous_components.items():

            for v in subgraph.get_vertices():

                if (
                        v
                        in self.root_causes[
                    subgraph_id
                ]["lineage_defying"]
                        or
                        v
                        in self.root_causes[
                    subgraph_id
                ]["time_defying"]
                        or
                        v
                        in self.root_causes[
                    subgraph_id
                ]["effect_defying"]
                ):
                    continue

                (
                    data_before_anomaly_v,
                    data_during_anomaly_v,
                ) = self._split_data_by_regime(
                    data,
                    v,
                )

                data_before_anomaly_v_batches = (
                    self._create_normal_batches(
                        data_before_anomaly_v,
                        data_during_anomaly_v,
                        max_batches=10,
                    )
                )

                data_during_anomaly_v = (
                    ts_to_lagged_df(
                        data_during_anomaly_v,
                        window_size=self.max_delay + 1,
                        dropna=True,
                    )
                )

                parent_relations = (
                    self._get_exact_parent_relations(
                        v
                    )
                )

                missing_effects = [
                    (
                        u,
                        v,
                        gamma,
                    )
                    for (
                        u,
                        gamma,
                    ) in parent_relations
                    if (
                           u,
                           v,
                           gamma,
                       )
                       not in (
                           self
                           .estimated_direct_effects_in_anomalous_regime
                       )
                ]

                if missing_effects:
                    raise RuntimeError(
                        "Missing estimated direct effects for "
                        f"{v}: {missing_effects}"
                    )

                model_anomalous_regime = 0

                for (
                        u,
                        gamma,
                ) in parent_relations:
                    time_u = DTimeVar(
                        u,
                        -gamma,
                    )

                    beta = (
                        self
                        .estimated_direct_effects_in_anomalous_regime[
                            (
                                u,
                                v,
                                gamma,
                            )
                        ]
                    )

                    model_anomalous_regime += (
                            beta
                            * data_during_anomaly_v[
                                time_u
                            ]
                    )

                time_v = DTimeVar(
                    v,
                    0,
                )

                residuals_anomalous = np.asarray(
                    data_during_anomaly_v[
                        time_v
                    ]
                    - model_anomalous_regime
                ).ravel()

                residuals_normal_list = []

                for data_before_anomaly_v_batch in data_before_anomaly_v_batches:

                    data_before_anomaly_v_batch = (
                        ts_to_lagged_df(
                            data_before_anomaly_v_batch,
                            window_size=self.max_delay + 1,
                            dropna=True,
                        )
                    )

                    model_normal_regime = 0

                    for (
                            u,
                            gamma,
                    ) in parent_relations:
                        time_u = DTimeVar(
                            u,
                            -gamma,
                        )

                        beta = (
                            self
                            .estimated_direct_effects_in_anomalous_regime[
                                (
                                    u,
                                    v,
                                    gamma,
                                )
                            ]
                        )

                        model_normal_regime += (
                                beta
                                * data_before_anomaly_v_batch[
                                    time_u
                                ]
                        )

                    residuals_normal = np.asarray(
                        data_before_anomaly_v_batch[
                            time_v
                        ]
                        - model_normal_regime
                    ).ravel()

                    residuals_normal_list.append(
                        residuals_normal
                    )

                normal_distances = []

                for i in range(
                        len(residuals_normal_list) - 1
                ):
                    normal_distances.append(
                        energy_distance(
                            residuals_normal_list[i],
                            residuals_normal_list[i + 1],
                        )
                    )

                anomaly_distance = (
                    energy_distance(
                        residuals_normal_list[-1],
                        residuals_anomalous,
                    )
                )

                result = grubb_test(
                    [anomaly_distance]
                    + normal_distances,
                    sig_threshold=self.sig_threshold,
                )

                if (
                        result["anomaly_position"]
                        == 0
                ):
                    self.root_causes[
                        subgraph_id
                    ]["noise_defying"].append(
                        v
                    )

    def get_noise_defying_with_pvalue(self, data):
        """
        Same noise-defying procedure as SGRCA, but reconstruct the mechanism
        using exactly the parents present in the FT-DAG.
        """

        for (
            subgraph_id,
            subgraph,
        ) in self.maximally_connected_anomalous_components.items():

            for v in subgraph.get_vertices():

                if (
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["lineage_defying"]
                    or
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["time_defying"]
                    or
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["effect_defying"]
                ):
                    continue

                (
                    data_before_anomaly_v,
                    data_during_anomaly_v,
                ) = self._split_data_by_regime(
                    data,
                    v,
                )

                # Preserve SGRCA's normal-batch construction.
                data_before_anomaly_v_batches = (
                    self._create_normal_batches(
                        data_before_anomaly_v,
                        data_during_anomaly_v,
                        max_batches=10,
                    )
                )

                # Same as SGRCA: use the most recent normal batch.
                data_before_anomaly_v_batch = (
                    data_before_anomaly_v_batches[
                        -1
                    ]
                )

                data_during_anomaly_v = (
                    ts_to_lagged_df(
                        data_during_anomaly_v,
                        window_size=self.max_delay + 1,
                        dropna=True,
                    )
                )

                data_before_anomaly_v_batch = (
                    ts_to_lagged_df(
                        data_before_anomaly_v_batch,
                        window_size=self.max_delay + 1,
                        dropna=True,
                    )
                )

                model_anomalous_regime = 0
                model_normal_regime = 0

                parent_relations = (
                    self._get_exact_parent_relations(
                        v
                    )
                )

                # In FT-DAG-RCA all genuine direct effects are estimable.
                # This check should therefore always succeed for a node
                # reaching this stage.
                missing_effects = [
                    (
                        u,
                        v,
                        gamma,
                    )
                    for (
                        u,
                        gamma,
                    ) in parent_relations
                    if (
                        u,
                        v,
                        gamma,
                    )
                    not in (
                        self
                        .estimated_direct_effects_in_anomalous_regime
                    )
                ]

                if missing_effects:
                    raise RuntimeError(
                        "Missing estimated direct effects for "
                        f"{v}: {missing_effects}"
                    )

                # Reconstruct V_t's structural mechanism using exactly its
                # true FT-DAG parents.
                for (
                    u,
                    gamma,
                ) in parent_relations:

                    time_u = DTimeVar(
                        u,
                        -gamma,
                    )

                    beta = (
                        self
                        .estimated_direct_effects_in_anomalous_regime[
                            (
                                u,
                                v,
                                gamma,
                            )
                        ]
                    )

                    model_anomalous_regime += (
                        beta
                        * data_during_anomaly_v[
                            time_u
                        ]
                    )

                    model_normal_regime += (
                        beta
                        * data_before_anomaly_v_batch[
                            time_u
                        ]
                    )

                time_v = DTimeVar(
                    v,
                    0,
                )

                residuals_anomalous = np.asarray(
                    data_during_anomaly_v[
                        time_v
                    ]
                    - model_anomalous_regime
                ).ravel()

                residuals_normal = np.asarray(
                    data_before_anomaly_v_batch[
                        time_v
                    ]
                    - model_normal_regime
                ).ravel()

                # Exactly the same energy-distance permutation test
                # as SGRCA.
                result = permutation_test(
                    (
                        residuals_normal,
                        residuals_anomalous,
                    ),
                    statistic=energy_distance,
                    permutation_type="independent",
                    alternative="greater",
                    n_resamples=1000,
                    random_state=42,
                )

                if (
                    result.pvalue
                    < self.sig_threshold
                ):
                    self.root_causes[
                        subgraph_id
                    ]["noise_defying"].append(
                        v
                    )


class EasyRCA(SGRCA):
    """
    EasyRCA implemented as a restricted subclass of SGRCA.

    EasyRCA assumes that the supplied Summary Causal Graph is acyclic
    except possibly for self-loops.

    Under this assumption, all candidate linear direct effects considered
    by the method are treated as identifiable by adjustment.

    For the direct effect

        X_{t-gamma} -> Y_t,

    EasyRCA uses the parent-based adjustment set corresponding to
    Assaad, Ez-Zejjari & Zan (2023), Definition 11. In the terminology
    of SGRCA's `linear_direct_effect_is_identifiable` implementation,
    this is the third adjustment set.

    Unlike SGRCA, EasyRCA does not search for noise-defying root causes.
    """

    def __init__(
        self,
        g,
        anomalous_vertices,
        anomalies_start_time,
        anomaly_length=200,
        max_delay=1,
        sig_threshold=0.05,
        easy_to_find_root_causes=False
    ):
        # ------------------------------------------------------------
        # EasyRCA assumption:
        #
        # the SCG must be acyclic after ignoring self-loops.
        # ------------------------------------------------------------

        cycles = g.get_all_simple_cycles()

        acyclic_except_self_loops = all(
            len(cycle) == 1
            for cycle in cycles
        )

        if not acyclic_except_self_loops:
            raise ValueError(
                "EasyRCA requires an acyclic Summary Causal Graph "
                "except possibly for self-loops."
            )

        # Everything else is initialized exactly as in SGRCA.
        super().__init__(
            g=g,
            anomalous_vertices=anomalous_vertices,
            anomalies_start_time=anomalies_start_time,
            anomaly_length=anomaly_length,
            max_delay=max_delay,
            sig_threshold=sig_threshold,
            easy_to_find_root_causes=easy_to_find_root_causes
        )


    # ========================================================================
    # EasyRCA adjustment set
    # ========================================================================

    def _get_easyrca_adjustment_set(
        self,
        x,
        y,
        gamma,
    ):
        """
        Construct EasyRCA's parent-based adjustment set for

            X_{t-gamma} -> Y_t.

        This is the third adjustment set returned by SGRCA's
        `linear_direct_effect_is_identifiable`, namely the set based on
        Assaad, Ez-Zejjari & Zan (2023), Definition 11.

        It contains all possible parents of Y_t, except the exposure
        X_{t-gamma} itself.
        """

        exposure = DTimeVar(
            x,
            -gamma,
        )

        adjustment_set = set()

        # ------------------------------------------------------------
        # B = Parents(Y) \ {X, Y}
        #
        # For every other macro-parent B of Y, include
        #
        #     B_t, ..., B_{t-max_delay}.
        # ------------------------------------------------------------

        B = (
            set(self.g.get_parents(y))
            - {x, y}
        )

        for b in B:
            for lag in range(
                self.max_delay + 1
            ):
                adjustment_set.add(
                    DTimeVar(
                        b,
                        -lag,
                    )
                )

        # ------------------------------------------------------------
        # Include all possible temporal realizations of X:
        #
        #     X_t, ..., X_{t-max_delay},
        #
        # except the exposure X_{t-gamma}.
        # ------------------------------------------------------------

        for lag in range(
            self.max_delay + 1
        ):
            adjustment_set.add(
                DTimeVar(
                    x,
                    -lag,
                )
            )

        # ------------------------------------------------------------
        # If Y has a self-loop, include its possible lagged parents:
        #
        #     Y_{t-1}, ..., Y_{t-max_delay}.
        # ------------------------------------------------------------

        if (
            y,
            y,
        ) in self.g.get_directed_edges():

            for lag in range(
                1,
                self.max_delay + 1
            ):
                adjustment_set.add(
                    DTimeVar(
                        y,
                        -lag,
                    )
                )

        # Never adjust for the exposure itself.
        adjustment_set.discard(
            exposure
        )

        # Never condition on the outcome Y_t.
        adjustment_set.discard(
            DTimeVar(
                y,
                0,
            )
        )

        return adjustment_set


    # ========================================================================
    # Effect-defying root causes
    # ========================================================================

    def get_effect_defying(
        self,
        data,
    ):
        """
        Same effect-defying procedure as SGRCA, except:

        1. direct-effect identifiability is not tested;
        2. the EasyRCA parent-based adjustment set is always used.
        """

        for (
            subgraph_id,
            subgraph,
        ) in (
            self
            .maximally_connected_anomalous_components
            .items()
        ):

            for v in subgraph.get_vertices():

                # Same exclusions as SGRCA.
                if (
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["lineage_defying"]
                    or
                    v
                    in self.root_causes[
                        subgraph_id
                    ]["time_defying"]
                ):
                    continue

                # --------------------------------------------------------
                # Same regime splitting as SGRCA.
                # --------------------------------------------------------

                (
                    data_before_anomaly_v,
                    data_during_anomaly_v,
                ) = self._split_data_by_regime(
                    data,
                    v,
                )

                # --------------------------------------------------------
                # Same lagged representation as SGRCA.
                # --------------------------------------------------------

                data_before_anomaly_v = (
                    ts_to_lagged_df(
                        data_before_anomaly_v,
                        window_size=(
                            self.max_delay + 1
                        ),
                        dropna=True,
                    )
                )

                data_during_anomaly_v = (
                    ts_to_lagged_df(
                        data_during_anomaly_v,
                        window_size=(
                            self.max_delay + 1
                        ),
                        dropna=True,
                    )
                )

                # --------------------------------------------------------
                # Same normal batches as SGRCA.
                # --------------------------------------------------------

                data_before_anomaly_v_batches = (
                    self._create_normal_batches(
                        data_before_anomaly_v,
                        data_during_anomaly_v,
                        max_batches=10,
                    )
                )

                effect_change_found = False

                # --------------------------------------------------------
                # Same candidate macro parents / candidate lags as SGRCA.
                #
                # Difference:
                # no call to linear_direct_effect_is_identifiable().
                # --------------------------------------------------------

                for u in self.g.get_parents(v):

                    for gamma in range(
                        self.max_delay + 1
                    ):

                        # Y_t -> Y_t is not a direct-effect query.
                        if (
                            u == v
                            and gamma == 0
                        ):
                            continue

                        # ------------------------------------------------
                        # EasyRCA:
                        #
                        # acyclic SCG => direct effect is identifiable,
                        # and use adjustment set 3.
                        # ------------------------------------------------

                        adj_set = (
                            self
                            ._get_easyrca_adjustment_set(
                                u,
                                v,
                                gamma,
                            )
                        )

                        time_v = DTimeVar(
                            v,
                            0,
                        )

                        time_u = DTimeVar(
                            u,
                            -gamma,
                        )

                        lr = (
                            LinearRegressionCoefficient(
                                time_u,
                                time_v,
                                cond_list=list(
                                    adj_set
                                ),
                                drop_na=False,
                            )
                        )

                        # ------------------------------------------------
                        # Same normal-regime coefficient estimates
                        # as SGRCA.
                        # ------------------------------------------------

                        dep_before_list = []

                        for batch in (
                            data_before_anomaly_v_batches
                        ):

                            dep_before = (
                                lr.get_dependence(
                                    batch
                                )
                            )

                            dep_before_list.append(
                                dep_before
                            )

                        # ------------------------------------------------
                        # Same anomalous-regime coefficient estimate.
                        # ------------------------------------------------

                        dep_during = (
                            lr.get_dependence(
                                data_during_anomaly_v
                            )
                        )

                        self.estimated_direct_effects_in_anomalous_regime[
                            (
                                u,
                                v,
                                gamma,
                            )
                        ] = dep_during

                        # Same ordering used by SGRCA:
                        #
                        # anomalous estimate first.
                        dep_list = (
                            [dep_during]
                            + dep_before_list
                        )

                        # ------------------------------------------------
                        # Exactly the same Grubbs test as SGRCA.
                        # ------------------------------------------------

                        grubb_test_res = (
                            grubb_test(
                                dep_list,
                                sig_threshold=(
                                    self
                                    .sig_threshold
                                ),
                            )
                        )

                        if (
                            grubb_test_res[
                                "anomaly_position"
                            ]
                            == 0
                        ):
                            self.root_causes[
                                subgraph_id
                            ]["effect_defying"].append(
                                v
                            )

                            effect_change_found = True
                            break

                    if effect_change_found:
                        break

    # ========================================================================
    # Run
    # ========================================================================

    def run(self, data):
        """
        EasyRCA pipeline.

        Unlike SGRCA, EasyRCA does not search for noise-defying
        root causes.
        """
        if self.easy_to_find_root_causes:
            self.get_lineage_defying()
            self.get_time_defying()

        self.get_effect_defying(data)

        return self.root_causes


if __name__ == "__main__":
    # Example usage

    max_delay = 1
