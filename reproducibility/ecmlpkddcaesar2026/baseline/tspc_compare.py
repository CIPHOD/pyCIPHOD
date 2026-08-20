from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.graphs.partially_specified_graphs import (
    FtCompletedPartiallyDirectedAcyclicGraph,
)
from pyciphod.utils.time_series.data_format import DTimeVar

from pyciphod.utils.stat_tests.equality_tests import LinearRegressionCoefficientEqualityTest


DirectedEdge = tuple[DTimeVar, DTimeVar]
UndirectedPair = tuple[DTimeVar, DTimeVar]


def node_key(node: DTimeVar) -> tuple[str, int]:
    return str(node.name), int(node.time)


def canonical_pair(x: DTimeVar, y: DTimeVar) -> UndirectedPair:
    return tuple(sorted((x, y), key=node_key))  # type: ignore[return-value]


def admissible(parent: DTimeVar, child: DTimeVar) -> bool:
    """Only past/current -> current directions are admissible."""
    return (parent != child and int(child.time) == 0 and int(parent.time) in {-1, 0})


class TsPCCompare:
    """
    Steps
    -----
    1. Run TsPC separately in the normal and anomalous regimes.
    2. Build their union graph.
    3. For every candidate relation, compare its regression coefficient
       between the two regimes, using the other possible parents of the target
       in the union graph as a common conditioning set.
    4. Convert significant changed relations into root causes.

    Union rules
    -----------
    X -- Y + X -- Y       -> X -- Y
    X -> Y + X -- Y       -> X -> Y
    X -> Y + absence      -> X -> Y
    absence + X -> Y      -> X -> Y
    X -> Y + X -> Y       -> X -> Y
    X -> Y + Y -> X       -> keep both directions and record a conflict
    (X -> Y and Y -> X) + absence
                           -> keep both changed directions

    Root-cause rules
    ----------------
    Significant X -> Y:
        Y is a root cause.

    Significant X -- Y:
        both X and Y are root causes because the changed relation cannot
        be oriented.

    Double contemporaneous edge present in only one regime:
        both directed edges are retained in the union and marked as changed;
        both X and Y are root causes.

    Orientation conflict with adjacency present in both regimes:
        the two directions are retained in the union, but the conflict alone
        does not identify a changed mechanism. Candidate mechanisms targeting
        a conflict node are skipped, as in the previous convention.
    """

    def __init__(self, alpha: float = 0.05, pc_alpha: float = 0.05, max_sepset_size: int | None = None,) -> None:
        self.alpha = float(alpha)
        self.pc_alpha = float(pc_alpha)
        self.max_sepset_size = max_sepset_size

        self.tspc_normal: TsPC | None = None
        self.tspc_anomalous: TsPC | None = None

        self.normal_directed: set[DirectedEdge] = set()
        self.normal_undirected: set[UndirectedPair] = set()
        self.anomalous_directed: set[DirectedEdge] = set()
        self.anomalous_undirected: set[UndirectedPair] = set()

        self.union_directed: set[DirectedEdge] = set()
        self.union_undirected: set[UndirectedPair] = set()
        self.orientation_conflicts: set[UndirectedPair] = set()
        self.changed_orientation_conflicts: set[UndirectedPair] = set()
        self.conflict_nodes: set[DTimeVar] = set()

        self.changed_directed: set[DirectedEdge] = set()
        self.changed_undirected: set[UndirectedPair] = set()
        self.predicted_root_causes: set[str] = set()

        self.skipped_edges: set[DirectedEdge] = set()
        self.test_results: list[dict[str, Any]] = []

        self.g_hat = FtCompletedPartiallyDirectedAcyclicGraph()

        self.nb_pc_tests = 0
        self.nb_eq_tests = 0
        self.nb_ci_tests = 0

    @staticmethod
    def _check_data(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        if list(df1.columns) != list(df2.columns):
            raise ValueError("The two regimes must have identical columns.")
        if not all(isinstance(node, DTimeVar) for node in df1.columns):
            raise TypeError("Columns must be DTimeVar objects.")

        times = {int(node.time) for node in df1.columns}
        if not times.issubset({-1, 0}):
            raise ValueError(
                "This simplified implementation only supports lag 1 "
                "(column times must be -1 or 0)."
            )

        if not np.isfinite(df1.to_numpy(dtype=float)).all():
            raise ValueError("The normal regime contains NaN or infinity.")
        if not np.isfinite(df2.to_numpy(dtype=float)).all():
            raise ValueError("The anomalous regime contains NaN or infinity.")

    def _run_tspc(self, data: pd.DataFrame) -> TsPC:
        algo = TsPC(sparsity=self.pc_alpha, twd=False)
        algo.run(data=data, max_sepset_size=self.max_sepset_size)
        return algo

    @staticmethod
    def _extract_edges(graph: FtCompletedPartiallyDirectedAcyclicGraph,) -> tuple[set[DirectedEdge], set[UndirectedPair]]:
        directed = {(parent, child) for parent, child in graph.get_directed_edges() if int(child.time) == 0}
        undirected = {canonical_pair(x, y) for x, y in graph.get_undirected_edges() if int(x.time) == 0 and int(y.time) == 0}
        return directed, undirected

    @staticmethod
    def _directions_for_pair(pair: UndirectedPair, directed: set[DirectedEdge],) -> set[DirectedEdge]:
        x, y = pair
        return {edge for edge in ((x, y), (y, x)) if edge in directed}

    def _build_union(self) -> None:
        normal_pairs = self.normal_undirected | {canonical_pair(*edge) for edge in self.normal_directed}
        anomalous_pairs = self.anomalous_undirected | {canonical_pair(*edge) for edge in self.anomalous_directed}

        all_pairs = normal_pairs | anomalous_pairs

        for pair in all_pairs:
            x, y = pair
            present_normal = pair in normal_pairs
            present_anomalous = pair in anomalous_pairs

            normal_directions = self._directions_for_pair( pair, self.normal_directed)
            anomalous_directions = self._directions_for_pair(pair, self.anomalous_directed)
            directions = normal_directions | anomalous_directions

            # 1. Adjacency present in only one regime: the relation has
            # necessarily changed.
            if present_normal != present_anomalous:
                if len(directions) == 2:
                    # TsPC returned both contemporaneous orientations in the
                    # regime where the adjacency is present. Keep both in the
                    # union. Since the pair is absent in the other regime,
                    # both directed relations are changed and both targets are
                    # root causes.
                    self.union_directed.update(directions)
                    self.changed_directed.update(directions)
                    self.orientation_conflicts.add(pair)
                    self.changed_orientation_conflicts.add(pair)
                    self.conflict_nodes.update(pair)
                    self.predicted_root_causes.update({str(x.name), str(y.name)})

                elif len(directions) == 1:
                    edge = next(iter(directions))
                    self.union_directed.add(edge)
                    self.changed_directed.add(edge)
                    _, child = edge
                    self.predicted_root_causes.add(str(child.name))

                else:
                    # The only available representation is X -- Y.
                    self.union_undirected.add(pair)
                    self.changed_undirected.add(pair)
                    self.predicted_root_causes.update({str(x.name), str(y.name)})
                continue

            # 2. Adjacency present in both regimes.
            if len(directions) == 2:
                # Keep the two directed edges in the union and separately
                # record that their orientations conflict. Presence in both
                # regimes means the conflict alone is not yet evidence of a
                # changed adjacency or coefficient.
                self.union_directed.update(directions)
                self.orientation_conflicts.add(pair)
                self.conflict_nodes.update(pair)

            elif len(directions) == 1:
                self.union_directed.update(directions)

            else:
                self.union_undirected.add(pair)


    def _possible_parents(self) -> dict[DTimeVar, set[DTimeVar]]:
        parents: dict[DTimeVar, set[DTimeVar]] = defaultdict(set)

        for parent, child in self.union_directed:
            if admissible(parent, child):
                parents[child].add(parent)

        for x, y in self.union_undirected:
            # Union undirected edges are contemporaneous here.
            parents[x].add(y)
            parents[y].add(x)

        return parents
    

    def _test_edge(self, df1: pd.DataFrame, df2: pd.DataFrame, edge: DirectedEdge, possible_parents: dict[DTimeVar, set[DTimeVar]], edge_type: str,) -> bool:

        parent, child = edge

        conditioning_set = sorted(possible_parents.get(child, set()) - {parent}, key=node_key,)

        test = LinearRegressionCoefficientEqualityTest(x=parent, y=child, cond_list=conditioning_set,)

        pvalue = test.get_pvalue(df1, df2)

        changed = np.isfinite(pvalue) and pvalue <= self.alpha

        self.nb_eq_tests += 1

        self.test_results.append(
            {
                "parent": parent,
                "child": child,
                "conditioning_set": conditioning_set,
                "pvalue": pvalue,
                "changed": changed,
            }
        )

        return changed


    def run(self, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        self._check_data(df1, df2)

        self.tspc_normal = self._run_tspc(df1)
        self.tspc_anomalous = self._run_tspc(df2)

        self.nb_pc_tests = int(self.tspc_normal.nb_ci_tests or 0) + int(self.tspc_anomalous.nb_ci_tests or 0)

        self.normal_directed, self.normal_undirected = self._extract_edges(self.tspc_normal.g_hat)
        self.anomalous_directed, self.anomalous_undirected = self._extract_edges(self.tspc_anomalous.g_hat)

        self._build_union()
        possible_parents = self._possible_parents()

        # Directed union edges: a significant test identifies the target node.
        for edge in sorted(self.union_directed, key=lambda e: (*node_key(e[0]), *node_key(e[1])),):
            if edge in self.changed_directed:
                continue
            parent, child = edge

            if child in self.conflict_nodes:
                self.skipped_edges.add(edge)
                continue

            if self._test_edge(df1, df2, edge, possible_parents, "directed"):
                self.changed_directed.add(edge)
                self.predicted_root_causes.add(str(child.name))

        # Undirected union edges: if a change is detected in either admissible
        # direction, both endpoints are possible root causes.
        for pair in sorted(self.union_undirected, key=lambda p: (*node_key(p[0]), *node_key(p[1])),):
            if pair in self.changed_undirected:
                continue
            x, y = pair
            if x in self.conflict_nodes or y in self.conflict_nodes:
                continue

            changed = False
            if admissible(x, y):
                changed |= self._test_edge(df1, df2, (x, y), possible_parents, "undirected")
            if admissible(y, x):
                changed |= self._test_edge(df1, df2, (y, x), possible_parents, "undirected")

            if changed:
                self.changed_undirected.add(pair)
                self.predicted_root_causes.update({str(x.name), str(y.name)})

        for parent, child in self.changed_directed:
            self.g_hat.add_directed_edge(parent, child)

        for x, y in self.changed_undirected:
            self.g_hat.add_undirected_edge(x, y)

        self.nb_ci_tests = self.nb_pc_tests + self.nb_eq_tests
