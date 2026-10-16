from abc import ABC, abstractmethod
from itertools import combinations
from typing import Optional, Type

import pandas as pd


from pyciphod.utils.graphs.orientation_rules import (
    time_orientation, uc_rule, apply_ts_meek_rules, bk_orientation_rules,
)
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.utils.graphs.partially_specified_graphs import PartiallyDirectedGraphs, DirectedAcyclicGraph
from pyciphod.utils.stat_tests.independence_tests import (
    CiTests,
    FisherZTest as FisherZ,
)
from pyciphod.utils.time_series.data_format import DTimeVar

from pyciphod.utils.graphs.separation import d_separated

from pyciphod.utils.graphs.partially_specified_graphs import FtCompletedPartiallyDirectedAcyclicGraph


# helpers
def node_key(node):
    return (str(node.name), int(node.time))

def edge_key(edge):
    u, v = edge
    return (
        str(u.name), int(u.time),
        str(v.name), int(v.time),
    )

def sorted_edges(edges):
    return sorted(edges, key=edge_key)



class TsConstraintBased(ABC):
    """Base class for temporal constraint-based causal discovery algorithms."""

    def __init__(
        self,
        sparsity: float = 0.05,
        ci_test: Type[CiTests] = FisherZ,
        background_knowledge: Optional[BackgroundKnowledge] = None,
        twd: bool = False,
        oracle: bool = False,
        true_ftdag: DirectedAcyclicGraph = None
    ):
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = {}
        self.g_hat = self._initialize_graph()
        self._oracle = oracle
        if self._oracle:
            self._true_ftdag = true_ftdag

    def _initialize_graph(self):
        """Return the temporal graph object used by the algorithm."""
        return FtCompletedPartiallyDirectedAcyclicGraph()

    @abstractmethod
    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the temporal graph."""
        pass

    def _missing_sepset_bk(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Look for a sepset for edges skipped during `_skeleton` because of BK."""
        pass

    def _validate_temporal_data(self, data: pd.DataFrame):
        """Check that the input data contains DTimeVar columns."""
        if data is None:
            raise ValueError("data must be provided.")

        nodes = list(data.columns)
        if not all(isinstance(node, DTimeVar) for node in nodes):
            raise TypeError("All columns must be DTimeVar objects.")

        return nodes

    @abstractmethod
    def _orientation(self):
        """Orient edges using temporal order, UC rule and Meek rules."""
        pass

    def _stationary_expand_graph(self, nodes):
        """Expand learned temporal edges across the lag window by stationarity."""
        node_lookup = {(node.name, node.time): node for node in nodes}
        time_points = sorted({node.time for node in nodes})

        directed_edges = sorted_edges(self.g_hat.get_directed_edges())
        undirected_edges = sorted_edges(self.g_hat.get_undirected_edges())

        for x, y in directed_edges:
            for shift in time_points:
                x_new = node_lookup.get((x.name, x.time + shift))
                y_new = node_lookup.get((y.name, y.time + shift))

                if x_new is None or y_new is None or x_new == y_new:
                    continue

                if not self.g_hat.is_adjacent(x_new, y_new):
                    self.g_hat.add_directed_edge(x_new, y_new)

        for x, y in undirected_edges:
            for shift in time_points:
                x_new = node_lookup.get((x.name, x.time + shift))
                y_new = node_lookup.get((y.name, y.time + shift))

                if x_new is None or y_new is None or x_new == y_new:
                    continue

                if not self.g_hat.is_adjacent(x_new, y_new):
                    self.g_hat.add_undirected_edge(x_new, y_new)

    def run(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Execute the full temporal constraint-based algorithm."""
        nodes = self._validate_temporal_data(data)

        self._skeleton(data=data, max_sepset_size=max_sepset_size)
        if self._bk:
            self._missing_sepset_bk(data=data, max_sepset_size=max_sepset_size)
        self._orientation()
        self._stationary_expand_graph(nodes)
        return self.g_hat


class TsPC(TsConstraintBased):
    """Temporal PC algorithm for lag-augmented time-series data."""

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Construct the skeleton of the temporal graph using CI tests."""
        nodes = self._validate_temporal_data(data)

        candidate_edges = [
            (x, y)
            for x, y in combinations(sorted(nodes), 2)
            if ((x.time <= 0 and y.time == 0) or (y.time <= 0 and x.time == 0))
        ]

        forbidden_edges = self._bk.get_forbidden_edges()
        mandatory_edges = self._bk.get_mandatory_edges()
        mandatory_orientation = self._bk.get_mandatory_orientations()

        self.g_hat.add_undirected_edges_from(candidate_edges)

        if max_sepset_size is None:
            max_sepset_size = len(nodes) - 2

        s = 0
        repeat = True

        while repeat and s <= max_sepset_size:
            repeat = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}

            for x in nodes:
                for y in sorted(adj[x]):
                    if not ((x.time <= 0 and y.time == 0) or (y.time <= 0 and x.time == 0)):
                        continue

                    # BK step: skip testing edges already constrained by BK, to avoid
                    # removing an edge that BK requires to exist, or contradicting a
                    # forbidden edge with a spurious dependence test.
                    # Following Schubert et al., "Integrating Background Knowledge for
                    # Scalable Causal Discovery."
                    if any(
                        edge in mandatory_edges or edge in forbidden_edges
                        for edge in ((x, y), (y, x))
                    ):
                        continue

                    # Candidate separating sets must be adjacent to x (standard PC),
                    # and must avoid nodes whose relation to x is already fixed/forbidden
                    # by BK, following Schubert et al.
                    possible_S = [
                        a for a in adj[x]
                        if (
                            a != x
                            and a != y
                            and (x, a) not in mandatory_orientation
                            and (x, a) not in forbidden_edges
                            and (a, x) not in forbidden_edges
                        )
                    ]

                    if len(possible_S) >= s:
                        repeat = True

                    for S in combinations(possible_S, s):
                        if self._oracle:
                            gamma_max = -min(v.get_time() for v in nodes)
                            x_in_ftdag = DTimeVar(x.get_name(),gamma_max+x.get_time())
                            y_in_ftdag = DTimeVar(y.get_name(),gamma_max+y.get_time())
                            candidate_sepset_in_ftag = [
                                                        DTimeVar(Z.get_name(), gamma_max + Z.get_time())
                                                        for Z in S
                                                    ]
                            is_sep = d_separated(self._true_ftdag, [x_in_ftdag], [y_in_ftdag], candidate_sepset_in_ftag)
                            self.nb_ci_tests += 1
                            if is_sep:
                                self.g_hat.remove_undirected_edge(x, y)
                                self.g_hat.remove_undirected_edge(y, x)
                                self.sepset[(x, y)] = S
                                self.sepset[(y, x)] = S
                                break
                                    
                        else:
                            test = self._ci_test(x, y, list(S), self._twd)
                            self.performed_tests.add((x, y, S))
                            self.nb_ci_tests += 1

                            data_test = data
                            if self._twd:
                                data_test = data.dropna(subset=[x, y] + list(S))

                            try:
                                pval = test.get_pvalue(data_test)
                            except Exception:
                                pval = test.get_pvalue_by_permutation(data_test)

                            if pval > self._sparsity:
                                self.g_hat.remove_undirected_edge(x, y)
                                self.g_hat.remove_undirected_edge(y, x)
                                self.sepset[(x, y)] = S
                                self.sepset[(y, x)] = S
                                break

            s += 1

    def _missing_sepset_bk(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        """Resolve edges skipped by BK during `_skeleton`.

        `_skeleton` intentionally does not test pairs that are mandatory or
        forbidden by background knowledge, to avoid a CI test contradicting
        BK (Schubert et al.). Forbidden edges therefore remain in the
        skeleton after `_skeleton` and must still be resolved here: we search
        for a BK-consistent separating set and remove them if one is found.
        Mandatory edges are never removed, so they are not revisited.

        Note: if no BK-consistent separating set is found up to
        `max_sepset_size`, the edge is left in place (data/BK contradiction);
        forcing its removal without a valid sepset would compromise the
        soundness of downstream v-structure orientation.
        """
        nodes = self._validate_temporal_data(data)

        forbidden_edges = self._bk.get_forbidden_edges()
        mandatory_orientation = self._bk.get_mandatory_orientations()

        if max_sepset_size is None:
            max_sepset_size = len(nodes) - 2

        s = 0
        repeat = True

        while repeat and s <= max_sepset_size:
            repeat = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}

            for x, y in forbidden_edges:
                if not ((x.time <= 0 and y.time == 0) or (y.time <= 0 and x.time == 0)):
                    continue

                if not self.g_hat.is_adjacent(x, y):
                    continue

                possible_S = [
                    a for a in adj[x]
                    if (
                        a != x
                        and a != y
                        and (x, a) not in mandatory_orientation
                        and (x, a) not in forbidden_edges
                        and (a, x) not in forbidden_edges
                    )
                ]

                if len(possible_S) >= s:
                    repeat = True

                for S in combinations(possible_S, s):
                    
                    if self._oracle:
                        gamma_max = -min(v.get_time() for v in nodes)
                        x_in_ftdag = DTimeVar(x.get_name(),gamma_max+x.get_time())
                        y_in_ftdag = DTimeVar(y.get_name(),gamma_max+y.get_time())
                        candidate_sepset_in_ftag = [
                                                    DTimeVar(Z.get_name(), gamma_max + Z.get_time())
                                                    for Z in S
                                                ]
                        is_sep = d_separated(self._true_ftdag, [x_in_ftdag], [y_in_ftdag], candidate_sepset_in_ftag)
                        self.nb_ci_tests += 1
                        if is_sep:
                            self.g_hat.remove_undirected_edge(x, y)
                            self.g_hat.remove_undirected_edge(y, x)
                            self.sepset[(x, y)] = S
                            self.sepset[(y, x)] = S
                            break
                    
                    else:
                        test = self._ci_test(x, y, list(S), self._twd)
                        self.performed_tests.add((x, y, S))
                        self.nb_ci_tests += 1

                        data_test = data
                        if self._twd:
                            data_test = data.dropna(subset=[x, y] + list(S))

                        try:
                            pval = test.get_pvalue(data_test)
                        except Exception:
                            pval = test.get_pvalue_by_permutation(data_test)

                        if pval > self._sparsity:
                            self.g_hat.remove_undirected_edge(x, y)
                            self.g_hat.remove_undirected_edge(y, x)
                            self.sepset[(x, y)] = S
                            self.sepset[(y, x)] = S
                            break

            s += 1

    def _orientation(self):
        """Orient edges using temporal order, UC rule, BK and Meek rules."""
        time_orientation(self.g_hat)
        uc_rule(g=self.g_hat, sepset=self.sepset)
        bk_orientation_rules(g=self.g_hat, bk=self._bk)
        repeat = True
        while repeat:
            repeat = apply_ts_meek_rules(self.g_hat)