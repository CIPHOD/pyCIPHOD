"""
TPC_BKSCG: TPC with the background knowledge of a summary causal graph (SCG),
plus a final "support" orientation step.

Public entry point:  run_tpc_bkscg(data, scg, ...)

What it does
------------
1. Builds the BK of the SCG (`bk_from_scg`):
     - forbidden edges between micro nodes of series that are non-adjacent in the SCG,
     - forbidden edges X_{t-k} -> X_{t-l} (k != l) for series without self-loop,
     - mandatory instantaneous orientation X_0 -> Y_0 when X -> Y is the only SCG edge
       between X and Y.
2. Runs TsPC with this BK (skeleton, time order, UC rule, BK, ...).
3. Support step (K3: every SCG edge X -> Y is supported by at least one micro edge
   X_{t-l} -> Y_t, 0 <= l <= gamma_max). For every SCG edge X -> Y such that
   X_0 - Y_0 is still undirected: if no lagged edge X_{-l} - Y_0 (l >= 1) is in the
   learned skeleton, the only possible support is instantaneous, hence X_0 -> Y_0.
   If the same argument applies to both directions (Y_0 -> X_0 too), the learned
   skeleton contradicts the SCG: the edge is left undirected and reported.
4. Closes under Meek's rules, then expands by stationarity (as TsPC.run does).
"""
import itertools
import time
from typing import Dict, Optional, Type

import pandas as pd

from pyciphod.causal_discovery.basic.ts_constraint_based import TsPC
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge
from pyciphod.utils.graphs.orientation_rules import (
    time_orientation, ts_uc_rule, apply_ts_meek_rules, bk_orientation_rules,
)
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest
from pyciphod.utils.time_series.data_format import DTimeVar


# ----------------------------------------------------------------------------
# BK from the SCG
# ----------------------------------------------------------------------------
def bk_from_scg(scg: SummaryCausalGraph, gamma_max: int) -> BackgroundKnowledge:
    bk = BackgroundKnowledge()
    nodes = list(scg.get_vertices())
    all_kl = list(itertools.product(range(gamma_max + 1), repeat=2))
    for x, y in itertools.combinations(nodes, 2):
        if y not in set(scg.get_adjacencies(x)):
            for k, l in all_kl:
                bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(y, -l))
                bk.add_forbidden_edge(DTimeVar(y, -k), DTimeVar(x, -l))
        else:
            for a, b in ((x, y), (y, x)):
                if a in set(scg.get_parents(b)) and a not in set(scg.get_children(b)):
                    bk.add_mandatory_orientation(DTimeVar(a, 0), DTimeVar(b, 0))
    for x in nodes:
        if x not in set(scg.get_adjacencies(x)):
            for k, l in all_kl:
                if k != l:
                    bk.add_forbidden_edge(DTimeVar(x, -k), DTimeVar(x, -l))
    return bk


# ----------------------------------------------------------------------------
# TsPC with the support step
# ----------------------------------------------------------------------------
class TsPCWithScgSupport(TsPC):
    """TsPC whose orientation phase also uses the SCG support constraint (see module doc)."""

    def __init__(
        self,
        scg: SummaryCausalGraph,
        sparsity: float = 0.05,
        ci_test: Type[CiTests] = FisherZTest,
        background_knowledge: Optional[BackgroundKnowledge] = None,
        twd: bool = False,
        oracle: bool = False,
        true_ftdag=None,
    ):
        super().__init__(
            sparsity=sparsity, ci_test=ci_test,
            background_knowledge=background_knowledge,
            twd=twd, oracle=oracle, true_ftdag=true_ftdag,
        )
        self._scg = scg
        self._gamma_max = scg.get_lag_max()
        self.n_support_orientations = 0
        self.support_conflicts = []  # pairs (X, Y) of series whose SCG support is contradicted

    def run(self, data: pd.DataFrame = None, max_sepset_size: int = None):
        nodes = self._validate_temporal_data(data)
        present = {(n.get_name(), n.get_time()) for n in nodes}
        missing = [(s, -l) for s in self._scg.get_vertices()
                   for l in range(self._gamma_max + 1) if (s, -l) not in present]
        if missing:
            raise ValueError(
                f"data must contain every series at lags 0..{self._gamma_max}; missing e.g. {missing[:3]}")
        return super().run(data=data, max_sepset_size=max_sepset_size)

    # -- helpers on the graph ------------------------------------------------
    def _undirected_pairs(self):
        return {frozenset(e) for e in self.g_hat.get_undirected_edges()}

    def _orient(self, a, b):
        """Replace the undirected edge a - b by a -> b (same remove pattern as TsPC._skeleton)."""
        self.g_hat.remove_undirected_edge(a, b)
        self.g_hat.remove_undirected_edge(b, a)
        self.g_hat.add_directed_edge(a, b)

    def _lag_supported(self, x: str, y: str) -> bool:
        """Is some X_{-l} (1 <= l <= gamma_max) adjacent to Y_0 in the learned skeleton?"""
        return any(self.g_hat.is_adjacent(DTimeVar(x, -l), DTimeVar(y, 0))
                   for l in range(1, self._gamma_max + 1))

    # -- support step ----------------------------------------------------------
    def _orient_unsupported_scg_edges(self):
        scg = self._scg
        names = list(scg.get_vertices())
        undirected = self._undirected_pairs()

        forced = set()  # ordered pairs (x, y): X_0 -> Y_0 is forced
        for x in names:
            for y in names:
                if x == y or x not in set(scg.get_parents(y)):
                    continue  # no SCG edge X -> Y
                if frozenset((DTimeVar(x, 0), DTimeVar(y, 0))) not in undirected:
                    continue  # absent or already directed
                reverse_in_scg = y in set(scg.get_parents(x))
                if (not reverse_in_scg) or (not self._lag_supported(x, y)):
                    forced.add((x, y))

        for x, y in sorted(forced):
            if (y, x) in forced:
                if x < y:
                    self.support_conflicts.append((x, y))
                continue  # contradictory: leave undirected
            self._orient(DTimeVar(x, 0), DTimeVar(y, 0))
            self.n_support_orientations += 1

    def _orientation(self):
        g = self.g_hat
        time_orientation(g)
        ts_uc_rule(g=g, sepset=self.sepset)
        bk_orientation_rules(g=g, bk=self._bk)
        self._orient_unsupported_scg_edges()
        repeat = True
        while repeat:
            repeat = apply_ts_meek_rules(g)


# ----------------------------------------------------------------------------
# Single function to call
# ----------------------------------------------------------------------------
def run_tpc_bkscg(
    data: pd.DataFrame,
    scg: SummaryCausalGraph,
    alpha: float = 0.05,
    ci_test: Type[CiTests] = FisherZTest,
    oracle: bool = False,
    true_ftdag=None,
    twd: bool = False,
) -> Dict:
    """
    Run TPC with the SCG background knowledge and the support step.

    `data`: lagged DataFrame whose columns are DTimeVar (times <= 0); columns with
    lag > gamma_max are dropped. Returns a dict with the learned edges, the number of
    CI tests, the wall-clock time and the support-step diagnostics.
    """
    gamma = scg.get_lag_max()
    sub = data[[c for c in data.columns if -c.time <= gamma]]
    t0 = time.perf_counter()
    tpc = TsPCWithScgSupport(
        scg=scg, sparsity=alpha, ci_test=ci_test,
        background_knowledge=bk_from_scg(scg, gamma),
        twd=twd, oracle=oracle, true_ftdag=true_ftdag,
    )
    tpc.run(sub)
    return dict(
        directed=set(tpc.g_hat.get_directed_edges()),
        undirected=set(tpc.g_hat.get_undirected_edges()),
        nb_tests=tpc.nb_ci_tests,
        time=time.perf_counter() - t0,
        n_support_orientations=tpc.n_support_orientations,
        support_conflicts=list(tpc.support_conflicts),
    )