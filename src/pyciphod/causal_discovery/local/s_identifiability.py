from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import pandas as pd

from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest as FisherZ
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.causal_discovery.local.s_orientability import sOrientability, OrientEdge


class sIdentifiability:
    """
    A priori s-identifiability from the SCG alone (Theorems s-id-total and s-id-cde).
    Time convention: lag = outcome.get_time() - treatment.get_time() (>= 0 for a possible effect).
    The linear direct effect is s-identifiable iff the CDE is.
    """
    def __init__(self, scg: SummaryCausalGraph):
        self._scg = scg
        self._sor = sOrientability(scg)
        self._gamma_max = scg.get_lag_max()

    def _non_s_orientable_with(self, n: str) -> Set[str]:
        """N(S_n, SCG): series not s-orientable with S_n."""
        return {z for z in self._scg.get_vertices()
                if z != n and not self._sor.is_s_orientable(DTimeVar(z, 0), DTimeVar(n, 0))[0]}

    def _an_under(self, nX: str, nY: str) -> Set[str]:
        """Ancestors of S_Y (Y included) in the SCG without the edges going out of S_X."""
        seen, stack = {nY}, [nY]
        while stack:
            for p in self._scg.get_parents(stack.pop()):
                if p != nX and p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def is_s_identifiable_TE(self, treatment: DTimeVar, outcome: DTimeVar) -> bool:
        lag = outcome.get_time() - treatment.get_time()
        nX, nY = treatment.get_name(), outcome.get_name()
        if lag < 0 or nX not in set(self._scg.get_ancestors(nY)) | {nY}:
            return True  # zero effect
        N = self._non_s_orientable_with(nX)
        return not (N if lag > 0 else N & self._an_under(nX, nY))

    def is_s_identifiable_CDE(self, treatment: DTimeVar, outcome: DTimeVar) -> bool:
        lag = outcome.get_time() - treatment.get_time()
        nX, nY = treatment.get_name(), outcome.get_name()
        if not 0 <= lag <= self._gamma_max or nX not in self._scg.get_parents(nY):
            return True  # zero effect
        return not self._non_s_orientable_with(nY)

    def is_s_identifiable(self, query: str, treatment: DTimeVar, outcome: DTimeVar) -> bool:
        if query in ("ATE", "TE"):
            return self.is_s_identifiable_TE(treatment, outcome)
        if query == "CDE":
            return self.is_s_identifiable_CDE(treatment, outcome)
        raise ValueError(f"unknown query {query!r}")


class ConstructAdjSet:
    """Algorithms Instantiate, BatchOrientViaColliders and ConstructAdjustmentSet."""
    ZERO = "zero effect"
    NOT_S_ORIENTABLE = "an edge is not s-orientable"

    def __init__(
        self,
        data: pd.DataFrame,
        scg: SummaryCausalGraph,
        alpha: float = 0.05,
        ci_test: Type[CiTests] = FisherZ,
        oracle: bool = False,
        true_ftdag=None,
        cache: Optional[Dict] = None,
    ):
        self.scg = scg
        self._gamma_max = scg.get_lag_max()
        self._oe = OrientEdge(data, scg, alpha, ci_test, oracle, true_ftdag, cache)

    @property
    def nb_ci_tests(self) -> int:
        return self._oe.nb_ci_tests

    def _instantiate(self, query: str, treatment: DTimeVar, outcome: DTimeVar
                     ) -> Tuple[DTimeVar, Set[DTimeVar], Set[DTimeVar], Set[DTimeVar]]:
        target = treatment if query in ("ATE", "TE") else outcome
        u, nT = target.get_time(), target.get_name()
        pa = set(self.scg.get_parents(nT))
        de = set(self.scg.get_descendants(nT)) | {nT}
        L = {DTimeVar(v, u - k) for v in pa for k in range(1, self._gamma_max + 1)}
        Q = {DTimeVar(v, u) for v in pa - de}
        R = {DTimeVar(v, u) for v in (pa & de) - {nT}}
        return target, L, Q, R

    def _batch_orient_via_colliders(self, nT: str, B: List[str]) -> Tuple[Set[str], Set[str]]:
        """B: series of nodes adjacent to T_0. Returns (parents of T_0 found, undecided)."""
        T0 = DTimeVar(nT, 0)
        Z, U = set(), set(B)
        for a, b in combinations(sorted(B), 2):
            if a not in U or b not in U or b in self.scg.get_adjacencies(a):
                continue
            S = self._oe._find_sepset(DTimeVar(a, 0), DTimeVar(b, 0))
            if S is not None and T0 not in S:
                Z |= {a, b}
                U -= {a, b}
        return Z, U

    def _run(self, query: str, treatment: DTimeVar, outcome: DTimeVar) -> Union[Set[DTimeVar], str]:
        is_te = query in ("ATE", "TE")
        lag = outcome.get_time() - treatment.get_time()
        nX, nY = treatment.get_name(), outcome.get_name()
        if is_te and (lag < 0 or nX not in set(self.scg.get_ancestors(nY)) | {nY}):
            return self.ZERO
        if not is_te and (not 0 <= lag <= self._gamma_max or nX not in self.scg.get_parents(nY)):
            return self.ZERO

        target, L, Q, R = self._instantiate(query, treatment, outcome)
        u, nT = target.get_time(), target.get_name()
        T0 = DTimeVar(nT, 0)

        Z, B = set(), []
        for r in sorted(R):
            n = r.get_name()
            if self._oe._find_sepset(DTimeVar(n, 0), T0) is not None:
                continue  # non-adjacent
            if nT not in self.scg.get_parents(n):
                Z.add(n)  # n -> T by (K3)
            else:
                B.append(n)

        Zb, U = self._batch_orient_via_colliders(nT, B)
        Z |= Zb
        for n in sorted(U):
            o = self._oe._run(DTimeVar(n, 0), T0)
            if o == OrientEdge.NOT_S_ORIENTABLE:
                return self.NOT_S_ORIENTABLE
            if o == f"{DTimeVar(n, 0)} -> {T0}":
                Z.add(n)

        if is_te and outcome.get_time() == u and nY in Z:
            return self.ZERO
        return (L | Q | {DTimeVar(n, u) for n in Z}) - {treatment}

    def run(self, query: str, treatment: DTimeVar, outcome: DTimeVar) -> Union[Set[DTimeVar], str]:
        if query not in ("ATE", "TE", "CDE"):
            raise ValueError(f"unknown query {query!r}")
        return self._run(query, treatment, outcome)