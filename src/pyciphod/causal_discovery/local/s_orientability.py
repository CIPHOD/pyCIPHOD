from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Type, Union

import pandas as pd

from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest as FisherZ
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.separation import d_separated


class sOrientability:
    """
    A priori s-orientability from the SCG alone. For A != B at the same time, the relation
    is NOT s-orientable iff S_A <-> S_B, both have a self-loop, and pa(S_A) = pa(S_B).
    Lagged relations are always s-orientable.
    Assumes get_parents(S) contains S iff S has a self-loop.
    """
    LAGGED = "lagged"

    def __init__(self, scg: SummaryCausalGraph):
        self._scg = scg

    def is_s_orientable(self, A: DTimeVar, B: DTimeVar) -> Tuple[bool, Union[int, str, None]]:
        """(True, k) with k the first violated condition (0-3 or "lagged"), else (False, None)."""
        nA, nB = A.get_name(), B.get_name()
        if A.get_time() != B.get_time():
            return True, self.LAGGED
        if nA == nB:
            raise ValueError("A and B must be distinct nodes")

        paA, paB = set(self._scg.get_parents(nA)), set(self._scg.get_parents(nB))
        conditions = [
            nB in paA or nA in paB,
            nB in paA and nA in paB,
            nA in paA and nB in paB,
            paA == paB,
        ]
        violated = next((k for k, c in enumerate(conditions) if not c), None)
        return (False, None) if violated is None else (True, violated)


class OrientEdge:
    """
    Time convention: current time = 0, lags negative (DTimeVar(v, -k)).
    Requires window_size >= gamma_max + 1.
    """
    NON_ADJACENT = "non adjacent"
    NOT_S_ORIENTABLE = "not s-orientable"

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
        if oracle and true_ftdag is None:
            raise ValueError("oracle=True requires true_ftdag")
        self._data, self._alpha = data, alpha
        self.scg = scg
        self._CIT = ci_test
        self._oracle, self._true_ftdag = oracle, true_ftdag
        self._gamma_max = scg.get_lag_max()
        self.nb_ci_tests = 0
        self._sor = sOrientability(scg)
        self._de_bar_cache: Dict[Tuple[str, str], Set[str]] = {}
        self._cache: Dict[Tuple[FrozenSet, FrozenSet], bool] = cache if cache is not None else {}

    def is_s_orientable(self, A: DTimeVar, B: DTimeVar) -> Tuple[bool, Union[int, str, None]]:
        return self._sor.is_s_orientable(A, B)

    # ---------------- macro-level helpers ----------------
    def _pa(self, n: str) -> Set[str]:
        return set(self.scg.get_parents(n))

    def _de(self, n: str) -> Set[str]:
        return set(self.scg.get_descendants(n)) | {n}

    def _pa_bar(self, n: str, other: str) -> Set[str]:
        """Parents of S_n in the SCG without the edges between S_n and S_other."""
        return self._pa(n) - {other}

    def _de_bar(self, n: str, other: str) -> Set[str]:
        """Descendants of S_n (n included) in the SCG without the edges between S_n and S_other."""
        key = (n, other)
        if key not in self._de_bar_cache:
            removed = {(n, other), (other, n)}
            seen, stack = {n}, [n]
            while stack:
                u = stack.pop()
                for c in self.scg.get_children(u):
                    if (u, c) not in removed and c not in seen:
                        seen.add(c)
                        stack.append(c)
            self._de_bar_cache[key] = seen
        return self._de_bar_cache[key]

    # ---------------- conditional independence ----------------
    def _independent(self, X: DTimeVar, Y: DTimeVar, cond: List[DTimeVar]) -> bool:
        key = (frozenset((X, Y)), frozenset(cond))
        if key in self._cache:
            return self._cache[key]

        self.nb_ci_tests += 1
        if self._oracle:
            s = self._gamma_max  # data time 0 <-> FT-DAG time gamma_max
            shift = lambda v: DTimeVar(v.get_name(), v.get_time() + s)
            res = d_separated(self._true_ftdag, [shift(X)], [shift(Y)], [shift(v) for v in cond])
        else:
            pval = self._CIT(X, Y, list(cond)).get_pvalue(self._data)
            res = bool(pval > self._alpha) if pval == pval else False  # NaN -> dependent

        self._cache[key] = res
        return res

    # ---------------- candidate sets L, Q, R ----------------
    def _candidate_sets(self, nX: str, nY: str, ell: int) -> Tuple[Set[DTimeVar], Set[str], Set[str]]:
        """
        L: lagged parents, Q: non-descendant instantaneous parents (both always conditioned on),
        R: instantaneous parents that may be descendants (subsets enumerated), R excludes Q.
        X_t is the most recent node, Y_{t-ell} the other one.
        """
        gam = self._gamma_max
        if ell == 0:
            paX, deX = self._pa_bar(nX, nY), self._de_bar(nX, nY)
            paY, deY = self._pa_bar(nY, nX), self._de_bar(nY, nX)
            Q = ((paX - deX) | (paY - deY)) - {nX, nY}
            R = (((paX & deX) | (paY & deY)) - {nX, nY}) - Q
            lagged = self._pa(nX) | self._pa(nY)
            L = {DTimeVar(v, -k) for v in lagged for k in range(1, gam + 1)}
        else:
            paX, deX = self._pa(nX), self._de(nX)
            Q = paX - deX
            R = (paX & deX) - {nX}
            L = {DTimeVar(v, -k) for v in paX for k in range(1, gam + 1)} - {DTimeVar(nY, -ell)}
        return L, Q, R

    # ---------------- FindSepset ----------------
    def _find_sepset(self, X: DTimeVar, Y: DTimeVar) -> Optional[List[DTimeVar]]:
        """Return Z with X _||_ Y | Z, or None if X and Y are adjacent."""
        if X.get_time() < Y.get_time():
            X, Y = Y, X
        ell = X.get_time() - Y.get_time()
        if ell > self._gamma_max:
            raise ValueError(f"lag {ell} > gamma_max={self._gamma_max}: non-adjacent by assumption")

        nX, nY = X.get_name(), Y.get_name()
        X0, Y0 = DTimeVar(nX, 0), DTimeVar(nY, -ell)

        L, Q, R_names = self._candidate_sets(nX, nY, ell)
        fixed = sorted(L | {DTimeVar(v, 0) for v in Q})
        R = sorted(DTimeVar(v, 0) for v in R_names)

        for s in range(len(R) + 1):
            for T in combinations(R, s):
                cond = fixed + list(T)
                if self._independent(X0, Y0, cond):
                    return cond
        return None

    # ---------------- choice of C ----------------
    def _choose_parent(self, nU: str, nV: str) -> str:
        """
        Pick S_C in pa(S_U) \\ pa(S_V) minimizing the cost of OrientFromParent:
        C = V is free, otherwise minimize |R(V, C)|, then |L|+|Q|, then name.
        """
        candidates = self._pa(nU) - self._pa(nV)
        if nV in candidates:
            return nV

        def cost(nC: str) -> Tuple[int, int]:
            L, Q, R = self._candidate_sets(nV, nC, 0)
            return len(R), len(L) + len(Q)

        return min(sorted(candidates), key=cost)

    # ---------------- OrientFromParent ----------------
    def _orient_from_parent(self, U: DTimeVar, V: DTimeVar, C: str) -> str:
        """U_t, V_t adjacent, S_U <-> S_V, S_C in pa(S_U) \\ pa(S_V). Orients U_t - V_t."""
        D = next((DTimeVar(C, -l) for l in range(1, self._gamma_max + 1)
                  if self._find_sepset(U, DTimeVar(C, -l)) is None), None)
        if D is None:
            if C == V.get_name():
                return f"{V} -> {U}"
            D = DTimeVar(C, 0)

        Z = self._find_sepset(V, D)
        if Z is not None and U in Z:
            return f"{U} -> {V}"  # non-collider
        return f"{V} -> {U}"      # collider, or acyclicity if Z is None

    # ---------------- OrientEdge ----------------
    def _run(self, A: DTimeVar, B: DTimeVar) -> str:
        """NON_ADJACENT, an orientation string (most recent node at time 0), or NOT_S_ORIENTABLE."""
        if A.get_time() > B.get_time():
            A, B = B, A
        t = B.get_time()
        A, B = DTimeVar(A.get_name(), A.get_time() - t), DTimeVar(B.get_name(), 0)
        if A == B:
            raise ValueError("A and B must be distinct nodes")
        nA, nB = A.get_name(), B.get_name()
        lag = -A.get_time()

        # Step 1: adjacency
        if lag > self._gamma_max:
            return self.NON_ADJACENT
        scg_edge = nA in self.scg.get_parents(nB) if lag > 0 else nB in self.scg.get_adjacencies(nA)
        if not scg_edge or self._find_sepset(A, B) is not None:
            return self.NON_ADJACENT

        # Step 2: lagged edge
        if lag > 0:
            return f"{A} -> {B}"

        # Step 3: single SCG direction
        paA, paB = self._pa(nA), self._pa(nB)
        if nB not in paA:
            return f"{A} -> {B}"
        if nA not in paB:
            return f"{B} -> {A}"

        # Step 4: S_A <-> S_B
        if paA - paB:
            return self._orient_from_parent(A, B, self._choose_parent(nA, nB))
        if paB - paA:
            return self._orient_from_parent(B, A, self._choose_parent(nB, nA))
        assert not self._sor.is_s_orientable(A, B)[0]
        return self.NOT_S_ORIENTABLE