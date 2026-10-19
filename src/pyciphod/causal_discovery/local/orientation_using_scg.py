"""Direct implementation of the pseudo-codes FindSepset, OrientFromParent, OrientEdge.

Conventions
-----------
- Nodes are DTimeVar(name, time) with time <= 0: X_t -> DTimeVar("X", 0),
  Y_{t-l} -> DTimeVar("Y", -l).
- `ci(X, Y, cond)` is any conditional independence test returning True when
  X is independent of Y given `cond` (a list of DTimeVar).
- Orientations are returned relative to the order of the arguments:
  "->" means first node -> second node, "<-" the opposite.
"""
from itertools import combinations

from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.separation import d_separated

NOT_ADJACENT = "not adjacent"
NOT_S_ORIENTABLE = "not s-orientable"


class OracleCI:
    """Oracle CI test: d-separation in the true FT-DAG.
    Assumes time 0 (present) sits at time index `gamma_max` in `ft_dag`."""

    def __init__(self, ft_dag, gamma_max):
        self.ft_dag, self.gamma_max = ft_dag, gamma_max

    def _shift(self, v):
        return DTimeVar(v.get_name(), self.gamma_max + v.get_time())

    def __call__(self, X, Y, cond):
        return d_separated(
            self.ft_dag, [self._shift(X)], [self._shift(Y)], [self._shift(z) for z in cond]
        )


class SCGOrienter:
    def __init__(self, scg, ci):
        self.scg = scg
        self.ci = ci
        self.gamma_max = scg.get_lag_max()

    # ----------------------------------------------------------------------- #
    # Algorithm FindSepset                                                    #
    # ----------------------------------------------------------------------- #
    def find_sepset(self, X, Y):
        """Separating set of X_t and Y_{t-l} (X at time 0, 0 <= l <= gamma_max).
        Returns a list of nodes, or None if X and Y are adjacent."""
        scg = self.scg
        x, y, lag = X.get_name(), Y.get_name(), -Y.get_time()

        if lag == 0:
            # Q and R are computed in the SCG without the edges between x and y
            scg_xy = scg.copy()
            for edge in [(x, y), (y, x)]:
                if edge in scg.get_directed_edges():
                    scg_xy.remove_directed_edge(*edge)
            series = [x, y]
            lagged_series = set(scg.get_parents(x)) | set(scg.get_parents(y))
            excluded = {x, y}
        else:
            scg_xy = scg
            series = [x]
            lagged_series = set(scg.get_parents(x))
            excluded = {x}

        Q, R = set(), set()
        for s in series:
            parents = set(scg_xy.get_parents(s))
            descendants = set(scg_xy.get_descendants(s)) | {s}  # s is its own descendant
            Q |= parents - descendants               # instantaneous parents, non-descendants
            R |= (parents & descendants) - excluded  # instantaneous parents, descendants
        R -= Q

        # L: lagged copies of the parents (all lags 1..gamma_max), except Y_{t-l} itself
        L = [DTimeVar(v, -k)
             for v in sorted(lagged_series)
             for k in range(1, self.gamma_max + 1)
             if not (v == y and k == lag)]
        always = L + [DTimeVar(v, 0) for v in sorted(Q)]

        # try every subset T of R, smallest first
        for size in range(len(R) + 1):
            for T in combinations(sorted(R), size):
                cond = always + [DTimeVar(v, 0) for v in T]
                if self.ci(X, Y, cond):
                    return cond
        return None  # no separating set: adjacent

    # ----------------------------------------------------------------------- #
    # Algorithm OrientFromParent                                              #
    # ----------------------------------------------------------------------- #
    def orient_from_parent(self, U, V, c):
        """U, V: nodes at time 0 with U <-> V in the SCG; c: a series with
        c in parents(U) \\ parents(V). Returns "->" (U_t -> V_t) or "<-" (U_t <- V_t)."""
        # D: a past copy of C adjacent to U_t (then C_{t-l} -> U_t and C_{t-l} not adjacent to V_t)
        D = None
        for lag in range(1, self.gamma_max + 1):
            if self.find_sepset(U, DTimeVar(c, -lag)) is None:
                D = DTimeVar(c, -lag)
                break

        # otherwise C_t -> U_t
        if D is None:
            if c == V.get_name():
                return "<-"
            D = DTimeVar(c, 0)

        Z = self.find_sepset(V, D)
        if Z is not None and U in Z:
            return "->"  # U_t is a non-collider on D - U_t - V_t
        return "<-"      # collider (or D adjacent to V_t: V_t -> D -> U_t)

    # ----------------------------------------------------------------------- #
    # Algorithm OrientEdge                                                    #
    # ----------------------------------------------------------------------- #
    def orient_edge(self, A, B):
        """A_{t'} and B_t, with B at time 0 and A at time <= 0.
        Returns NOT_ADJACENT, "->" (A -> B), "<-" (A <- B) or NOT_S_ORIENTABLE."""
        scg = self.scg
        a, b = A.get_name(), B.get_name()
        lag = -A.get_time()

        # Step 1: adjacency
        if lag > self.gamma_max:
            return NOT_ADJACENT
        if self.find_sepset(B, A) is not None:
            return NOT_ADJACENT

        # Step 2: lagged edge, oriented by time
        if lag > 0:
            return "->"

        # Step 3: instantaneous edge, only one direction in the SCG
        if b not in scg.get_parents(a):
            return "->"
        if a not in scg.get_parents(b):
            return "<-"

        # Step 4: A <-> B in the SCG, look for a parent of one that is not a parent of the other
        only_a = sorted(set(scg.get_parents(a)) - set(scg.get_parents(b)))
        only_b = sorted(set(scg.get_parents(b)) - set(scg.get_parents(a)))
        if only_a:
            return self.orient_from_parent(A, B, only_a[0])
        if only_b:
            result = self.orient_from_parent(B, A, only_b[0])  # orientation of B vs A
            return "<-" if result == "->" else "->"
        return NOT_S_ORIENTABLE
    
    
# --------------------------------------------------------------------------- #
# s-identifiability                                                           #
# --------------------------------------------------------------------------- #
# class sIdentifiability:
#     """Adjustment set for an "ATE" or "CDE" query.

#     The treatment is at treatment.get_time() <= 0 and the outcome at t = 0.
#     By stationarity, everything is oriented at t = 0 and then shifted by
#     `shift` (treatment time for ATE, 0 for CDE).

#     After `get_adj_set()`, `effect_is_zero` is True when the total effect is
#     trivially 0 (outcome not a descendant of the treatment in the SCG, or
#     Y_t -> X_t); the returned set is then empty and should not be used.
#     `get_adj_set()` returns None when an edge relevant to the adjustment set
#     stays undirected (not s-orientable, or SCG incompatible with the data).
#     """

#     def __init__(self, data, treatment, outcome, query, scg, alpha,
#                  ci_test=FisherZTest, oracle=False, true_ftdag=None):
#         if query not in ("ATE", "CDE"):
#             raise ValueError(f"query must be 'ATE' or 'CDE', got {query!r}.")
#         self._treatment, self._outcome, self._query = treatment, outcome, query
#         self._scg = scg
#         self._gamma_max = scg.get_lag_max()
#         self._finder = SepsetFinder(data, scg, alpha, ci_test, oracle, true_ftdag)
#         self.effect_is_zero = False

#     @property
#     def nb_ci_tests(self) -> int:
#         return self._finder.nb_ci_tests

#     def _orienter(self, a: str, b: str) -> OrientInstantaneousEdge:
#         return OrientInstantaneousEdge(
#             None, DTimeVar(a, 0), DTimeVar(b, 0), self._scg, finder=self._finder
#         )

#     def _are_adjacent(self, a: str, b: str) -> bool:
#         return self._finder.are_adjacent(DTimeVar(a, 0), DTimeVar(b, 0))

#     def _find_sepset(self, a: str, b: str):
#         return self._finder.find_sepset(DTimeVar(a, 0), DTimeVar(b, 0))

#     def _instantiate(self) -> Tuple[str, int, Set[str], Set[DTimeVar]]:
#         """Returns (target, shift, R, N): R = candidate names to orient,
#         N = safe set (L ∪ Q) as time-shifted DTimeVar."""
#         x = self._treatment.get_name()
#         if self._query == "ATE":
#             target, shift = x, self._treatment.get_time()
#         else:
#             target, shift = self._outcome.get_name(), 0

#         parents = set(self._scg.get_parents(target))
#         # Q(T_u), R(T_u) use the descendants of the TARGET T (treatment for ATE,
#         # outcome for CDE), with the convention T in desc(T).
#         desc_t = set(self._scg.get_descendants(target)) | {target}
#         excluded = {target} | ({x} if self._treatment.get_time() == 0 else set())

#         L = {DTimeVar(v, shift - k) for v in parents
#              for k in range(1, self._gamma_max + 1)}
#         Q = {DTimeVar(v, shift) for v in parents - desc_t - excluded}
#         R = (parents & desc_t) - excluded
#         return target, shift, R, L | Q

#     def _batch_orient_colliders(self, target: str, A: Set[str]) -> Tuple[Set[str], Set[str]]:
#         """Non-adjacent pairs (u, v) in A with target outside their sepset form
#         a v-structure u -> target <- v. Returns (parents found, undecided)."""
#         P, U = set(), set(A)
#         adj = {n: set(self._scg.get_adjacencies(n)) for n in A}
#         for u, v in combinations(sorted(A), 2):
#             if u not in U or v not in U or v in adj[u]:
#                 continue
#             sepset = self._find_sepset(u, v)  # not None: u_t, v_t non-adjacent
#             if sepset is not None and DTimeVar(target, 0) not in sepset:
#                 P |= {u, v}
#                 U -= {u, v}
#         return P, U

#     def get_adj_set(self) -> Optional[Set[DTimeVar]]:
#         """Valid adjustment set, or None if an edge could not be oriented
#         (non s-orientable pair, or undirected fallback)."""
#         self.effect_is_zero = False
#         x, y = self._treatment.get_name(), self._outcome.get_name()

#         if self._query == "ATE":
#             desc_x = set(self._scg.get_descendants(x)) | {x}
#             if y not in desc_x:  # Y not a descendant of X in the SCG
#                 self.effect_is_zero = True
#                 return set()

#         target, shift, R, N = self._instantiate()
#         P: Set[str] = set()  # identified parents of target
#         A: Set[str] = set()  # adjacent, direction still ambiguous

#         for r in sorted(R):
#             if self._are_adjacent(target, r):
#                 # The SCG forces r -> target unless target is also a parent of r.
#                 (A if target in self._scg.get_parents(r) else P).add(r)

#         P_col, U = self._batch_orient_colliders(target, A)
#         P |= P_col

#         for r in sorted(U):
#             orientation, _ = self._orienter(target, r).orient()
#             if orientation == f"{target}t <- {r}t":
#                 P.add(r)
#             elif orientation == f"{target}t -> {r}t":
#                 pass  # r is a child of target: not in the adjustment set
#             elif orientation == NOT_ADJACENT:
#                 pass
#             else:
#                 # NOT_S_ORIENTABLE, or undirected fallback "Tt - rt":
#                 # we cannot tell whether r is a parent of target.
#                 return None

#         # Total effect: Y_t -> X_t means the effect is 0.
#         if self._query == "ATE" and shift == 0 and y in P:
#             self.effect_is_zero = True
#             return set()

#         return (N | {DTimeVar(p, shift) for p in P}) - {self._treatment}