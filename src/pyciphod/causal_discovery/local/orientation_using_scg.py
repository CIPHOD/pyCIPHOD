

class SCGIncompatibilityWarning(UserWarning):
    """An expected separating set was not found: the SCG may be locally
    inconsistent with the data. Execution continues (with another candidate,
    or by leaving the edge undirected if no candidate remains)."""


# Possible orientations returned by OrientInstantaneousEdge.orient()
NOT_ADJACENT = "not adjacent"
NOT_S_ORIENTABLE = "non s-orientable"  # theoretical case (Theorem): SCG is symmetric


# --------------------------------------------------------------------------- #
# s-orientability                                                             #
# --------------------------------------------------------------------------- #
class sOrientability:
    def __init__(self, scg: SummaryCausalGraph, A: DTimeVar, B: DTimeVar):
        if A == B:
            raise ValueError("A and B must be different variables.")
        self._scg, self._A, self._B = scg, A, B

    def classify(self) -> Optional[int]:
        """
        0: no edge between A and B in the SCG.
        1: exactly one directed edge (already oriented).
        2: bidirectional edge, at least one node without self-loop.
        3: bidirectional edge, both self-loops, different parent sets.
        None: bidirectional edge, both self-loops, same parent sets
              (not s-orientable).
        """
        a, b = self._A.get_name(), self._B.get_name()
        edges = set(self._scg.get_directed_edges())
        ab, ba = (a, b) in edges, (b, a) in edges

        if not (ab or ba):
            return 0
        if ab != ba:
            return 1
        if (a, a) not in edges or (b, b) not in edges:
            return 2
        if set(self._scg.get_parents(a)) != set(self._scg.get_parents(b)):
            return 3
        return None

    def is_s_orientable(self) -> bool:
        return self.classify() is not None


# --------------------------------------------------------------------------- #
# FindSepset (Algorithm FindSepset of the paper)                              #
# --------------------------------------------------------------------------- #
class SepsetFinder:
    """Holds the data / SCG / CI test, the sepset cache and the CI counter.

    One instance can be shared by every orienter working on the same
    data/SCG pair: tests are never repeated and counted only once.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        scg: SummaryCausalGraph,
        alpha: float,
        ci_test: CiTests = FisherZ,
        oracle: bool = False,
        true_ftdag=None,
        cache: Optional[Dict] = None,
    ):
        if oracle and true_ftdag is None:
            raise ValueError("true_ftdag is required when oracle=True.")
        self._data, self._scg = data, scg
        self._gamma_max = scg.get_lag_max()
        self._ci_test, self._alpha = ci_test, alpha
        self._oracle, self._ftdag = oracle, true_ftdag
        self._edges = frozenset(scg.get_directed_edges())
        self._cache: Dict = {} if cache is None else cache
        self.nb_ci_tests = 0

    # ---- primitives ------------------------------------------------------- #
    def independent(self, X: DTimeVar, Y: DTimeVar, cond: List[DTimeVar]) -> bool:
        """One (oracle or statistical) independence test, counted."""
        self.nb_ci_tests += 1
        if self._oracle:
            g = self._gamma_max  # position of time 0 in the FT-DAG
            to_ftdag = lambda v: DTimeVar(v.get_name(), g + v.get_time())
            return d_separated(
                self._ftdag, [to_ftdag(X)], [to_ftdag(Y)], [to_ftdag(z) for z in cond]
            )
        return self._ci_test(X, Y, cond).get_pvalue(self._data) > self._alpha

    @staticmethod
    def _key(X: DTimeVar, Y: DTimeVar):
        kx, ky = (X.get_name(), X.get_time()), (Y.get_name(), Y.get_time())
        # FindSepset is symmetric for instantaneous pairs.
        return tuple(sorted((kx, ky))) if Y.get_time() == 0 else (kx, ky)

    def find_sepset(self, X: DTimeVar, Y: DTimeVar) -> Optional[List[DTimeVar]]:
        """Cached separating set of X_t and Y_{t-l}, or None if adjacent."""
        key = self._key(X, Y)
        if key not in self._cache:
            self._cache[key] = self._search_sepset(X, Y)
        return self._cache[key]

    def are_adjacent(self, X: DTimeVar, Y: DTimeVar) -> bool:
        return self.find_sepset(X, Y) is None

    def _search_sepset(self, X: DTimeVar, Y: DTimeVar) -> Optional[List[DTimeVar]]:
        x, y, lag = X.get_name(), Y.get_name(), Y.get_time()
        if X.get_time() != 0 or not -self._gamma_max <= lag <= 0:
            raise ValueError("X must be at time 0 and Y's lag in [-gamma_max, 0].")

        scg = self._scg
        if lag == 0:
            # Instantaneous case: compute Q and R in the SCG without x-y edges.
            graph = scg.copy()
            for e in ((x, y), (y, x)):
                if e in self._edges:
                    graph.remove_directed_edge(*e)
            ends = (x, y)
            d_parents = set(scg.get_parents(x)) | set(scg.get_parents(y))
            excluded = {x, y}
        else:
            graph, ends, d_parents = scg, (x,), set(scg.get_parents(x))
            excluded = {x}  # Y_t (same series as Y_{t-l}) stays a valid candidate

        q, r = set(), set()
        for n in ends:
            parents = set(graph.get_parents(n))
            desc = set(graph.get_descendants(n)) | {n}  # convention: n in desc(n)
            q |= parents - desc
            r |= parents & desc
        q -= excluded
        r -= excluded
        r -= q  # Q and R are built from a union over X and Y: avoid duplicates

        # L_t, minus Y_{t-l} itself in the lagged case.
        fixed = [DTimeVar(v, -k) for v in sorted(d_parents)
                 for k in range(1, self._gamma_max + 1)
                 if not (lag < 0 and v == y and -k == lag)]
        fixed += [DTimeVar(v, 0) for v in sorted(q)]
        candidates = [DTimeVar(v, 0) for v in sorted(r)]

        for size in range(len(candidates) + 1):
            for Z in combinations(candidates, size):
                cond = fixed + list(Z)
                if self.independent(X, Y, cond):
                    return cond
        return None


# --------------------------------------------------------------------------- #
# Edge orientation (Algorithms OrientEdge / OrientFromParent)                 #
# --------------------------------------------------------------------------- #
class OrientInstantaneousEdge:
    def __init__(
        self,
        data: Optional[pd.DataFrame],
        A: DTimeVar,
        B: DTimeVar,
        scg: SummaryCausalGraph,
        alpha: float = 0.05,
        ci_test: CiTests = FisherZ,
        oracle: bool = False,
        true_ftdag=None,
        cache: Optional[Dict] = None,
        finder: Optional[SepsetFinder] = None,
    ):
        """`finder` (optional) lets several orienters share cache and CI counter.
        Otherwise a private SepsetFinder is built (`cache` can still be shared)."""
        self._A, self._B, self._scg = A, B, scg
        self._gamma_max = scg.get_lag_max()
        self._edges = set(scg.get_directed_edges())
        self._finder = finder or SepsetFinder(
            data, scg, alpha, ci_test, oracle, true_ftdag, cache
        )

    @property
    def nb_ci_tests(self) -> int:
        return self._finder.nb_ci_tests

    @property
    def undirected(self) -> str:
        """Label returned when the edge is left undirected (fallback)."""
        return f"{self._A.get_name()}t - {self._B.get_name()}t"

    # thin wrappers kept for backward compatibility
    def _find_sepset(self, X, Y):
        return self._finder.find_sepset(X, Y)

    def _are_adjacent(self, X, Y):
        return self._finder.are_adjacent(X, Y)

    def _warn_undirected(self, context: str) -> None:
        warnings.warn(
            f"SCG too incompatible with data to orient edge "
            f"{self._A.get_name()}-{self._B.get_name()} ({context}): "
            f"the edge is left undirected.",
            SCGIncompatibilityWarning,
        )

    # ---- orientation helpers ---------------------------------------------- #
    def _probe_lags(self, probe: DTimeVar, far_name: str, target: DTimeVar,
                    context: str) -> Tuple[bool, Optional[bool]]:
        """Scan lags l = 1..gamma_max for `probe` adjacent to far_name_{t-l}.
        At the first such lag with a valid sepset between `target` and
        far_name_{t-l}, return (True, probe in sepset), i.e. whether `probe`
        is a parent of `target`. Returns (adjacency_found, None) otherwise."""
        found = False
        for l in range(1, self._gamma_max + 1):
            far = DTimeVar(far_name, -l)
            if not self._are_adjacent(probe, far):
                continue
            found = True
            sepset = self._find_sepset(target, far)
            if sepset is not None:
                return True, probe in sepset
            warnings.warn(
                f"SCG incompatible with data: no separating set between "
                f"{target.get_name()}t and {far_name}t-{l} ({context}). "
                f"Trying another lag.",
                SCGIncompatibilityWarning,
            )
        return found, None

    def _resolve_via_lag(self, probe_name: str, far_name: str,
                         context: str) -> Tuple[bool, Optional[bool]]:
        """Case 2. `far_name` is the node without self-loop (also the target).
        Returns (found, is_parent):
          - (False, None): no adjacent lag -> instantaneous far -> probe is forced;
          - (True, bool): decided (probe is / is not a parent of the target);
          - (True, None): adjacent lag(s) but no valid sepset -> undecided."""
        return self._probe_lags(
            DTimeVar(probe_name, 0), far_name, DTimeVar(far_name, 0), context
        )

    def _resolve_via_z(self, probe_name: str, z_name: str, target_name: str,
                       context: str) -> Optional[bool]:
        """Case 3, for one candidate z. Lagged copies of z first, then Z_t.
        Returns True/False (probe is/is not a parent of target), or None if
        this candidate is inconclusive (an adjacent lag without sepset)."""
        probe, target = DTimeVar(probe_name, 0), DTimeVar(target_name, 0)
        found, is_parent = self._probe_lags(probe, z_name, target, context)
        if is_parent is not None:
            return is_parent
        if found:
            # Fact 4 of the paper requires that NO lagged copy is adjacent.
            return None
        sepset = self._find_sepset(target, DTimeVar(z_name, 0))
        if sepset is None:  # target adjacent to Z_t: only target -> probe is consistent
            return False
        return probe in sepset

    def _orient_lagged(self) -> Tuple[str, int]:
        """Steps 1-2 of OrientEdge: A_{t'} and B_t with t' != t."""
        A, B = self._A, self._B
        a, b = A.get_name(), B.get_name()
        a_first = A.get_time() < B.get_time()
        early, late = (A, B) if a_first else (B, A)
        lag = early.get_time() - late.get_time()  # < 0
        if -lag > self._gamma_max:
            return NOT_ADJACENT, -1
        if not self._are_adjacent(DTimeVar(late.get_name(), 0),
                                  DTimeVar(early.get_name(), lag)):
            return NOT_ADJACENT, -1
        return (f"{a}t -> {b}t" if a_first else f"{a}t <- {b}t"), -1

    # ---- main entry point ------------------------------------------------- #
    def orient(self) -> Tuple[str, Optional[int]]:
        """Returns (orientation, case). Orientation is one of:
          'not adjacent', 'At -> Bt', 'At <- Bt',
          'At - Bt'          (undirected: no candidate gave a valid sepset,
                              a SCGIncompatibilityWarning is emitted),
          'non s-orientable' (theoretical case: SCG symmetric in A and B).
        Case is -1 for lagged pairs."""
        a, b = self._A.get_name(), self._B.get_name()
        ta, tb = self._A.get_time(), self._B.get_time()
        if ta != tb:
            return self._orient_lagged()
        if ta != 0:
            raise ValueError("Instantaneous pairs must be given at time 0.")

        case = sOrientability(self._scg, self._A, self._B).classify()

        if case == 0 or not self._are_adjacent(self._A, self._B):
            return NOT_ADJACENT, case

        def arrow(parent: str) -> str:
            return f"{a}t -> {b}t" if parent == a else f"{a}t <- {b}t"

        edges = self._edges

        if case == 1:
            return arrow(a if (a, b) in edges else b), case

        if case == 2:
            # v: node without self-loop (target), u: the other one (probe).
            u, v = (a, b) if (b, b) not in edges else (b, a)
            context = f"self-loop test on {v}"
            found, is_parent = self._resolve_via_lag(u, v, context)
            if found and is_parent is None:
                self._warn_undirected(f"exhausted all lags, {context}")
                return self.undirected, case
            # No adjacent lag: well-specification forces the instantaneous v -> u.
            return arrow(u if is_parent else v), case

        if case == 3:
            pa, pb = set(self._scg.get_parents(a)), set(self._scg.get_parents(b))
            for probe, target, diff in ((a, b, pa - pb), (b, a, pb - pa)):
                for z in sorted(diff):
                    is_parent = self._resolve_via_z(
                        probe, z, target, f"parents({probe}) \\ parents({target})"
                    )
                    if is_parent is not None:
                        return arrow(probe if is_parent else target), case
            self._warn_undirected("exhausted all candidate conditioning variables in case 3")
            return self.undirected, case

        return NOT_S_ORIENTABLE, case
    
    
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