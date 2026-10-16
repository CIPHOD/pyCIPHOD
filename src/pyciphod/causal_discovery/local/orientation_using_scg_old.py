import warnings
from typing import List, Optional
import pandas as pd
from itertools import combinations

from pyciphod.causal_discovery.basic.constraint_based import ConstraintBased
from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest as FisherZ
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.separation import d_separated


class SCGIncompatibilityWarning(UserWarning):
    """Raised when an expected separating set could not be found, suggesting the
    SCG is (locally) inconsistent with the data. Execution continues by trying
    an alternative candidate (another lag or another conditioning variable)."""
    pass


class SCGTooIncompatibleError(RuntimeError):
    """Raised when no candidate (lag or conditioning variable) could produce a
    valid separating set, making it impossible to orient the edge from the
    SCG/data pair."""
    pass


class OrientInstantaneousEdge:
    def __init__(
        self,
        data: pd.DataFrame,
        A: DTimeVar,
        B: DTimeVar,
        scg: SummaryCausalGraph,
        alpha: float,
        ci_test: CiTests = FisherZ,
        oracle: bool = False,
        true_ftdag = None
    ):
        self._data = data
        self._A = A
        self._B = B
        self._scg = scg
        self._gamma_max = scg.get_lag_max()
        self._ci_test = ci_test
        self._alpha = alpha
        self._variables = scg.get_vertices()
        self.nb_ci_tests = 0
        self._oracle = oracle
        if oracle:
            self._ftdag = true_ftdag

        #print(f"[INIT] A={A}, B={B}, oracle={oracle}, gamma_max={self._gamma_max}")

    def _find_sepset(self, X: DTimeVar, Y: DTimeVar) -> Optional[List[DTimeVar]]:
        #print(f"[_find_sepset] Searching sepset for X={X}, Y={Y}")

        x_name, y_name = X.get_name(), Y.get_name()
        x_time, l = X.get_time(), Y.get_time()

        assert x_time == 0, "X must be evaluated at t (x_time == 0)"
        assert -self._gamma_max <= l <= 0, "Y's lag must be between 0 and gamma_max"

        parents_x = set(self._scg.get_parents(x_name))
        parents_y = set(self._scg.get_parents(y_name))

        if l == 0:
            D_tminus = [
                        DTimeVar(name=v, time=-k)
                        for v in self._variables
                        for k in range(1, self._gamma_max + 1)
                        if v in parents_x | parents_y
                    ]
            
            scg_xy = self._scg.copy()
            scg_xy.remove_directed_edge(x_name, y_name)
            scg_xy.remove_directed_edge(y_name, x_name)

            parents_x_xy = set(scg_xy.get_parents(x_name))
            parents_y_xy = set(scg_xy.get_parents(y_name))
            descendants_x_xy = set(scg_xy.get_descendants(x_name))
            descendants_y_xy = set(scg_xy.get_descendants(y_name))

            q_names = (
                (parents_x_xy - descendants_x_xy) | (parents_y_xy - descendants_y_xy)
            ) - {x_name, y_name}
            r_names = (
                (parents_x_xy & descendants_x_xy) | (parents_y_xy & descendants_y_xy)
            ) - {x_name, y_name}
        else:
            D_tminus = [
                        DTimeVar(name=v, time=-k)
                        for v in self._variables
                        for k in range(1, self._gamma_max + 1)
                        if v in parents_x
                    ]
            
            descendants_x = set(self._scg.get_descendants(x_name))

            q_names = (parents_x - descendants_x) - {x_name, y_name}
            r_names = (parents_x & descendants_x) - {x_name, y_name}

        Q_t = [DTimeVar(name=v, time=0) for v in q_names]
        R_t = [DTimeVar(name=v, time=0) for v in r_names]

        fixed_part = D_tminus + Q_t

        #print(f"[_find_sepset] fixed_part (D_tminus + Q_t) = {fixed_part}")
        #print(f"[_find_sepset] R_t (candidates to combine) = {R_t}")

        for s in range(len(R_t) + 1):
            for Z_subset in combinations(R_t, s):
                candidate_sepset = fixed_part + list(Z_subset)
                #print(f"[_find_sepset] Trying candidate sepset (size={len(candidate_sepset)}): {candidate_sepset}")

                if self._oracle:
                    x_in_ftdag = DTimeVar(X.get_name(),self._gamma_max+X.get_time())
                    y_in_ftdag = DTimeVar(Y.get_name(),self._gamma_max+Y.get_time())
                    candidate_sepset_in_ftag = [
                                                DTimeVar(Z.get_name(), self._gamma_max + Z.get_time())
                                                for Z in candidate_sepset
                                            ]
                    is_sep = d_separated(self._ftdag, [x_in_ftdag], [y_in_ftdag], candidate_sepset_in_ftag)
                    self.nb_ci_tests += 1
                    #print(f"[_find_sepset]   d_separated({x_in_ftdag}, {y_in_ftdag} | {candidate_sepset_in_ftag}) = {is_sep}")
                    if is_sep:
                        #print(f"[_find_sepset] -> FOUND sepset: {candidate_sepset}")
                        return candidate_sepset
                else:
                    test = self._ci_test(X, Y, candidate_sepset)
                    p_value = test.get_pvalue(self._data)
                    self.nb_ci_tests += 1
                    #print(f"[_find_sepset]   CI test p_value = {p_value:.4f} (alpha={self._alpha})")
                    if p_value > self._alpha:
                        #print(f"[_find_sepset] -> FOUND sepset: {candidate_sepset}")
                        return candidate_sepset

        #print(f"[_find_sepset] -> NO sepset found for X={X}, Y={Y}")
        return None

    def _are_adjacent(self, X: DTimeVar, Y: DTimeVar) -> bool:
        adjacent = self._find_sepset(X, Y) is None
        #print(f"[_are_adjacent] {X} <-> {Y} ? {adjacent}")
        return adjacent

    def _resolve_direction_via_lag(
        self,
        probe: DTimeVar,
        far_name: str,
        target: DTimeVar,
        forward: str,
        backward: str,
        context: str,
    ) -> Optional[str]:
        """
        Used for the self-loop tests (case 2). Iterates over lags l, looking for
        a lag at which `probe` is adjacent to `far_name`_{-l}; when found, tries
        to compute a separating set between `target` and `far_name`_{-l} to
        decide the orientation.

        If the expected separating set is not found, a warning is raised and the
        next lag is tried instead of crashing.

        Returns:
            - the resolved direction (forward/backward) if a valid separating
              set was found at some lag,
            - None if no lag showed adjacency at all (legitimate default case,
              caller should apply its own default direction).

        Raises:
            SCGTooIncompatibleError if adjacency was found at at least one lag
            but no valid separating set could be obtained at any of them.
        """
        #print(f"[_resolve_direction_via_lag] context='{context}', probe={probe}, far_name={far_name}, target={target}")

        adjacency_found = False
        for l in range(1, self._gamma_max + 1):
            far_node = DTimeVar(far_name, -l)
            #print(f"[_resolve_direction_via_lag] Testing lag l={l}, far_node={far_node}")

            if self._are_adjacent(probe, far_node):
                adjacency_found = True
                #print(f"ON EST LA, ON VA DONNER A SEPSET : target = {target}, far_node = {far_node}")
                sepset = self._find_sepset(target, far_node)
                if sepset is None:
                    #print(f"[_resolve_direction_via_lag] No sepset found at lag {l}, trying next lag")
                    warnings.warn(
                        f"SCG incompatible with data: no separating set found between "
                        f"{target.get_name()}t and {far_name}t-{l} while performing "
                        f"{context}. Trying another lag.",
                        SCGIncompatibilityWarning,
                    )
                    continue
                direction = forward if probe in sepset else backward
                #print(f"[_resolve_direction_via_lag] -> Resolved direction: {direction} (probe in sepset: {probe in sepset})")
                return direction

        if adjacency_found:
            #print(f"[_resolve_direction_via_lag] -> Adjacency found but no valid sepset at any lag: raising error")
            raise SCGTooIncompatibleError(
                f"SCG too incompatible with data to orient edge "
                f"{self._A.get_name()}-{self._B.get_name()}: exhausted all lags "
                f"while performing {context}."
            )
        #print(f"[_resolve_direction_via_lag] -> No adjacency found at any lag, returning None")
        return None

    def _resolve_direction_via_z(
        self,
        probe: DTimeVar,
        z_name: str,
        target: DTimeVar,
        forward: str,
        backward: str,
        context: str,
    ) -> Optional[str]:
        """
        Used for the parent-asymmetry test (case 3), for a single candidate
        conditioning variable `z_name`. Tries lagged versions of z first, then
        falls back to the instantaneous version.

        If an expected separating set is not found, a warning is raised and the
        method tries the next lag; if the instantaneous fallback also fails, it
        returns None so the caller can try a different z_name.

        Returns:
            - the resolved direction (forward/backward) if successful,
            - None if this z_name is inconclusive (caller should try another z).
        """
        #print(f"[_resolve_direction_via_z] context='{context}', probe={probe}, z_name={z_name}, target={target}")

        for l in range(1, self._gamma_max + 1):
            z_node = DTimeVar(z_name, -l)
            #print(f"[_resolve_direction_via_z] Testing lag l={l}, z_node={z_node}")

            if self._are_adjacent(probe, z_node):
                sepset = self._find_sepset(target, z_node)
                if sepset is None:
                    #print(f"[_resolve_direction_via_z] No sepset found at lag {l}, trying next lag")
                    warnings.warn(
                        f"SCG incompatible with data: no separating set found between "
                        f"{target.get_name()}t and {z_name}t-{l} while orienting via "
                        f"{z_name} ({context}). Trying another lag.",
                        SCGIncompatibilityWarning,
                    )
                    continue
                direction = forward if probe in sepset else backward
                #print(f"[_resolve_direction_via_z] -> Resolved direction via lag {l}: {direction}")
                return direction

        #print(f"[_resolve_direction_via_z] No lagged z resolved anything, trying instantaneous z_node={DTimeVar(z_name, 0)}")

        if self._are_adjacent(target, DTimeVar(z_name, 0)):
            #print(f"[_resolve_direction_via_z] -> target adjacent to instantaneous z: direction={backward}")
            return backward

        sepset = self._find_sepset(target, DTimeVar(z_name, 0))
        if sepset is None:
            #print(f"[_resolve_direction_via_z] -> No sepset found for instantaneous z either, returning None")
            warnings.warn(
                f"SCG incompatible with data: no separating set found between "
                f"{target.get_name()}t and {z_name}t while orienting via {z_name} "
                f"({context}).",
                SCGIncompatibilityWarning,
            )
            return None

        direction = forward if probe in sepset else backward
        #print(f"[_resolve_direction_via_z] -> Resolved direction via instantaneous z: {direction}")
        return direction

    def orient(self) -> str:
        A_t, B_t = self._A, self._B
        a_name, b_name = A_t.get_name(), B_t.get_name()
        parents_a = self._scg.get_parents(a_name)
        parents_b = self._scg.get_parents(b_name)

        #print(f"[orient] Orienting edge between A={A_t} and B={B_t}")
        #print(f"[orient] parents_a={parents_a}, parents_b={parents_b}")

        case = None

        if a_name not in self._scg.get_adjacencies(b_name) or not self._are_adjacent(A_t, B_t):
            case = 0
            #print(f"[orient] -> case {case}: not adjacent")
            return ("not adjacent", case)

        a_to_b = (a_name, b_name) in self._scg.get_directed_edges()
        b_to_a = (b_name, a_name) in self._scg.get_directed_edges()
        #print(f"[orient] a_to_b={a_to_b}, b_to_a={b_to_a}")

        if a_to_b and not b_to_a:
            case = 1
            #print(f"[orient] -> case {case}: {a_name}t -> {b_name}t")
            return (f"{a_name}t -> {b_name}t", case)
        elif b_to_a and not a_to_b:
            case = 1
            #print(f"[orient] -> case {case}: {a_name}t <- {b_name}t")
            return (f"{a_name}t <- {b_name}t", case)

        elif b_to_a and a_to_b:
            if (b_name, b_name) not in self._scg.get_directed_edges():
                case = 2
                #print(f"[orient] -> case {case}: no self-loop on {b_name}, resolving via lag")
                direction = self._resolve_direction_via_lag(
                    probe=self._A,
                    far_name=b_name,
                    target=self._B,
                    forward=f"{a_name}t -> {b_name}t",
                    backward=f"{a_name}t <- {b_name}t",
                    context=f"self-loop test on {b_name}",
                )
                if direction is None:
                    direction = f"{a_name}t <- {b_name}t"
                    #print(f"[orient] -> No direction resolved, using default: {direction}")
                return (direction, case)

            elif (a_name, a_name) not in self._scg.get_directed_edges():
                case = 2
                #print(f"[orient] -> case {case}: no self-loop on {a_name}, resolving via lag")
                direction = self._resolve_direction_via_lag(
                    probe=self._B,
                    far_name=a_name,
                    target=self._A,
                    forward=f"{a_name}t <- {b_name}t",
                    backward=f"{a_name}t -> {b_name}t",
                    context=f"self-loop test on {a_name}",
                )
                if direction is None:
                    direction = f"{a_name}t -> {b_name}t"
                    #print(f"[orient] -> No direction resolved, using default: {direction}")
                return (direction, case)

            elif parents_a != parents_b:
                case = 3
                diff_ab = parents_a - parents_b
                diff_ba = parents_b - parents_a
                #print(f"[orient] -> case {case}: parents differ. diff_ab={diff_ab}, diff_ba={diff_ba}")

                for z_name in diff_ab:
                    #print(f"[orient] Trying z_name={z_name} from diff_ab")
                    direction = self._resolve_direction_via_z(
                        probe=self._A,
                        z_name=z_name,
                        target=self._B,
                        forward=f"{a_name}t -> {b_name}t",
                        backward=f"{a_name}t <- {b_name}t",
                        context=f"parents({a_name}) \\ parents({b_name})",
                    )
                    if direction is not None:
                        #print(f"[orient] -> Resolved via z_name={z_name}: {direction}")
                        return (direction, case)

                for z_name in diff_ba:
                    #print(f"[orient] Trying z_name={z_name} from diff_ba")
                    direction = self._resolve_direction_via_z(
                        probe=self._B,
                        z_name=z_name,
                        target=self._A,
                        forward=f"{a_name}t <- {b_name}t",
                        backward=f"{a_name}t -> {b_name}t",
                        context=f"parents({b_name}) \\ parents({a_name})",
                    )
                    if direction is not None:
                        #print(f"[orient] -> Resolved via z_name={z_name}: {direction}")
                        return (direction, case)

                #print(f"[orient] -> Exhausted all candidates in case 3, raising error")
                raise SCGTooIncompatibleError(
                    f"SCG too incompatible with data to orient edge {a_name}-{b_name}: "
                    f"exhausted all candidate conditioning variables in case 3 "
                    f"(tried {sorted(diff_ab)} and {sorted(diff_ba)})."
                )
            else:
                #print(f"[orient] -> parents_a == parents_b: non s-orientable")
                return ("non s-orientable", case)
            
    
class sOrientability:
    def __init__(self, scg: SummaryCausalGraph, A: DTimeVar, B: DTimeVar):
        self._scg = scg
        self._A = A
        self._B = B

    def classify(self) -> int | None:
        """
        Classifies the pair (A, B) in the SCG.

        Returns:
            0: No edge between A and B.
            1: Exactly one directed edge (already oriented).
            2: Bidirectional edge, but at least one node has no self-loop
               (orientable via the lag test).
            3: Bidirectional edge, both nodes have self-loops, and their
               parent sets differ (orientable via the Z test).
            None: Bidirectional edge, both nodes have self-loops, and their
               parent sets are identical (not s-orientable).
        """
        a, b = self._A.get_name(), self._B.get_name()
        edges = set(self._scg.get_directed_edges())

        ab = (a, b) in edges
        ba = (b, a) in edges

        if not (ab or ba):
            return 0
        if ab != ba:
            return 1
        elif (a, a) not in edges or (b, b) not in edges:
            return 2
        elif self._scg.get_parents(a) != self._scg.get_parents(b):
            return 3
        else:
            return None

    def is_s_orientable(self) -> bool:
        """Return whether the pair is s-orientable."""
        return self.classify() is not None