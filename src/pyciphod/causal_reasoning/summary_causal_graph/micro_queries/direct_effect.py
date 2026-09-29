from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from typing import Hashable
from pyciphod.utils.time_series.data_format import DTimeVar


def linear_direct_effect_is_identifiable(
    g: SummaryCausalGraph,
    x,
    y,
    gamma: int,
    max_delay: int,
):
    """
    Test graphical identifiability of the linear direct effect

        alpha_{X_{t-gamma}, Y_t}

    from a Summary Causal Graph according to Theorem 1 of
    Ferreira and Assaad (AAAI 2024) and its extension in Assaad et al (2027)

    If identifiable, also return valid finite adjustment sets:

        - Ferreira & Assaad (2024), Definition 11
        - Ferreira & Assaad (2024), Definition 12
        - Assaad, Ez-Zejjari & Zan (2023), Definition 11,
          when the SCG is acyclic except possibly for self-loops.

    Parameters
    ----------
    g : SummaryCausalGraph
        Summary causal graph.

    x : Hashable
        Exposure macro-variable X.

    y : Hashable
        Outcome macro-variable Y.

    gamma : int
        Lag of the micro direct effect:
            X_{t-gamma} -> Y_t

    max_delay : int
        Maximum causal lag gamma_max.

    Returns
    -------
    identifiable : bool
        True if the direct effect is identifiable.

    adjustment_sets : dict or None
        None if the effect is non-identifiable.

        Otherwise, a dictionary containing the available
        finite adjustment sets.

    Notes
    -----
    Ferreira and Assaad (2024), Theorem 1 assumes x != y.
    """

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------

    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    if max_delay < 0:
        raise ValueError("max_delay must be non-negative.")

    if gamma > max_delay:
        raise ValueError(
            "gamma cannot be greater than max_delay."
        )

    vertices = g.get_vertices()

    if x not in vertices:
        raise ValueError(f"{x!r} is not a vertex of the SCG.")

    if y not in vertices:
        raise ValueError(f"{y!r} is not a vertex of the SCG.")

    directed_edges = g.get_directed_edges()

    descendants_y = g.get_descendants(y)

    # TODO This was not proven to be true in Ferreira and Assaad (2024), Theorem 1.
    # ---------------------------------------------------------
    # Lagged self-direct effect:
    #
    #     Y_{t-gamma} -> Y_t
    #
    # Ferreira and Assaad (2024), Theorem 1 assumes X != Y.
    # The case X == Y is handled by
    # Lemma~\ref{lemma:linear_self_direct_effect_identifiability}.
    #
    # For gamma > 0, the effect is non-identifiable iff
    #   1. Y has a self-loop, and
    #   2. SCC(Y) != {Y}.
    #
    # If Y has no self-loop, the effect is structurally zero.
    # ---------------------------------------------------------

    if x == y:
        # For gamma = 0, exposure and outcome are the same
        # micro variable, so this is not a direct-effect query.
        if gamma == 0:
            raise ValueError(
                "When x == y, gamma must be strictly positive: "
                "Y_t cannot be its own micro direct-effect exposure."
            )

        # Check that the macro variable exists before using
        # graph operations below.
        if y not in g.get_vertices():
            raise ValueError(f"{y!r} is not a vertex of the SCG.")

        # No self-loop means that no lagged micro edge
        #
        #     Y_{t-gamma} -> Y_t
        #
        # can occur in any compatible FT-DAG. Hence the
        # corresponding coefficient is structurally zero.
        if (y, y) not in directed_edges:
            return True, {
                "structural_zero": set()
            }

        # SCC(Y) != {Y} iff Y belongs to a directed cycle
        # containing at least one other macro variable.
        # y_in_nontrivial_scc = any(
        #     y in set(cycle) and len(set(cycle)) > 1
        #     for cycle in g.get_all_simple_cycles()
        # )
        #
        # if y_in_nontrivial_scc:
        #     return False, None
        scc_y = g.get_strongly_connected_components(y)

        if scc_y != {y}:
            return False, None

        # Otherwise:
        #
        #     SCC(Y) = {Y}
        #
        # and the lagged self-direct effect is identifiable.
        # Continue below to construct the adjustment sets.

    else:
        # ---------------------------------------------------------
        # If X -> Y is absent, the corresponding direct effect
        # is structurally zero.
        # ---------------------------------------------------------

        if x not in g.get_parents(y):
            return True, {
                "structural_zero": set()
            }


        # =========================================================
        # CONDITION 1
        #
        # X in Desc(Y)
        # and there exists a cycle containing X but not Y.
        # =========================================================

        if x in descendants_y:

            for cycle in g.get_all_simple_cycles():

                cycle_vertices = set(cycle)

                if (
                    x in cycle_vertices
                    and y not in cycle_vertices
                ):
                    return False, None

        # =========================================================
        # CONDITION 2
        # =========================================================

        # ---------------------------------------------------------
        # Special two-node non-direct path:
        #
        #       X <-> Y
        #
        # represented in the SCG by X -> Y and Y -> X.
        # For gamma = 0 this makes the effect non-identifiable.
        # ---------------------------------------------------------

        if gamma == 0 and (y, x) in directed_edges:
            return False, None

        # ---------------------------------------------------------
        # Search active non-direct paths X ... Y
        # ---------------------------------------------------------

        active_paths = g.get_active_paths(
            x,
            y,
            allowed_nodes=None,
            max_path_length=len(vertices),
        )

        for path in active_paths:

            # [X, Y] corresponds to the direct path.
            # The reciprocal two-node case was handled above.
            if len(path) < 3:
                continue

            internal_vertices = set(path[1:-1])

            # Every internal node must be a descendant of Y.
            if not internal_vertices.issubset(descendants_y):
                continue

            # -----------------------------------------------------
            # Condition 2a: gamma = 0
            # -----------------------------------------------------

            if gamma == 0:
                return False, None

            # -----------------------------------------------------
            # Condition 2b: gamma > 0
            #
            # No step of the path can be
            #
            #       Vi <- V{i+1}
            #
            # Equivalently each pair must contain the forward
            # directed edge Vi -> V{i+1}.
            #
            # Reciprocal edges are therefore also accepted.
            # -----------------------------------------------------

            forward_path = True

            for i in range(len(path) - 1):

                vi = path[i]
                vj = path[i + 1]

                if (vi, vj) not in directed_edges:
                    forward_path = False
                    break

            if forward_path:
                return False, None

    # =========================================================
    # IDENTIFIABLE:
    # construct the finite adjustment sets.
    # =========================================================

    adjustment_sets = {}

    # In your lagged dataframe convention:
    #
    # DTimeVar(X, 0)  = X_t
    # DTimeVar(X, -1) = X_{t-1}
    # etc.
    exposure = DTimeVar(x, -gamma)

    # =========================================================
    # Ferreira & Assaad 2024 - Definition 11
    #
    # Z = D_<t U A_<=t
    #
    # D_<t:
    #   descendants of Y from t-max_delay to t-1
    #
    # A_<=t:
    #   non-descendants of Y from t-max_delay to t
    #
    # excluding the exposure X_{t-gamma}.
    # =========================================================

    z_def11 = set()

    # D_<t
    for v in descendants_y:

        for lag in range(1, max_delay + 1):
            z_def11.add(
                DTimeVar(v, -lag)
            )

    # A_<=t
    non_descendants_y = g.get_non_descendants(y)

    for v in non_descendants_y:

        for lag in range(max_delay + 1):
            z_def11.add(
                DTimeVar(v, -lag)
            )

    z_def11.discard(exposure)

    adjustment_sets[
        "ferreira_assaad_2024_def11"
    ] = z_def11

    # =========================================================
    # Ferreira & Assaad 2024 - Definition 12
    #
    # Z =
    #   D_<t^{Anc(Y)}
    #       U
    #   A_<=t^{Anc(Y)}
    #
    # where
    #
    # D component:
    #   Anc(Y) intersection Desc(Y), strictly before t
    #
    # A component:
    #   Anc(Y) \ Desc(Y), up to and including t.
    # =========================================================

    ancestors_y = g.get_ancestors(y)

    z_def12 = set()

    ancestors_descendants_y = (
        ancestors_y & descendants_y
    )

    ancestors_non_descendants_y = (
        ancestors_y - descendants_y
    )

    # Anc(Y) intersection Desc(Y), strictly before t
    for v in ancestors_descendants_y:

        for lag in range(1, max_delay + 1):
            z_def12.add(
                DTimeVar(v, -lag)
            )

    # Anc(Y) \ Desc(Y), up to and including t
    for v in ancestors_non_descendants_y:

        for lag in range(max_delay + 1):
            z_def12.add(
                DTimeVar(v, -lag)
            )

    z_def12.discard(exposure)

    adjustment_sets[
        "ferreira_assaad_2024_def12"
    ] = z_def12

    # =========================================================
    # Assaad, Ez-Zejjari & Zan 2023 - Definition 11
    #
    # This result applies to an acyclic summary causal graph
    # with loops (ASCGL): after ignoring self-loops, there
    # must be no directed macro cycle.
    # =========================================================

    cycles = g.get_all_simple_cycles()

    acyclic_except_self_loops = all(
        len(cycle) == 1
        for cycle in cycles
    )

    if acyclic_except_self_loops:

        z_assaad_2023 = set()

        # -----------------------------------------------------
        # B = Parents(Y) \ {X, Y}
        #
        # Add:
        # B_t, ..., B_{t-max_delay}
        # -----------------------------------------------------

        B = g.get_parents(y) - {x, y}

        for b in B:

            for lag in range(max_delay + 1):
                z_assaad_2023.add(
                    DTimeVar(b, -lag)
                )

        # -----------------------------------------------------
        # X =
        #
        # {X_t, ..., X_{t-max_delay}}
        #       \ {X_{t-gamma}}
        # -----------------------------------------------------

        for lag in range(max_delay + 1):

            z_assaad_2023.add(
                DTimeVar(x, -lag)
            )

        # -----------------------------------------------------
        # Y =
        #
        # {Y_{t-1}, ..., Y_{t-max_delay}}
        #
        # only if Y has a self-loop.
        # -----------------------------------------------------

        if (y, y) in directed_edges:
            for lag in range(1, max_delay + 1):
                z_assaad_2023.add(
                    DTimeVar(y, -lag)
                )

        z_assaad_2023.discard(exposure)
        z_assaad_2023.discard(DTimeVar(y, 0))

        adjustment_sets[
            "assaad_ezzejjari_zan_2023_def11"
        ] = z_assaad_2023

    return True, adjustment_sets


def CDE_is_identifiable(Gs: SummaryCausalGraph, X : Hashable, Y : Hashable, gamma : int) -> bool:
    '''
    Implements Theorem 1 of "Average Controlled and Average Natural Micro Direct Effects in Summary
    Causal Graphs" by Ferreira and Assaad UAI 2026.
    
    :param Gs: Summary Causal Graph of interest
    :type summary: SummaryCausalGraph
    :param X: Treatment of interest
    :type X: Hashable
    :param Y: Outcome of interest
    :type Y: Hashable
    :param gamma: Lag of interest
    :type gamma: int
    :return: Boolean describing whether the Controlled Direct Effect is identifiable or not
    :rtype: bool
    '''
    
    if DTimeVar(X,gamma) not in Gs.get_possible_parents(Y,0):
        return True
    scc_Y = Gs.get_strongly_connected_components(Y)
    if len(scc_Y) > 1:
        return False
    confounded_adjacencies_Y = Gs.get_confounded_adjacencies(Y)
    ancestors_Y = Gs.get_ancestors(Y)
    if bool(set.intersection(confounded_adjacencies_Y, ancestors_Y)):
        return False
    return True

def NDE_is_identifiable(Gs: SummaryCausalGraph, X : Hashable, Y : Hashable, gamma : int) -> bool:
    '''
    Implements Theorem 2 of "Average Controlled and Average Natural Micro Direct Effects in Summary
    Causal Graphs" by Ferreira and Assaad UAI 2026.
    
    :param Gs: Summary Causal Graph of interest
    :type summary: SummaryCausalGraph
    :param X: Treatment of interest
    :type X: Hashable
    :param Y: Outcome of interest
    :type Y: Hashable
    :param gamma: Lag of interest
    :type gamma: int
    :return: Boolean describing whether the Natural Direct Effect is identifiable or not
    :rtype: bool
    '''

    if DTimeVar(X,gamma) not in Gs.get_possible_parents(Y,0):
        return True
    scc_Y = Gs.get_strongly_connected_components(Y)
    if len(scc_Y) > 1:
        return False
    scc_X = Gs.get_strongly_connected_components(X)
    if len(scc_X) > 1:
        return False
    pp_X = Gs.get_possible_parents(X, gamma)
    pp_Y = Gs.get_possible_parents(Y, 0)
    if bool(set.intersection(pp_X, pp_Y)):
        return False
    confounded_adjacencies_Y = Gs.get_confounded_adjacencies(Y)
    ancestors_Y = Gs.get_ancestors(Y)
    if bool(set.intersection(confounded_adjacencies_Y, ancestors_Y)):
        return False
    confounded_adjacencies_X = Gs.get_confounded_adjacencies(X)
    if bool(set.intersection(confounded_adjacencies_X, ancestors_Y)):
        return False
    return True


# def is_active(sg, path, adjustment_set = set()):
#     """
#     Determines whether a path is active in a (cyclic) summary causal graph sg when adjusting on adjustment_set.
#     :param sg: networkx directed graph
#     :param path: List nodes
#     :param adjustment_set: Set nodes
#     :return: bool
#     """
#     assert is_subset(path, sg.nodes)
#     assert is_subset(adjustment_set, sg.nodes)
#
#     colliders = set()
#     has_seen_right_arrow = False
#     for i in range(len(path) - 1):
#         if (sg.has_edge(path[i], path[i + 1])) and (sg.has_edge(path[i + 1], path[i])):     #V^i <-> V^{i+1}
#             pass
#         elif (sg.has_edge(path[i], path[i + 1])) and (not sg.has_edge(path[i + 1], path[i])): #V^i  -> V^{i+1}
#             if (i>0) and (path[i] in adjustment_set):
#                 return False
#             has_seen_right_arrow = True
#         elif (not sg.has_edge(path[i], path[i + 1])) and (sg.has_edge(path[i + 1], path[i])): #V^i <-  V^{i+1}
#             if has_seen_right_arrow:
#                 colliders.add(path[i])
#                 has_seen_right_arrow = False
#         else:                                                                           #V^i     V^{i+1}
#             raise ValueError("Path is not a path in sg: ", path[i], " and ", path[i + 1], " are not connected.")
#     for c in colliders:
#         if c in adjustment_set:
#             continue
#         for d in nx.descendants(sg, c):
#             if d in adjustment_set:
#                 break
#         else:
#             return False
#     return True
#
#
# def is_subset(small_iterable, big_iterable):
#     """
#     Determines whether small_iterable is a subset of big_iterable.
#     :param small_iterable: iterable nodes
#     :param big_iterable: iterable nodes
#     :return: bool
#     """
#
#     for v in small_iterable:
#         if v not in big_iterable:
#             return False
#     return True
#
#
# def is_non_direct(sg, path):
#     """
#     Determines whether path is non-direct. (path != X->Y ?)
#     :param sg: networkx directed graph
#     :param path: List nodes
#     :return: bool
#     """
#     assert is_subset(path, sg.nodes)
#     assert len(path) >= 2
#     assert path[0] != path[-1]
#
#     if len(path) > 2:
#         return True
#
#     x = path[0]
#     y = path[-1]
#     if sg.has_edge(y, x):
#         return True
#     if not sg.has_edge(x, y):
#         return False
#
#     return True
#
#
# # -----------------Code of paper {identifiability of direct from summary causal graphs}------------------------
#
# def is_identifiable_with_cycles(sg, x, y, gamma_xy, gamma_max = 1):
#     """
#     Determines whether the direct effect from x to y with lag gamma_xy is identifiable
#     for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
#     This corresponds to Theorem 1 in {identifiability of direct from summary causal graphs}.
#     :param sg: networkx directed graph
#     :param gamma_max: int
#     :param x: node
#     :param y: node
#     :param gamma_xy: int
#     :return: bool
#     """
#     assert x != y
#     assert (x in sg.nodes) and (y in sg.nodes)
#     #todo Charles: if  gamma_xy>gamma_max then identifiable and =0
#     assert (0 <= gamma_xy) and (gamma_xy <= gamma_max)
#
# # This code has exactly the same structure as Theorem 1 in {identifiability of direct effect from summary causal graphs}
#     if not sg.has_edge(x, y):
#         return True
#
#     undirected_sg = sg.to_undirected()
#     # Removed the edge X->Y from all_simple_paths except if Y->X is in SCG
#     all_simple_paths = list(nx.all_simple_paths(undirected_sg, x, y))
#     if (y, x) not in sg.edges:
#         all_simple_paths.remove([x, y])
#     for path in all_simple_paths:
#         if is_active(sg, path) and is_subset(path[1:-1], nx.descendants(sg, y)):
#             if gamma_xy == 0 and is_non_direct(sg, path):
#                 return False
#             if gamma_xy > 0:
#                 n = len(path)
#                 if n > 2:
#                     path_has_left_arrow = False
#                     for i in range(n - 1):
#                         if (not sg.has_edge(path[i], path[i + 1])) and sg.has_edge(path[i + 1], path[i]):
#                             path_has_left_arrow = True
#                             break
#                     if not path_has_left_arrow:
#                         return False
#                 if n == 2:
#                     if x in nx.descendants(sg,y):
#                         print("Error: cycle_basis is not not implemented for directed graphs")
#                         #todo Charles: networkx.exception.NetworkXNotImplemented: not implemented for directed type
#                         cycles_x = nx.cycle_basis(sg, x)
#                         for c in cycles_x:
#                             if not (y in c):
#                                 return False
#     return True
#
#
# def huge_adjustment_set(sg, x, y, gamma_xy, gamma_max = 1):
#     """
#     Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if it is identifiable
#     for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
#     Otherwise fails.
#     This corresponds to Corollary 1 in {identifiability of direct from summary causal graphs}.
#     The returned value of this function (x,S) can be interpreted as follows:
#     if x != None:
#         x is the direct effect.
#     else x == None:
#         The direct effect is can be estimated using the adjustment set S.
#         S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.
#
#     :param sg: networkx directed graph
#     :param gamma_max: int
#     :param x: node
#     :param y: node
#     :param gamma_xy: int
#     :return: (float, Set (nodes, int))
#     """
#
#     assert x != y
#     assert (x in sg.nodes) and (y in sg.nodes)
#     # There is no gamma_min_xy
#     # assert (0 <= gamma_min_xy) and (gamma_min_xy <= gamma_max)
#
#     if not sg.has_edge(x, y):
#         return (0,set())
#
#     # The function "is_identifiable_with_cycles" was called with wrong parameters
#     # if not is_identifiable_with_cycles(sg, gamma_max, x, y, 0):
#     if not is_identifiable_with_cycles(sg, x, y, 0, gamma_max):
#         raise ValueError("Direct effect of " + str(x) + "_t-" + gamma_xy + " on " + str(y) +"_t is not identifiable")
#
#     D = nx.descendants(sg, y)
#     A = sg.nodes - D
#     adjustment_set = set()
#     for v in D:
#         for gamma in range(1, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     for v in A:
#         for gamma in range(0, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     adjustment_set.remove((x,gamma_xy))
#
#     return (None,adjustment_set)
#
#
# def smaller_adjustment_set(sg, x, y, gamma_xy, gamma_max = 1):
#     """
#     Gives a single-door set to identify the direct effect from x to y with lag gamma_xy if it is identifiable
#     for full-time causal graphs with maximum lag gamma_max compatible with (cyclic) summary causal graph sg.
#     Otherwise fails.
#     This corresponds to Proposition 1 in {identifiability of direct from summary causal graphs}.
#     The returned value of this function (x,S) can be interpreted as follows:
#     if x != None:
#         x is the direct effect.
#     else x == None:
#         The direct effect is can be estimated using the adjustment set S.
#         S = {(v,gamma)} where a pair (v,gamma) represents the vertex v_{t-gamma}.
#
#     :param sg: networkx directed graph
#     :param gamma_max: int
#     :param x: node
#     :param y: node
#     :param gamma_xy: int
#     :return: Set (nodes, int) : (v,gamma) corresponds to the vertex v_{t-gamma}
#     """
#     assert x != y
#     assert (x in sg.nodes) and (y in sg.nodes)
#     # There is no gamma_min_xy
#     # assert (0 <= gamma_min_xy) and (gamma_min_xy <= gamma_max)
#
#     if not sg.has_edge(x, y):
#         print("x not in Parents(sg,y) so direct effect is zero")
#         return (0,set())
#
#     if not is_identifiable_with_cycles(sg, gamma_max, x, y, 0):
#         raise ValueError("Direct effect of " + str(x) + "_t-" + gamma_xy + " on " + str(y) +"_t is not identifiable")
#
#     D = (nx.ancestors(sg, y)).intersection(nx.descendants(sg, y))
#     A = nx.ancestors(sg, y) - D
#     adjustment_set = set()
#     for v in D:
#         for gamma in range(1, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     for v in A:
#         for gamma in range(0, gamma_max + 1):
#             adjustment_set.add((v, gamma))
#     adjustment_set.remove((x,gamma_xy))
#
#     return (None,adjustment_set)
#
#
# if __name__ == "__main__":
#     gamma_xy = 1
#     gamma_max = 2
#
#     ascg_1 = nx.DiGraph()
#     ascg_1.add_edge("X", "Y")
#     ascg_1.add_edge("X", "U")
#     ascg_1.add_edge("U", "Z")
#     ascg_1.add_edge("Z", "W")
#     ascg_1.add_edge("W", "Y")
#
#     ascg_2 = nx.DiGraph()
#     ascg_2.add_edge("X", "Y")
#     ascg_2.add_edge("U", "X")
#     ascg_2.add_edge("Z", "U")
#     ascg_2.add_edge("Z", "W")
#     ascg_2.add_edge("W", "Y")
#
#     ascg_3 = nx.DiGraph()
#     ascg_3.add_edge("X", "Y")
#     ascg_3.add_edge("X", "U")
#     ascg_3.add_edge("U", "Z")
#     ascg_3.add_edge("W", "Z")
#     ascg_3.add_edge("Y", "W")
#
#     list_ascgs = [ascg_1, ascg_2, ascg_3]
#
#     def add_fixed_edges(scg):
#         scg.add_edges_from([("X", "X"), ("Y", "Y"), ("U", "U"), ("W", "W"), ("Z", "Z")])
#         scg.add_edge("X", "Y")
#         scg.add_edge("X", "U")
#         scg.add_edge("U", "X")
#         scg.add_edge("Y", "W")
#         scg.add_edge("W", "Y")
#
#     scg1_1 = nx.DiGraph()
#     add_fixed_edges(scg1_1)
#     scg1_1.add_edge("U", "Z")
#     scg1_1.add_edge("Z", "W")
#
#     scg1_2 = nx.DiGraph()
#     add_fixed_edges(scg1_2)
#     scg1_2.add_edge("U", "Z")
#     scg1_2.add_edge("Z", "U")
#     scg1_2.add_edge("Z", "W")
#
#     scg1_3 = nx.DiGraph()
#     add_fixed_edges(scg1_3)
#     scg1_3.add_edge("U", "Z")
#     scg1_3.add_edge("Z", "W")
#     scg1_3.add_edge("W", "Z")
#
#     scg1_4 = nx.DiGraph()
#     add_fixed_edges(scg1_4)
#     scg1_4.add_edge("U", "Z")
#     scg1_4.add_edge("W", "Z")
#
#     scg1_5 = nx.DiGraph()
#     add_fixed_edges(scg1_5)
#     scg1_5.add_edge("Z", "U")
#     scg1_5.add_edge("Z", "W")
#
#     list_scgs_identifiable = [scg1_1, scg1_2, scg1_3, scg1_4, scg1_5]
#
#     scg2_1 = nx.DiGraph()
#     add_fixed_edges(scg2_1)
#     scg2_1.add_edge("Z", "U")
#     scg2_1.add_edge("W", "Z")
#
#     scg2_2 = nx.DiGraph()
#     add_fixed_edges(scg2_2)
#     scg2_2.add_edge("Z", "U")
#     scg2_2.add_edge("W", "Z")
#     scg2_2.add_edge("Z", "W")
#
#     scg2_3 = nx.DiGraph()
#     add_fixed_edges(scg2_3)
#     scg2_3.add_edge("Z", "U")
#     scg2_3.add_edge("U", "Z")
#     scg2_3.add_edge("W", "Z")
#
#     list_scgs_identifiable_with_cond = [scg2_1, scg2_2, scg2_3]
#
#     scg3_1 = nx.DiGraph()
#     add_fixed_edges(scg3_1)
#     scg3_1.add_edge("Z", "U")
#     scg3_1.add_edge("U", "Z")
#     scg3_1.add_edge("W", "Z")
#     scg3_1.add_edge("Z", "W")
#
#     list_scgs_not_identifiable = [scg3_1]
#
#     print("ASCGs")
#     for ascg in list_ascgs:
#         res = is_identifiable_with_cycles(ascg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#
#     print("Identifiable SCGs")
#     for scg in list_scgs_identifiable:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#         # if res:
#         #     print(huge_adjustment_set(scg, "X", "Y", gamma_xy, gamma_max=gamma_max))
#
#     print("Identifiable SCGs if gamma_xy >0")
#     for scg in list_scgs_identifiable_with_cond:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#
#     print("Not identifiable SCGs")
#     for scg in list_scgs_not_identifiable:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
#
#     special_scg = nx.DiGraph()
#     add_fixed_edges(special_scg)
#     special_scg.add_edge("Y", "X")
#     list_special_scgs = [special_scg]
#
#     print("Special SCGs")
#     for scg in list_special_scgs:
#         res = is_identifiable_with_cycles(scg, "X", "Y", gamma_xy, gamma_max=gamma_max)
#         print(res)
