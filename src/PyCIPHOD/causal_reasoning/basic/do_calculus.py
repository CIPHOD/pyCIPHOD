from typing import Iterable, Set
import networkx as nx

from pyciphod.utils.graphs.graphs import DirectedMixedGraph, DirectedAcyclicGraph


def _to_set(nodes: Iterable) -> Set[str]:
    if nodes is None:
        return set()
    if isinstance(nodes, (str,)):
        return {nodes}
    return set(nodes)


def _copy_dmg(g: DirectedMixedGraph) -> DirectedMixedGraph:
    """Make a shallow copy of the DirectedMixedGraph structure (vertices + directed + confounded edges).
    Uncertain or undirected edges are ignored because do-calculus conditions use directed and bidirected confounding.
    """
    new = DirectedMixedGraph()
    for v in g.get_vertices():
        new.add_vertex(v)
    for (u, v) in g.get_directed_edges():
        new.add_directed_edge(u, v)
    for (u, v) in g.get_confounded_edges():
        # add_confounded_edge expects unordered pair, ensure we add once
        new.add_confounded_edge(u, v)
    return new


def _remove_incoming(dmg: DirectedMixedGraph, nodes: Iterable[str]) -> None:
    for n in list(_to_set(nodes)):
        for p in list(dmg.get_parents(n)):
            dmg.remove_directed_edge(p, n)


def _remove_outgoing(dmg: DirectedMixedGraph, nodes: Iterable[str]) -> None:
    for n in list(_to_set(nodes)):
        for c in list(dmg.get_children(n)):
            dmg.remove_directed_edge(n, c)


def _build_augmented_dag(dmg: DirectedMixedGraph) -> DirectedAcyclicGraph:
    """Build a DAG suitable for d-separation checks by converting each confounded (bidirected) edge u<->v
    into a latent node L_uv with edges L_uv -> u and L_uv -> v. Directed edges are kept as-is.
    Returns a DirectedAcyclicGraph (package class).
    """
    aug = DirectedAcyclicGraph()
    # add observed nodes
    for v in dmg.get_vertices():
        aug.add_vertex(v)
    # add directed edges
    for (u, v) in dmg.get_directed_edges():
        aug.add_directed_edge(u, v)
    # add latent nodes for confounded edges
    for (u, v) in dmg.get_confounded_edges():
        # canonical order
        a, b = sorted((u, v))
        latent = f"__L__{a}__{b}"
        # avoid collision with existing nodes
        if latent not in aug.get_vertices():
            aug.add_vertex(latent)
        aug.add_directed_edge(latent, u)
        aug.add_directed_edge(latent, v)
    return aug


def _is_d_separated_by_moralization(dmg: DirectedMixedGraph, X: Iterable, Y: Iterable, Z: Iterable) -> bool:
    """Check d-separation (m-separation handling latent confounders) by moralizing the ancestral graph.
    Steps:
      1. Build augmented DAG (latent nodes for bidirected confounding) as a DirectedAcyclicGraph.
      2. Compute ancestors of X U Y U Z in the augmented DAG.
      3. Induce the subgraph on the ancestors and moralize it (connect parents, drop directions).
      4. Remove conditioned nodes Z and test whether any x in X is connected to any y in Y.
    Returns True if X is d-separated from Y given Z (i.e. no active path).
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)

    aug = _build_augmented_dag(dmg)

    # ancestors of X U Y U Z
    nodes_of_interest = set()
    for n in Xs | Ys | Zs:
        if n in aug.get_vertices():
            nodes_of_interest.add(n)
            nodes_of_interest.update(aug.get_ancestors(n))

    if not nodes_of_interest:
        return True

    # build induced subgraph H as DirectedAcyclicGraph
    H = DirectedAcyclicGraph()
    for v in nodes_of_interest:
        H.add_vertex(v)
    for (u, v) in aug.get_directed_edges():
        if u in nodes_of_interest and v in nodes_of_interest:
            H.add_directed_edge(u, v)

    # moralize H into an undirected networkx graph
    M = nx.Graph()
    M.add_nodes_from(H.get_vertices())
    # add undirected edges for all directed edges
    for (u, v) in H.get_directed_edges():
        M.add_edge(u, v)
    # connect all parents of each node
    for node in H.get_vertices():
        parents = list(H.get_parents(node))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                M.add_edge(parents[i], parents[j])

    # remove conditioning nodes Z
    Z_in_M = [z for z in Zs if z in M]
    if Z_in_M:
        M.remove_nodes_from(Z_in_M)

    # check connectivity
    for x in Xs:
        for y in Ys:
            if x not in M or y not in M:
                continue
            if nx.has_path(M, x, y):
                return False
    return True


# Public API: three rules

def rule1_applies(dmg: DirectedMixedGraph, X: Iterable, Y: Iterable, Z: Iterable, W: Iterable = None) -> bool:
    """Rule 1 (insertion/deletion of observations)
    If Y is independent of Z given X and W in G_{bar{X}} (graph where incoming edges to X are removed),
    then P(Y | do(X), Z, W) = P(Y | do(X), W).

    Returns True if the condition holds on the graph (i.e., rule can be applied).
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)
    Ws = _to_set(W)

    g_copy = _copy_dmg(dmg)
    # remove incoming edges to X
    _remove_incoming(g_copy, Xs)

    # test d-separation: Y _||_ Z | X U W in G_barX
    cond = Xs | Ws
    return _is_d_separated_by_moralization(g_copy, Ys, Zs, cond)


def rule2_applies(dmg: DirectedMixedGraph, X: Iterable, Y: Iterable, Z: Iterable, W: Iterable = None) -> bool:
    """Rule 2 (action/observation exchange)
    If Y is independent of Z given X and W in G_{bar{X},underline{Z}} (incoming edges to X removed, outgoing from Z removed),
    then P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W).

    Returns True if the condition holds on the graph.
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)
    Ws = _to_set(W)

    g_copy = _copy_dmg(dmg)
    _remove_incoming(g_copy, Xs)
    _remove_outgoing(g_copy, Zs)

    cond = Xs | Ws
    return _is_d_separated_by_moralization(g_copy, Ys, Zs, cond)


def rule3_applies(dmg: DirectedMixedGraph, X: Iterable, Y: Iterable, Z: Iterable, W: Iterable = None) -> bool:
    """Rule 3 (insertion/deletion of actions)
    Let Z* be the subset of Z that are not ancestors of any W in G_{bar{X}}. If Y is independent of Z given X and W
    in G_{bar{X},overline{Z*}} (incoming edges to X removed and incoming edges to Z* removed), then
    P(Y | do(X), do(Z), W) = P(Y | do(X), W).

    Returns True if the condition holds on the graph.
    """
    Xs = _to_set(X)
    Ys = _to_set(Y)
    Zs = _to_set(Z)
    Ws = _to_set(W)

    # G_barX
    g_barx = _copy_dmg(dmg)
    _remove_incoming(g_barx, Xs)

    # compute ancestors of W in g_barx (in directed sense using augmented DAG)
    aug_barx = _build_augmented_dag(g_barx)
    ancestors_of_W = set()
    for w in Ws:
        if w in aug_barx.get_vertices():
            ancestors_of_W.add(w)
            ancestors_of_W.update(aug_barx.get_ancestors(w))

    # Z* are elements of Z that are NOT ancestors of any W
    Z_star = {z for z in Zs if z not in ancestors_of_W}

    # G_barX_barZstar: remove incoming edges to nodes in Z*
    g_copy = _copy_dmg(g_barx)
    _remove_incoming(g_copy, Z_star)

    cond = Xs | Ws
    return _is_d_separated_by_moralization(g_copy, Ys, Zs, cond)


__all__ = [
    "rule1_applies",
    "rule2_applies",
    "rule3_applies",
]
