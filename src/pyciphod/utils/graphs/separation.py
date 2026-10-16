from typing import Iterable, List, Set
from pyciphod.utils.graphs.graphs import Graph
from pyciphod.utils.time_series.data_format import DTimeVar

# def d_separated(graph: Graph, X: Iterable[str], Y: Iterable[str], Z: Iterable[str] = None, max_path_length: int = 100) -> bool:
#     """Return True if sets X and Y are d-separated given Z in `graph`.

#     The function searches for any active path between any x in X and y in Y; if none exists, returns True.

#     Parameters:
#     - graph: Graph or subclass
#     - X, Y: iterables of node labels
#     - Z: iterable of conditioning node labels
#     - max_path_length: cutoff for path search (avoids combinatorial explosion)
#     """
#     Xs = set(X)
#     Ys = set(Y)
#     Zs = set(Z or [])

#     vertices = set(graph.get_vertices())
#     # sanity: restrict to existing nodes
#     Xs = Xs & vertices
#     Ys = Ys & vertices
#     Zs = Zs & vertices

#     if not Xs or not Ys:
#         # trivially separated if one is empty
#         return True

#     # For each pair, search for a simple path up to cutoff that is active
#     for x in Xs:
#         for y in Ys:
#             if x == y:
#                 # same node: not separated unless conditioning removes (we consider not separated)
#                 return False
#             try:
#                 for path in graph.get_simple_paths(x, y, allowed_nodes=None, cutoff=max_path_length):
#                     if graph.is_active_path(path, Zs):
#                         #print(path)
#                         # found an active path => not d-separated
#                         return False
#             except Exception:
#                 # on errors conservatively assume not separated
#                 return False
#     # no active path found
#     return True

def d_separated(graph: Graph, X: Iterable[str], Y: Iterable[str], Z: Iterable[str] = None, max_path_length: int = 100) -> bool:
    """Return True if sets X and Y are d-separated given Z in `graph`.

    The function searches for any active path between any x in X and y in Y; if none exists, returns True.

    Parameters:
    - graph: Graph or subclass
    - X, Y: iterables of node labels
    - Z: iterable of conditioning node labels
    - max_path_length: cutoff for path search (avoids combinatorial explosion)
    """
    Xs = set(X)
    Ys = set(Y)
    Zs = set(Z or [])

    vertices = set(graph.get_vertices())

    # --- DEBUG: check for silent drops during intersection with vertices ---
    #print(f"[d_separated] DEBUG Xs before filter: {Xs}")
    #print(f"[d_separated] DEBUG Ys before filter: {Ys}")
    #print(f"[d_separated] DEBUG Zs before filter: {Zs}")
    #print(f"[d_separated] DEBUG vertices: {vertices}")

    # sanity: restrict to existing nodes
    Xs = Xs & vertices
    Ys = Ys & vertices
    Zs = Zs & vertices

    #print(f"[d_separated] DEBUG Xs after filter:  {Xs}")
    #print(f"[d_separated] DEBUG Ys after filter:  {Ys}")
    #print(f"[d_separated] DEBUG Zs after filter:  {Zs}")
    if Z and len(Zs) != len(set(Z) & set(Z)):
        pass  # placeholder, real check below
    dropped_z = set(Z or []) - Zs
    # if dropped_z:
    #     #print(f"[d_separated] WARNING: nodes dropped from Z (not found in vertices?): {dropped_z}")
    # --- END DEBUG ---

    if not Xs or not Ys:
        return True

    # For each pair, search for a simple path up to cutoff that is active
    for x in Xs:
        for y in Ys:
            if x == y:
                #print(f"[d_separated] DEBUG x == y ({x}), returning False")
                return False
            try:
                path_count = 0
                for path in graph.get_simple_paths(x, y, allowed_nodes=None, cutoff=max_path_length):
                    path_count += 1
                    active = graph.is_active_path(path, Zs)
                    #print(f"[d_separated] DEBUG path={path} active={active}")
                    if active:
                        #print(f"[d_separated] DEBUG => ACTIVE PATH FOUND between {x} and {y}: {path}")
                        return False
                #print(f"[d_separated] DEBUG total paths checked between {x} and {y}: {path_count}")
            except Exception as e:
                #print(f"[d_separated] DEBUG EXCEPTION during path search {x}->{y}: {e!r}")
                return False
    return True




def graph_to_latent_graph(graph, latent_prefix: str = "U"):
    """
    Convert an ADMG into a DAG by replacing each bidirected edge X <-> Y
    with a fresh latent node U_xy and edges:

        U_xy -> X
        U_xy -> Y

    Directed edges are copied unchanged.

    Returns
    -------
    latent_graph
        A DAG-like graph containing:
        - all original observed nodes
        - one fresh latent node per bidirected edge
    latent_nodes
        The set of newly created latent node names.
    """
    latent_graph = graph.copy()  # start with a copy of the original graph (or create a new empty graph if needed)

    used_names = set(list(graph.get_vertices()))

    latent_nodes: Set[str] = set()

    for i, (x, y) in enumerate(sorted(graph.get_confounded_edges())):
        base_name = f"{latent_prefix}_{x}_{y}"
        u = _fresh_name(base_name, used_names)
        used_names.add(u)
        latent_nodes.add(u)

        latent_graph.add_vertex(u)
        latent_graph.add_directed_edge(u, x)
        latent_graph.add_directed_edge(u, y)
        latent_graph.remove_confounded_edge(x, y)

    return latent_graph, latent_nodes


def _fresh_name(base_name: str, used_names: Set[str]) -> str:
    if base_name not in used_names:
        return base_name

    k = 1
    while f"{base_name}_{k}" in used_names:
        k += 1
    return f"{base_name}_{k}"


def m_separated(graph: Graph, X: Iterable[str], Y: Iterable[str], Z: Iterable[str] = None, max_path_length: int = 100) -> bool:
    """Return True if sets X and Y are m-separated given Z in `graph`.

    The function searches for any active path between any x in X and y in Y; if none exists, returns True.

    Parameters:
    - graph: Graph or subclass
    - X, Y: iterables of node labels
    - Z: iterable of conditioning node labels
    - max_path_length: cutoff for path search (avoids combinatorial explosion)
    """
    # For m-separation, we can treat all edges as undirected for path search, but still apply the same active path rules.
    # We can use the same d_separated function but with a modified graph that treats all edges as undirected.
    latent_graph, _ = graph_to_latent_graph(graph)
    return d_separated(latent_graph, X, Y, Z, max_path_length)


def ts_d_separated(
    graph: Graph,
    X: Iterable[DTimeVar],
    Y: Iterable[DTimeVar],
    Z: Iterable[DTimeVar] = None,
    max_path_length: int = 100,
) -> bool:
    """Return True if sets X and Y are d-separated given Z in `graph`,
    where X, Y, Z are iterables of DTimeVar (time-indexed variables).

    The function converts each DTimeVar to its corresponding node label
    in `graph` (via time_var_to_str) and then searches for any active
    path between any x in X and y in Y; if none exists, returns True.

    Parameters:
    - graph: Graph or subclass, whose vertices are labeled using
      time_var_to_str(DTimeVar(...)) strings
    - X, Y: iterables of DTimeVar
    - Z: iterable of DTimeVar (conditioning set)
    - max_path_length: cutoff for path search (avoids combinatorial explosion)
    """
    def to_labels(time_vars: Iterable[DTimeVar]) -> set:
        labels = set()
        for tv in time_vars:
            if not isinstance(tv, DTimeVar):
                raise TypeError(f"Expected DTimeVar, got {type(tv).__name__}")
            labels.add(time_var_to_str(tv))
        return labels

    Xs = to_labels(X)
    Ys = to_labels(Y)
    Zs = to_labels(Z or [])

    vertices = set(graph.get_vertices())

    # sanity: restrict to existing nodes
    Xs = Xs & vertices
    Ys = Ys & vertices
    Zs = Zs & vertices

    if not Xs or not Ys:
        # trivially separated if one is empty
        return True

    # For each pair, search for a simple path up to cutoff that is active
    for x in Xs:
        for y in Ys:
            if x == y:
                # same node: not separated unless conditioning removes (we consider not separated)
                return False
            try:
                for path in graph.get_simple_paths(x, y, allowed_nodes=None, cutoff=max_path_length):
                    if graph.is_active_path(path, Zs):
                        # found an active path => not d-separated
                        return False
            except Exception:
                # on errors conservatively assume not separated
                return False

    # no active path found
    return True