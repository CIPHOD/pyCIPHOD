from typing import Iterable, List, Set
from itertools import combinations

from .graphs import Graph


def d_separated(graph: Graph, X: Iterable[str], Y: Iterable[str], Z: Iterable[str] = None, max_path_length: int = 10) -> bool:
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
