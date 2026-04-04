from typing import Set, Tuple, Iterable, FrozenSet
from itertools import combinations, product

from pyciphod.utils.graphs.graphs import Graph


def _edge_points_to(graph: Graph, src: str, dst: str) -> bool:
    """
    Détermine si une arête entre src et dst peut pointer vers dst sous forme certaine ou incertaine.
    On considère que les types suivants indiquent une flèche pointant vers dst when present in the multigraph:
    - '->' or any uncertain type that ends with '>' (e.g. '*->', '-->')
    - '<->' contributes des flèches dans les deux sens (traitée comme pointant vers dst)
    - '-' (undirected) ne pointe pas vers dst
    - '*-o' or '*-' or '-||' : points depend on convention; we'll treat '*->' and '-->' and '->' and '<->' as pointing to dst.

    Cette fonction est permissive pour gérer les graphes partiellement spécifiés.
    """
    # types that clearly point to dst when the edge is (src, dst, type)
    pointing_types = {"->", "o->", "-->"}
    # '<->' is bidirectional: treat as pointing to dst as well
    pointing_types.add("<->")

    try:
        edge_types = graph.get_edge_types(src, dst)
    except Exception:
        # if no edge data available, assume no pointing
        edge_types = set()

    # If any of the edge types corresponds to an arrow into dst, return True
    for et in edge_types:
        if et in pointing_types:
            return True
    return False


def get_colliders(graph: Graph) -> Set[Tuple[str, str, str]]:
    """
    Retourne tous les colliders (v-structures) du graphe donné (shielded et unshielded).

    Un collider est un triplet (x, z, y) tel que :
    - x et z sont adjacents
    - y et z sont adjacents
    - les arêtes entre x-z et y-z ont des marques pointant vers z (i.e. x -> z <- y), en tenant compte
      des arêtes incertaines et des arêtes bidirectionnelles '<->' comme pointant vers z.

    Contrairement à `get_unshielded_colliders`, ici on n'exige pas que x et y ne soient pas adjacents.

    Retour : set de tuples (x, z, y). L'ordre (x,y) est normalisé (tri) pour éviter duplications symétriques.
    """
    colliders = set()
    vertices = list(graph.get_vertices())

    for z in vertices:
        adj = graph.get_adjacencies(z)
        if len(adj) < 2:
            continue
        adj_list = list(adj)
        n = len(adj_list)
        for i in range(n):
            x = adj_list[i]
            for j in range(i + 1, n):
                y = adj_list[j]
                # Now check if edges point towards z from x and y
                x_points = _edge_points_to(graph, x, z)
                y_points = _edge_points_to(graph, y, z)
                if x_points and y_points:
                    # normalize order of x,y to avoid duplicates
                    a, b = (x, y) if x <= y else (y, x)
                    colliders.add((a, z, b))
    return colliders


def get_unshielded_colliders(graph: Graph) -> Set[Tuple[str, str, str]]:
    """
    Retourne les colliders non protégés (unshielded colliders) du graphe.

    Filtre les colliders retournés par `get_colliders` pour ne garder que ceux où les deux parents
    ne sont pas adjacents.
    """
    all_colliders = get_colliders(graph)
    unshielded = set()
    for (x, z, y) in all_colliders:
        # x and y are not adjacent
        if y not in graph.get_adjacencies(x):
            unshielded.add((x, z, y))
    return unshielded


# Helpers that operate directly on Graph (no new global sg creation)

def _has_directed_edge(graph: Graph, u: str, v: str) -> bool:
    """True if there is a definite or uncertain arrow u->v in the graph's representation."""
    # check directed edges
    if (u, v) in graph.get_directed_edges():
        return True
    # check multigraph edge types
    try:
        types = graph.get_edge_types(u, v)
    except Exception:
        types = set()
    for t in types:
        if isinstance(t, str) and (t in {"->", "*->", "-->"} or t.endswith('>')):
            return True
        if t == '<->':
            return True
    return False


def _has_edge_either_dir(graph: Graph, u: str, v: str) -> bool:
    """True if there is any adjacency between u and v (directed, confounded, undirected, uncertain)."""
    return v in graph.get_adjacencies(u)


def is_active_on_graph(graph: Graph, path: list, adjustment_set: set = None) -> bool:
    """
    Reproduit la logique de `is_active` utilisée dans les routines SCG, mais en travaillant directement
    sur l'objet `Graph` du projet (sans construire un `sg` séparé).
    La logique est adaptée de `causal_reasoning.cluster_graph.scg.micro_queries.direct_effect.is_active`.
    """
    if adjustment_set is None:
        adjustment_set = set()

    colliders = set()
    has_seen_right_arrow = False
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        # check connectivity and direction
        a_to_b = _has_directed_edge(graph, a, b)
        b_to_a = _has_directed_edge(graph, b, a)
        if a_to_b and b_to_a:
            # bidirectional (treat like <->)
            pass
        elif a_to_b and not b_to_a:
            # a -> b
            if (i > 0) and (path[i] in adjustment_set):
                return False
            has_seen_right_arrow = True
        elif (not a_to_b) and b_to_a:
            # a <- b
            if has_seen_right_arrow:
                colliders.add(path[i])
                has_seen_right_arrow = False
        else:
            # no directed arrow either way; check if there is an adjacency (undirected/uncertain without arrows)
            if _has_edge_either_dir(graph, a, b):
                # treat as non-collider segment: if the middle node is in adjustment_set, path is blocked
                if (i > 0) and (path[i] in adjustment_set):
                    return False
                # non-collider resets seen-right-arrow
                has_seen_right_arrow = False
                continue
            # not connected at all
            raise ValueError("Path is not a path in graph: {} and {} are not connected.".format(a, b))
    # For each collider, at least one descendant is in adjustment_set (or collider itself)
    for c in colliders:
        if c in adjustment_set:
            continue
        for d in graph.get_descendants(c):
            if d in adjustment_set:
                break
        else:
            return False
    return True


def _all_simple_paths_graph(graph: Graph, source: str, target: str, allowed_nodes: Iterable[str] = None, cutoff: int = 10):
    """
    Générateur de chemins simples entre source et target en se basant uniquement sur `get_adjacencies`.
    Evite la création d'un nouveau graphe en itérant récursivement sur les voisins.
    """
    allowed = None if allowed_nodes is None else set(allowed_nodes)

    if allowed is not None and (source not in allowed or target not in allowed):
        return

    visited = [source]

    def dfs(current):
        if len(visited) > cutoff:
            return
        if current == target:
            yield list(visited)
            return
        # neighbors via adjacencies
        neighs = graph.get_adjacencies(current)
        if allowed is not None:
            neighs = neighs & allowed
        for n in neighs:
            if n in visited:
                continue
            visited.append(n)
            for p in dfs(n):
                yield p
            visited.pop()

    for p in dfs(source):
        yield p


def _active_path_exists(graph: Graph, u: str, v: str, allowed_nodes: Iterable[str] = None, max_path_length: int = 10) -> bool:
    """
    Recherche s'il existe un chemin actif entre u et v dans `graph` sans construire un `sg` séparé.
    """
    try:
        for path in _all_simple_paths_graph(graph, u, v, allowed_nodes=allowed_nodes, cutoff=max_path_length):
            if is_active_on_graph(graph, path, adjustment_set=set()):
                return True
    except ValueError:
        return False
    return False


def _all_nonempty_subsets(iterable: Iterable[str]):
    s = list(iterable)
    for r in range(1, len(s) + 1):
        for comb in combinations(s, r):
            yield frozenset(comb)


def get_cluster_super_unshielded_colliders(graph: Graph, max_vertices_for_search: int = 12, max_path_length: int = 8) -> Set[Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str]]]:
    """
    Retourne l'ensemble des clustered super-unshielded colliders (CSUC) du graphe `graph`.

    Implémentation exhaustive utilisant uniquement les méthodes/attributs du `Graph` fourni.
    """
    vertices = list(graph.get_vertices())
    n = len(vertices)
    if n == 0:
        return set()
    if n > max_vertices_for_search:
        raise ValueError(f"Graph too large for exhaustive CSUC search (|V|={n}). Increase max_vertices_for_search with caution.")

    # Precompute active_any between all pairs (unrestricted)
    active_any = {}
    for u, v in product(vertices, repeat=2):
        if u == v:
            active_any[(u, v)] = False
        else:
            active_any[(u, v)] = _active_path_exists(graph, u, v, allowed_nodes=None, max_path_length=max_path_length)

    candidates = set()
    Vset = set(vertices)

    # enumerate Y (non-empty)
    for Y in _all_nonempty_subsets(vertices):
        outside = Vset - Y
        if not outside:
            continue
        # enumerate X as non-empty subset of outside
        for X in _all_nonempty_subsets(outside):
            rest = outside - X
            if not rest:
                continue
            # enumerate Z as non-empty subset of rest
            for Z in _all_nonempty_subsets(rest):
                # ensure disjointness by construction
                if X & Y or Y & Z or X & Z:
                    continue
                # Check separation across sides: no active path between any x in X and z in Z
                separated = True
                for x in X:
                    for z in Z:
                        if active_any.get((x, z), False) or active_any.get((z, x), False):
                            separated = False
                            break
                    if not separated:
                        break
                if not separated:
                    continue
                # Connection to the middle: for each x in X and y in Y exists active path within X∪Y
                XY_allowed = X | Y
                XY_ok = True
                for x in X:
                    for y in Y:
                        if not _active_path_exists(graph, x, y, allowed_nodes=XY_allowed, max_path_length=max_path_length):
                            XY_ok = False
                            break
                    if not XY_ok:
                        break
                if not XY_ok:
                    continue
                # Similarly for ZY
                ZY_allowed = Z | Y
                ZY_ok = True
                for z in Z:
                    for y in Y:
                        if not _active_path_exists(graph, z, y, allowed_nodes=ZY_allowed, max_path_length=max_path_length):
                            ZY_ok = False
                            break
                    if not ZY_ok:
                        break
                if not ZY_ok:
                    continue
                # At this point (X,Y,Z) satisfy (1)-(2)
                candidates.add((frozenset(X), frozenset(Y), frozenset(Z)))

    # Filter maximal triplets (no strict superset in candidates)
    maximal = set()
    for trip in candidates:
        X, Y, Z = trip
        is_max = True
        for other in candidates:
            if other == trip:
                continue
            X2, Y2, Z2 = other
            if X <= X2 and Y <= Y2 and Z <= Z2 and (X < X2 or Y < Y2 or Z < Z2):
                is_max = False
                break
        if is_max:
            maximal.add(trip)

    return maximal

