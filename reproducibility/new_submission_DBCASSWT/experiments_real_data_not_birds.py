import pandas as pd

from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest


def partial_causal_order(g):
    """
    Return a partial causal order for a partially directed graph.

    Nodes are placed in the same spot if they are mutually reachable when:
    - directed edges u -> v are followed in their direction
    - undirected edges u - v are treated as bidirectional

    So this correctly merges:
    - undirected-path components
    - directed cycles
    - mixed structures like A -> B - C -> A

    Parameters
    ----------
    nodes : iterable
        Node labels.
    has_directed_edge : callable
        has_directed_edge(u, v) -> bool
    has_undirected_edge : callable
        has_undirected_edge(u, v) -> bool

    Returns
    -------
    list[list]
        One valid topological order of blocks.
        Example: [['A', 'B'], ['C'], ['D', 'E']]
    """
    nodes = list(g.get_vertices())

    def has_mixed_edge(u, v, g):
        # directed edge u -> v
        if has_directed_edge(u, v, g):
            return True
        # undirected edge u - v acts both ways
        if has_undirected_edge(u, v, g):
            return True
        return False

    # ------------------------------------------------------------
    # 1) Strongly connected components in the mixed graph
    #    (directed edges + undirected edges as bidirectional)
    # ------------------------------------------------------------
    visited = set()
    finish_order = []

    def dfs1(u):
        visited.add(u)
        for v in nodes:
            if v != u and has_mixed_edge(u, v, g) and v not in visited:
                dfs1(v)
        finish_order.append(u)

    for u in nodes:
        if u not in visited:
            dfs1(u)

    visited.clear()
    sccs = []

    def dfs2(u, comp):
        visited.add(u)
        comp.append(u)
        for v in nodes:
            if v == u or v in visited:
                continue
            # reverse graph:
            # reverse of u->v is v->u
            # undirected edges stay undirected, so they also work here
            if has_mixed_edge(v, u, g):
                dfs2(v, comp)

    for u in reversed(finish_order):
        if u not in visited:
            comp = []
            dfs2(u, comp)
            sccs.append(sorted(comp))

    # map node -> block id
    block_id = {}
    for i, comp in enumerate(sccs):
        for u in comp:
            block_id[u] = i

    # ------------------------------------------------------------
    # 2) Build DAG between SCC blocks
    # ------------------------------------------------------------
    parents = [set() for _ in range(len(sccs))]

    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            if has_mixed_edge(u, v, g):
                bu = block_id[u]
                bv = block_id[v]
                if bu != bv:
                    parents[bv].add(bu)

    # ------------------------------------------------------------
    # 3) Topological order of the block DAG
    # ------------------------------------------------------------
    remaining = set(range(len(sccs)))
    order = []

    while remaining:
        current = [
            b for b in remaining
            if not any(p in remaining for p in parents[b])
        ]

        if not current:
            raise ValueError("Unexpected cycle after SCC compression.")

        for b in current:
            order.append(sccs[b])

        for b in current:
            remaining.remove(b)

    return order


def has_directed_edge(u, v, g):
    return (u, v) in g.get_directed_edges() and (v, u) not in g.get_directed_edges()

def has_undirected_edge(u, v, g):
    return u != v and (u, v) in g.get_undirected_edges() and (v, u) in g.get_undirected_edges()

if __name__ == '__main__':
    sig_level = 0.01
    m = pd.read_csv("./p_val_matrix.csv")
    m.index = m.columns

    # m = m.iloc[10:20, 10:20]

    g = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    g.add_vertices(list(m.columns))
    count_sep = 0
    for i in range(len(m.columns)):
        col_i = m.columns[i]
        for j in range(i+1, len(m.columns)):
            col_j = m.columns[j]
            if m[col_i].loc[col_j] < sig_level:
                g.add_undirected_edge(col_i, col_j)
            else:
                count_sep = count_sep + 1

    print(count_sep)

    restpc = RestPC(sparsity=sig_level, ci_test=CopulaTest)
    restpc.g_hat = g
    restpc._uc_rule()
    suc = restpc.g_hat.get_all_unshielded_colliders()
    print(len(suc))
    # unique_suc = []
    # for s in suc :
    #     if (s[2], s[1], s[1]) not in unique_suc:
    #         unique_suc.append(s)
    # print(len(unique_suc))

    # restpc.g_hat.draw_graph()

    max_parents = 0
    vertex_with_max_parents = None
    for v in restpc.g_hat.get_vertices():
        nb_parents = len(restpc.g_hat.get_parents(v))
        if nb_parents > max_parents:
            max_parents = nb_parents
            vertex_with_max_parents = v

    list_max_parents = [vertex_with_max_parents]
    for v in restpc.g_hat.get_vertices():
        nb_parents = len(restpc.g_hat.get_parents(v))
        if nb_parents == max_parents:
            list_max_parents.append(v)

    print(vertex_with_max_parents, max_parents)
    print(list_max_parents)

    co = partial_causal_order(restpc.g_hat)
    print(co)
    # csuc = restpc.g_hat.get_all_cluster_super_unshielded_colliders(max_vertices_for_search=100, max_path_length=10)
    # print(csuc)
