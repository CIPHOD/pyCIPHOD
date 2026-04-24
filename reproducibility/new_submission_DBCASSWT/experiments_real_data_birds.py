import pandas as pd

from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest

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

    csuc = restpc.g_hat.get_all_cluster_super_unshielded_colliders(max_vertices_for_search=100, max_path_length=10)
    print(csuc)
