from typing import Set, Tuple, FrozenSet, Iterable
import numpy as np
import pandas as pd
import random

from pyciphod.utils.scms.dynamic_scm import create_random_linear_dt_dynamic_scm, DtDynamicSCM, create_random_linear_dt_dynamic_scm_from_ftadmg
from pyciphod.utils.time_series.data_format import DTimeVar
from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicDifferenceGraph
from pyciphod.causal_discovery.basic.constraint_based import PC, RestPC, FCI
from pyciphod.utils.graphs.temporal_graphs import create_random_ft_dag, FtDirectedAcyclicGraph
from pyciphod.utils.stat_tests.independence_tests import LinearRegressionCoefficientTTest, FisherZTest, CopulaTest, CIMhTest, KernelPartialCorrelationTest
from pyciphod.utils.stat_tests.dependency_measures import Copula, compute_copula_fit


def f1_score_o(g, g_hat):
    def arrowhead_set(graph):
        out = set()
        vertices = list(graph.get_vertices())
        for u in vertices:
            for v in vertices:
                if u == v:
                    continue
                if graph.is_adjacent(u, v) and graph.is_pointed_edge(u, v):
                    out.add((u, v))
        return out

    true_set = arrowhead_set(g)
    pred_set = arrowhead_set(g_hat)
    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def f1_score_a(g, g_hat):
    def adjacency_set(graph):
        out = set()
        vertices = list(graph.get_vertices())
        for u in vertices:
            for v in graph.get_adjacencies(u):
                if u == v:
                    continue
                out.add(frozenset((u, v)))
        return out

    true_set = adjacency_set(g)
    pred_set = adjacency_set(g_hat)

    TP = len(pred_set & true_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if prec + rec == 0.0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)



def chain_v_structure_1(n, name_prefix=""):
    X1 = DTimeVar(f"{name_prefix}X", 1)
    X2 = DTimeVar(f"{name_prefix}X", 2)
    X3 = DTimeVar(f"{name_prefix}X", 3)
    Y1 = DTimeVar(f"{name_prefix}Y", 1)
    Y2 = DTimeVar(f"{name_prefix}Y", 2)
    Y3 = DTimeVar(f"{name_prefix}Y", 3)
    Z1 = DTimeVar(f"{name_prefix}Z", 1)
    Z2 = DTimeVar(f"{name_prefix}Z", 2)
    Z3 = DTimeVar(f"{name_prefix}Z", 3)

    U1 = DTimeVar(f"{name_prefix}U", 1)
    U2 = DTimeVar(f"{name_prefix}U", 2)
    U3 = DTimeVar(f"{name_prefix}U", 3)
    W1 = DTimeVar(f"{name_prefix}W", 1)
    W2 = DTimeVar(f"{name_prefix}W", 2)
    W3 = DTimeVar(f"{name_prefix}W", 3)

    D1 = DTimeVar(f"{name_prefix}D1", 3)
    D2 = DTimeVar(f"{name_prefix}D2", 3)
    D3 = DTimeVar(f"{name_prefix}D3", 3)
    D4 = DTimeVar(f"{name_prefix}D4", 3)
    D5 = DTimeVar(f"{name_prefix}D5", 3)


    azz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = random.choice([1, -1]) * random.uniform(-1, -0.5)
    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    add = random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd = random.choice([1, -1]) * random.uniform(-1, -0.5)


    auu = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aww = random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u1 = np.random.normal(size=n)
    u2 = auu * u1 + np.random.normal(size=n)
    u3 = auu * u2 + np.random.normal(size=n)
    w1 = np.random.normal(size=n)
    w2 = aww * w1 + np.random.normal(size=n)
    w3 = aww * w2 + np.random.normal(size=n)

    x1 = np.random.normal(size=n)
    x2 = axx * x1 + aux * u1 + np.random.normal(size=n)
    x3 = axx * x2 + aux * u2 + np.random.normal(size=n)

    y1 = np.random.normal(size=n)
    y2 = ayy * y1 + awy * w1 + np.random.normal(size=n)
    y3 = ayy * y2 + awy * w2 + np.random.normal(size=n)

    z1 = np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x2 + ayz * y2 + np.random.normal(size=n)

    d_1_1 = np.random.normal(size=n)
    d_1_2 = add * d_1_1 + azd * z1 + np.random.normal(size=n)
    d_1_3 = add * d_1_2 + azd * z2 + np.random.normal(size=n)

    d_2_1 = np.random.normal(size=n)
    d_2_2 = add * d_2_1 + azd * z1 + np.random.normal(size=n)
    d_2_3 = add * d_2_2 + azd * z2 + np.random.normal(size=n)

    d_3_1 = np.random.normal(size=n)
    d_3_2 = add * d_3_1 + azd * z1 + np.random.normal(size=n)
    d_3_3 = add * d_3_2 + azd * z2 + np.random.normal(size=n)

    d_4_1 = np.random.normal(size=n)
    d_4_2 = add * d_4_1 + azd * z1 + np.random.normal(size=n)
    d_4_3 = add * d_4_2 + azd * z2 + np.random.normal(size=n)

    d_5_1 = np.random.normal(size=n)
    d_5_2 = add * d_5_1 + azd * z1 + np.random.normal(size=n)
    d_5_3 = add * d_5_2 + azd * z2 + np.random.normal(size=n)


    df = pd.DataFrame(
        {X3: x3,
         Y3: y3,
         Z3: z3,
         U3: u3,
         W3: w3,
         D1: d_1_3,
         D2: d_2_3,
         D3: d_3_3,
         D4: d_4_3,
         D5: d_5_3}
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X3, Y3, Z3, U3, W3, D1, D2, D3, D4, D5])
    ge.add_directed_edge(U3, X3)
    ge.add_directed_edge(U3, Z3)
    ge.add_directed_edge(W3, Y3)
    ge.add_directed_edge(W3, Z3)
    ge.add_directed_edge(X3, Z3)
    ge.add_directed_edge(Y3, Z3)
    ge.add_undirected_edge(U3, X3)
    ge.add_undirected_edge(W3, Y3)


    ge.add_directed_edge(U3, D1)
    ge.add_directed_edge(W3, D1)
    ge.add_directed_edge(X3, D1)
    ge.add_directed_edge(Y3, D1)

    ge.add_directed_edge(U3, D2)
    ge.add_directed_edge(W3, D2)
    ge.add_directed_edge(X3, D2)
    ge.add_directed_edge(Y3, D2)

    ge.add_directed_edge(U3, D3)
    ge.add_directed_edge(W3, D3)
    ge.add_directed_edge(X3, D3)
    ge.add_directed_edge(Y3, D3)

    ge.add_directed_edge(U3, D4)
    ge.add_directed_edge(W3, D4)
    ge.add_directed_edge(X3, D4)
    ge.add_directed_edge(Y3, D4)

    ge.add_directed_edge(U3, D5)
    ge.add_directed_edge(W3, D5)
    ge.add_directed_edge(X3, D5)
    ge.add_directed_edge(Y3, D5)

    ge.add_undirected_edge(Z3, D1)
    ge.add_undirected_edge(Z3, D2)
    ge.add_undirected_edge(Z3, D3)
    ge.add_undirected_edge(Z3, D4)
    ge.add_undirected_edge(Z3, D5)
    ge.add_undirected_edge(D1, D2)
    ge.add_undirected_edge(D1, D3)
    ge.add_undirected_edge(D1, D4)
    ge.add_undirected_edge(D1, D5)
    ge.add_undirected_edge(D2, D3)
    ge.add_undirected_edge(D2, D4)
    ge.add_undirected_edge(D2, D5)
    ge.add_undirected_edge(D3, D4)
    ge.add_undirected_edge(D3, D5)
    ge.add_undirected_edge(D4, D5)


    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3, U3, W3, D1, D2, D3, D4, D5])
    gt.add_directed_edge(X3, Z3)
    gt.add_directed_edge(Y3, Z3)
    gt.add_directed_edge(U3, X3)
    gt.add_directed_edge(W3, Y3)


    gt.add_directed_edge(Z3, D1)
    gt.add_directed_edge(Z3, D2)
    gt.add_directed_edge(Z3, D3)
    gt.add_directed_edge(Z3, D4)
    gt.add_directed_edge(Z3, D5)

    return df, ge, gt


def chain_v_structure_2(n, name_prefix=""):
    X1 = DTimeVar(f"{name_prefix}X", 1)
    X2 = DTimeVar(f"{name_prefix}X", 2)
    X3 = DTimeVar(f"{name_prefix}X", 3)
    Y1 = DTimeVar(f"{name_prefix}Y", 1)
    Y2 = DTimeVar(f"{name_prefix}Y", 2)
    Y3 = DTimeVar(f"{name_prefix}Y", 3)
    Z1 = DTimeVar(f"{name_prefix}Z", 1)
    Z2 = DTimeVar(f"{name_prefix}Z", 2)
    Z3 = DTimeVar(f"{name_prefix}Z", 3)

    U1 = DTimeVar(f"{name_prefix}U", 1)
    U2 = DTimeVar(f"{name_prefix}U", 2)
    U3 = DTimeVar(f"{name_prefix}U", 3)
    W1 = DTimeVar(f"{name_prefix}W", 1)
    W2 = DTimeVar(f"{name_prefix}W", 2)
    W3 = DTimeVar(f"{name_prefix}W", 3)

    D1 = DTimeVar(f"{name_prefix}D1", 3)
    D2 = DTimeVar(f"{name_prefix}D2", 3)
    D3 = DTimeVar(f"{name_prefix}D3", 3)

    V1 = DTimeVar(f"{name_prefix}V1", 3)
    V2 = DTimeVar(f"{name_prefix}V2", 3)


    azz = 2*random.choice([1, -1]) * random.uniform(-1, -0.5)
    axx = 2*random.choice([1, -1]) * random.uniform(-1, -0.5)
    axz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayy = 2*random.choice([1, -1]) * random.uniform(-1, -0.5)
    ayz = random.choice([1, -1]) * random.uniform(-1, -0.5)
    add = 2*random.choice([1, -1]) * random.uniform(-1, -0.5)
    azd = random.choice([1, -1]) * random.uniform(-1, -0.5)


    auu = 2* random.choice([1, -1]) * random.uniform(-1, -0.5)
    aux = random.choice([1, -1]) * random.uniform(-1, -0.5)
    aww = 2* random.choice([1, -1]) * random.uniform(-1, -0.5)
    awy = random.choice([1, -1]) * random.uniform(-1, -0.5)

    u1 = np.random.normal(size=n)
    u2 = auu * u1 + np.random.normal(size=n)
    u3 = auu * u2 + np.random.normal(size=n)
    w1 = np.random.normal(size=n)
    w2 = aww * w1 + np.random.normal(size=n)
    w3 = aww * w2 + np.random.normal(size=n)

    x1 = np.random.normal(size=n)
    x2 = axx * x1 + aux * u1 + np.random.normal(size=n)
    x3 = axx * x2 + aux * u2 + np.random.normal(size=n)

    y1 = np.random.normal(size=n)
    y2 = ayy * y1 + awy * w1 + np.random.normal(size=n)
    y3 = ayy * y2 + awy * w2 + np.random.normal(size=n)

    z1 = np.random.normal(size=n)
    z2 = azz * z1 + axz * x1 + ayz * y1 + np.random.normal(size=n)
    z3 = azz * z2 + axz * x2 + ayz * y2 + np.random.normal(size=n)

    d_1_1 = np.random.normal(size=n)
    d_1_2 = add * d_1_1 + azd * z1 + np.random.normal(size=n)
    d_1_3 = add * d_1_2 + azd * z2 + np.random.normal(size=n)

    d_2_1 = np.random.normal(size=n)
    d_2_2 = add * d_2_1 + azd * z1 + np.random.normal(size=n)
    d_2_3 = add * d_2_2 + azd * z2 + np.random.normal(size=n)

    d_3_1 = np.random.normal(size=n)
    d_3_2 = add * d_3_1 + azd * z1 + np.random.normal(size=n)
    d_3_3 = add * d_3_2 + azd * z2 + np.random.normal(size=n)

    v_1_1 = np.random.normal(size=n)
    v_1_2 = add * v_1_1 + azd * x1 + np.random.normal(size=n)
    v_1_3 = add * v_1_2 + azd * x2 + np.random.normal(size=n)

    v_2_1 = np.random.normal(size=n)
    v_2_2 = add * v_2_1 + azd * y1 + np.random.normal(size=n)
    v_2_3 = add * v_2_2 + azd * y2 + np.random.normal(size=n)


    df = pd.DataFrame(
        {X3: x3,
         Y3: y3,
         Z3: z3,
         U3: u3,
         W3: w3,
         D1: d_1_3,
         D2: d_2_3,
         D3: d_3_3,
         V1: v_1_3,
         V2: v_2_3}
    )

    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices([X3, Y3, Z3, U3, W3, D1, D2, D3, V1, V2])
    ge.add_directed_edge(U3, X3)
    ge.add_directed_edge(U3, Z3)
    ge.add_directed_edge(W3, Y3)
    ge.add_directed_edge(W3, Z3)
    ge.add_directed_edge(X3, Z3)
    ge.add_directed_edge(Y3, Z3)
    ge.add_undirected_edge(U3, X3)
    ge.add_undirected_edge(W3, Y3)

    ge.add_undirected_edge(U3, V1)
    ge.add_undirected_edge(X3, V1)
    ge.add_undirected_edge(W3, V2)
    ge.add_undirected_edge(Y3, V2)

    ge.add_directed_edge(U3, D1)
    ge.add_directed_edge(W3, D1)
    ge.add_directed_edge(X3, D1)
    ge.add_directed_edge(Y3, D1)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_directed_edge(U3, D2)
    ge.add_directed_edge(W3, D2)
    ge.add_directed_edge(X3, D2)
    ge.add_directed_edge(Y3, D2)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_directed_edge(U3, D3)
    ge.add_directed_edge(W3, D3)
    ge.add_directed_edge(X3, D3)
    ge.add_directed_edge(Y3, D3)
    ge.add_directed_edge(V1, D1)
    ge.add_directed_edge(V2, D1)

    ge.add_undirected_edge(Z3, D1)
    ge.add_undirected_edge(Z3, D2)
    ge.add_undirected_edge(Z3, D3)
    ge.add_undirected_edge(D1, D2)
    ge.add_undirected_edge(D1, D3)
    ge.add_undirected_edge(D2, D3)


    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices([X3, Y3, Z3, U3, W3, D1, D2, D3, V1, V2])
    gt.add_directed_edge(X3, Z3)
    gt.add_directed_edge(Y3, Z3)
    gt.add_directed_edge(U3, X3)
    gt.add_directed_edge(W3, Y3)


    gt.add_directed_edge(Z3, D1)
    gt.add_directed_edge(Z3, D2)
    gt.add_directed_edge(Z3, D3)
    gt.add_directed_edge(X3, V1)
    gt.add_directed_edge(Y3, V2)

    return df, ge, gt


def two_chain_v_structure(n):
    df1, ge1, gt1 = chain_v_structure_2(n)
    df2, ge2, gt2 = chain_v_structure_2(n, name_prefix="s_")

    df = pd.concat([df1, df2], axis=1)
    ge = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    ge.add_vertices(ge1.get_vertices())
    ge.add_vertices(ge2.get_vertices())
    for u, v in ge1.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge2.get_directed_edges():
        ge.add_directed_edge(u, v)
    for u, v in ge1.get_undirected_edges():
        ge.add_undirected_edge(u, v)
    for u, v in ge2.get_undirected_edges():
        ge.add_undirected_edge(u, v)

    gt = CompletedPartiallyDirectedAcyclicDifferenceGraph()
    gt.add_vertices(gt1.get_vertices())
    gt.add_vertices(gt2.get_vertices())
    for u, v in gt1.get_directed_edges():
        gt.add_directed_edge(u, v)
    for u, v in gt2.get_directed_edges():
        gt.add_directed_edge(u, v)

    return df, ge, gt


if __name__ == '__main__':
    structure = "two_chain_v_structure"
    n_sample = 1000

    list_pc_f1_a_t = []
    list_pc_f1_o_t = []
    list_pc_f1_a_e = []
    list_pc_f1_o_e = []

    list_fci_f1_a_t = []
    list_fci_f1_o_t = []
    list_fci_f1_a_e = []
    list_fci_f1_o_e = []

    list_restpc_f1_a_t = []
    list_restpc_f1_o_t = []
    list_restpc_f1_a_e = []
    list_restpc_f1_o_e = []

    for iter in range(2):
        print("####### iter ", iter, " #######")
        if structure == "two_chain_v_structure":
            data, ge, gt = chain_v_structure_2(n_sample)

        pc = PC(sparsity=0.05, ci_test=KernelPartialCorrelationTest)
        pc.run(data)
        fci = FCI(sparsity=0.05, ci_test=KernelPartialCorrelationTest)
        fci.run(data)
        restpc = RestPC(sparsity=0.05, ci_test=KernelPartialCorrelationTest)
        restpc.run(data)

        ft_pc_a = f1_score_a(gt, pc.g_hat)
        list_pc_f1_a_t.append(ft_pc_a)
        ft_pc_o = f1_score_o(gt, pc.g_hat)
        list_pc_f1_o_t.append(ft_pc_o)

        fe_pc_a = f1_score_a(ge, pc.g_hat)
        list_pc_f1_a_e.append(fe_pc_a)
        fe_pc_o = f1_score_o(ge, pc.g_hat)
        list_pc_f1_o_e.append(fe_pc_o)


        ft_fci_a = f1_score_a(gt, fci.g_hat)
        list_fci_f1_a_t.append(ft_fci_a)
        ft_fci_o = f1_score_o(gt, fci.g_hat)
        list_fci_f1_o_t.append(ft_fci_o)

        fe_fci_a = f1_score_a(ge, fci.g_hat)
        list_fci_f1_a_e.append(fe_fci_a)
        fe_fci_o = f1_score_o(ge, fci.g_hat)
        list_fci_f1_o_e.append(fe_fci_o)


        ft_restpc_a = f1_score_a(gt, restpc.g_hat)
        list_restpc_f1_a_t.append(ft_restpc_a)
        ft_restpc_o = f1_score_o(gt, restpc.g_hat)
        list_restpc_f1_o_t.append(ft_restpc_o)

        fe_restpc_a = f1_score_a(ge, restpc.g_hat)
        list_restpc_f1_a_e.append(fe_restpc_a)
        fe_restpc_o = f1_score_o(ge, restpc.g_hat)
        list_restpc_f1_o_e.append(fe_restpc_o)

        # print(ft_pc_o, fe_pc_o)
        # print(ft_restpc_o, fe_restpc_o)

    print("################")

    print("PC")
    print("F1-t-a", np.mean(list_pc_f1_a_t), np.var(list_pc_f1_a_t))
    print("F1-t-o", np.mean(list_pc_f1_o_t), np.var(list_pc_f1_o_t))
    print("F1-e-a", np.mean(list_pc_f1_a_e), np.var(list_pc_f1_a_e))
    print("F1-e-o", np.mean(list_pc_f1_o_e), np.var(list_pc_f1_o_e))

    print("FCI")
    print("F1-t-a", np.mean(list_fci_f1_a_t), np.var(list_fci_f1_a_t))
    print("F1-t-o", np.mean(list_fci_f1_o_t), np.var(list_fci_f1_o_t))
    print("F1-e-a", np.mean(list_fci_f1_a_e), np.var(list_fci_f1_a_e))
    print("F1-e-o", np.mean(list_fci_f1_o_e), np.var(list_fci_f1_o_e))

    print("RestPC")
    print("F1-t-a", np.mean(list_restpc_f1_a_t), np.var(list_restpc_f1_a_t))
    print("F1-t-o", np.mean(list_restpc_f1_o_t), np.var(list_restpc_f1_o_t))
    print("F1-e-a", np.mean(list_restpc_f1_a_e), np.var(list_restpc_f1_a_e))
    print("F1-e-o", np.mean(list_restpc_f1_o_e), np.var(list_restpc_f1_o_e))
