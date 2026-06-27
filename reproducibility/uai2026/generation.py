import numpy as np
from pyciphod.causal_reasoning.summary_causal_graph.micro_queries.direct_effect import CDE_is_identifiable, NDE_is_identifiable
from pyciphod.utils.graphs.partially_specified_graphs import SummaryCausalGraph
from pyciphod.utils.graphs.temporal_graphs import FtAcyclicDirectedMixedGraph, create_random_ft_admg

n_graphs = 1000
num_ts_list = [n for n in range(2,16)]
p_edge_list = [p/100 for p in range(15,0,-1)]
causally_stationnary = False
max_delay_list = [2]
allow_instatenous = True
allow_unmeasured_confounding = True
seed = 0
res_CDE = []
res_NDE = []
for p_edge in p_edge_list:
    res_CDE_1 = []
    res_NDE_1 = []
    for num_ts in num_ts_list:
        res_CDE_2 = 0
        res_NDE_2 = 0
        for i in range(n_graphs):
            seed += 1
            np.random.seed(seed)
            max_delay = np.random.choice(max_delay_list) 
            ftadmg = create_random_ft_admg(num_ts, p_edge, causally_stationnary, max_delay,
                                            allow_instatenous, allow_unmeasured_confounding, seed = seed)
            scg = ftadmg.get_summary_causal_graph()
            vertices = list(scg.get_vertices())
            X = np.random.choice(vertices)
            Y = np.random.choice(vertices)
            gamma =  np.random.randint(0, scg.get_lag_max() + 1)
            if CDE_is_identifiable(scg, X, Y, gamma):
                res_CDE_2 += 1
            if NDE_is_identifiable(scg, X, Y, gamma):
                res_NDE_2 += 1
        res_CDE_1.append(res_CDE_2/n_graphs)
        res_NDE_1.append(res_NDE_2/n_graphs)
    res_CDE.append(res_CDE_1)
    res_NDE.append(res_NDE_1)
np.savetxt('pyCIPHOD/reproducibility/uai2026/results_CDE.txt', res_CDE)
np.savetxt('pyCIPHOD/reproducibility/uai2026/results_NDE.txt', res_NDE)
            

