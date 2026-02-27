import pandas as pd
from itertools import combinations

from utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicGraph
from utils.independence_tests.basic import CiTests, FisherZ



class PC:
    """
    Implements the PC algorithm for causal discovery from data.
    """
    def __init__(self, data : pd.DataFrame, sparsity, ci_test = CiTests):
        self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test

        self._nodes = list(data.columns)
        self._cpdag = CompletedPartiallyDirectedAcyclicGraph()
        self._cpdag.add_undirected_edges_from(list(combinations(self._nodes, 2)))
        
        self.nb_ci_tests = 0
        self.sepset = dict()
        
    def _skeleton(self) :
        s = 0
        repeat = True
        while repeat : 
            repeat = False
            adj = dict()
            for x in self._nodes : 
                adj[x] = self._cpdag.get_adjacencies(x)
            for x in self._nodes :
                if len(adj[x]) - 1 >= s :
                    repeat = True  
                    for y in adj[x] : 
                        for S in combinations([a for a in adj[x] if a!= y], s):
                            test = self._ci_test(x, y, list(S))
                            self.nb_ci_tests += 1
                            if test.get_pvalue(self._data) > self._sparsity :
                                self._cpdag.remove_undirected_edge(x, y)
                                self.sepset[(x,y)] = S
                                self.sepset[(y,x)] = S
            s += 1
            