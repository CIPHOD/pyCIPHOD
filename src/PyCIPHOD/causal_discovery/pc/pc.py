import pandas as pd
from itertools import combinations

from utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicGraph
from utils.independence_tests.basic import CiTests, FisherZ
from utils.background_knowledge.background_knowledge import BackgroundKnowledge


class PC:
    """
    Implements the PC algorithm for causal discovery from observational data.

    Attributes:
        _data (pd.DataFrame): Observational data.
        _sparsity (float): Significance threshold for conditional independence tests.
        _ci_test (class): A test of the conditional independence test class (e.g., FisherZ).
        _nodes (list): List of variable names in the data.
        cpdag (CompletedPartiallyDirectedAcyclicGraph).
        nb_ci_tests (int): Number of CI tests performed.
        sepset (dict): Separation sets for node pairs in the skeleton.
    """

    def __init__(self, data: pd.DataFrame,  sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None, twd = False):
        """Initialize PC algorithm with data, sparsity threshold, and CI test."""
        self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge
        self._twd = twd # Test wise deletion
        self._nodes = list(data.columns)
        self.performed_tests = set()
        
        self.cpdag = CompletedPartiallyDirectedAcyclicGraph()
        
        self.nb_ci_tests = 0
        self.sepset = dict()

    def _skeleton(self):
        """
        Construct the skeleton of the graph using an order-independent approach
        following Colombo & Maathuis (2014). Iteratively removes edges based on CI tests.
        """
        self.cpdag.add_undirected_edges_from(list(combinations(self._nodes, 2)))
        data_test = self._data
        s = 0
        repeat = True
        while repeat:
            repeat = False
            adj = {x: self.cpdag.get_adjacencies(x) for x in self._nodes}
            for x in self._nodes:
                if len(adj[x]) - 1 >= s:
                    repeat = True
                    for y in adj[x]:
                        for S in combinations([a for a in adj[x] if a != y], s):
                            test = self._ci_test(x, y, list(S))
                            self.performed_tests.add((x,y,S))
                            self.nb_ci_tests += 1
                            if self._twd:
                                data_test = self._data.dropna(subset=[x, y] + list(S))
                            if test.get_pvalue(data_test) > self._sparsity:
                                self.cpdag.remove_undirected_edge(x, y)
                                self.sepset[(x, y)] = self.sepset[(y, x)] = S
                                break
            s += 1
            
    
    def _apply_background_knowledge(self):
        """
        Apply background knowledge constraints to the CPDAG:
        1) Remove forbidden edges and add mandatory edges in the skeleton.
        2) Orient edges according to mandatory and forbidden orientations if present.
        """
        if not self._bk:
            return 

        # --- Step 1: enforce mandatory and forbidden edges ---
        # Remove forbidden edges if present
        # for u, v in self._bk.get_forbidden_edges():
        #     if (u, v) in self.cpdag.get_undirected_edges() or (v, u) in self.cpdag.get_undirected_edges():
        #         self.cpdag.remove_undirected_edge(u, v)
        #         self.cpdag.remove_undirected_edge(v, u)

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.cpdag.get_undirected_edges() and (v, u) not in self.cpdag.get_undirected_edges():
                self.cpdag.add_undirected_edge(u, v)

        # --- Step 2: enforce orientations ---
        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if (u, v) in self.cpdag.get_undirected_edges() or (v, u) in self.cpdag.get_undirected_edges():
                self.cpdag.remove_undirected_edge(u, v)
                self.cpdag.remove_undirected_edge(v, u)
                self.cpdag.add_directed_edge(u, v)

        # Forbidden orientations
        for u, v in self._bk.get_forbidden_orientations():
            if (u, v) in self.cpdag.get_undirected_edges():
                # Orient as v -> u instead
                self.cpdag.remove_undirected_edge(u, v)
                self.cpdag.add_directed_edge(v, u)
            elif (v, u) in self.cpdag.get_undirected_edges():
                # Orient as u -> v instead
                self.cpdag.remove_undirected_edge(v, u)
                self.cpdag.add_directed_edge(u, v)
            

    def _uc_rule(self):
        """
        Apply the Unshielded Collider (UC) rule:
        For each unshielded triple x - y - z with x and z not adjacent, 
        orient x -> y <- z if y not in sepset(x, z).
        """
        adj = {x: self.cpdag.get_adjacencies(x) for x in self._nodes}
        for x in self._nodes:
            for y in adj[x]:
                for z in adj[y]:
                    if z == x or z in adj[x]:
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.cpdag.remove_undirected_edge(x, y)
                        self.cpdag.remove_undirected_edge(y, z)
                        self.cpdag.add_directed_edges_from([(x, y), (z, y)])

    def _meek_rules(self):
        """
        Apply Meek's orientation rules iteratively using only edge sets:
        Rule 1, Rule 2, Rule 3 for propagating orientations in a CPDAG.
        Returns True if any edge was oriented, False otherwise.
        """
        changed = False
        adj = {x: self.cpdag.get_adjacencies(x) for x in self._nodes}
        
        for x in self._nodes:
            for y in adj[x]:
                # Rule 1:
                for z in set(adj[y]) - set(adj[x]):
                    if (x,y) in self.cpdag.get_directed_edges() and (y,z) in self.cpdag.get_undirected_edges():
                        changed = True
                        self.cpdag.remove_undirected_edge(z, y)
                        self.cpdag.add_directed_edge(y, z)
                # Rule 2:
                for z in set(adj[y]) & set(adj[x]):
                    if (x,y) in self.cpdag.get_directed_edges() and (y,z) in self.cpdag.get_directed_edges() and (x,z) in self.cpdag.get_undirected_edges():
                        changed = True
                        self.cpdag.remove_undirected_edge(x, z)
                        self.cpdag.add_directed_edge(x, z)
        
        # Rule 3: 
        for x in self._nodes:
            for y in adj[x]:
                if (x, y) not in self.cpdag.get_directed_edges():
                    continue
                for z in set(adj[y]) - set(adj[x]):
                    if (z, y) not in self.cpdag.get_directed_edges():
                        continue
                    for w in set(adj[x]) & set(adj[y]) & set(adj[z]):
                        undirected_triplet = [(w, y), (w, x), (z, w)]
                        if all(edge in self.cpdag.get_undirected_edges() for edge in undirected_triplet):
                            changed = True
                            self.cpdag.remove_undirected_edge(w, y)
                            self.cpdag.add_directed_edge(w, y)

        return changed

    def _orientation(self):
        """Orient edges using the UC rule and iterative Meek rules until convergence."""
        self._uc_rule()
        repeat = True
        while repeat:
            repeat = self._meek_rules()

    def run(self):
        """
        Execute the full PC algorithm:
        1. Construct skeleton (order-independent)
        2. Apply UC and Meek orientation rules
        """
        self._skeleton()
        self._apply_background_knowledge()
        self._orientation()