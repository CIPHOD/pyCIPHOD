from abc import ABC, abstractmethod
from typing import Type, Optional
import pandas as pd
from itertools import combinations

from pyciphod.utils.graphs.partially_specified_graphs import CompletedPartiallyDirectedAcyclicGraph
from pyciphod.utils.graphs.meek_rules import meek_rule_1, meek_rule_2, meek_rule_3
from pyciphod.utils.stat_tests.independence_tests import CiTests, FisherZTest as FisherZ
from pyciphod.utils.graphs.background_knowledge import BackgroundKnowledge


class ConstraintBased(ABC):
    def __init__(self, sparsity: float = 0.05, ci_test: Type[CiTests] = FisherZ, background_knowledge: Optional[BackgroundKnowledge] = None, twd: Optional[bool] = False):
        # self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge if background_knowledge is not None else BackgroundKnowledge()
        self._twd = twd # Test wise deletion
        self.performed_tests = set()
        self.nb_ci_tests = 0
        self.sepset = dict()
        self.g_hat = self._initialize_graph() # Graph representation (e.g., CPDAG for PC, PAG for FCI)

    @abstractmethod
    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        pass

    @abstractmethod
    def _apply_background_knowledge(self):
        """Apply background knowledge constraints to the graph structure."""
        pass

    @abstractmethod
    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None, dependence_matrix:Optional[pd.DataFrame] = None, effective_n:Optional[pd.DataFrame] = None):
        """Construct the skeleton of the graph using CI tests."""
        pass

    @abstractmethod
    def _orientation(self):
        """Orient edges using rules based on the skeleton and separation sets."""
        pass

    def run(self, data: pd.DataFrame = None, dependence_matrix:Optional[pd.DataFrame] = None, effective_n:Optional[pd.DataFrame] = None):
        """Execute the full constraint-based algorithm."""
        self._skeleton(data, dependence_matrix=dependence_matrix, effective_n=effective_n)
        self._apply_background_knowledge()
        self._orientation()


class PC(ConstraintBased):
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

    def __init__(self, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None, twd = False):
        super().__init__(sparsity, ci_test, background_knowledge, twd)
        # Additional attributes for FCI can be defined here, such as the PAG representation and rules for handling latent confounders and selection bias.

    def _initialize_graph(self):
        """Return the graph object used by the algorithm."""
        return CompletedPartiallyDirectedAcyclicGraph()

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None, dependence_matrix:Optional[pd.DataFrame] = None, effective_n:Optional[pd.DataFrame] = None):
        """
        Construct the skeleton of the graph using an order-independent approach
        following Colombo & Maathuis (2014). Iteratively removes edges based on CI tests.
        """
        if max_sepset_size is None:
            max_sepset_size = len(data.columns) - 2
        nodes = list(data.columns)
        self.g_hat.add_undirected_edges_from(list(combinations(nodes, 2)))
        data_test = data
        s = 0
        repeat = True
        while repeat and s <= max_sepset_size:
            repeat = False
            adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
            treated = []
            for x in nodes:
                if len(adj[x]) - 1 >= s:
                    repeat = True
                    for y in sorted(adj[x]):
                        for S in combinations([a for a in sorted(adj[x]) if a != y], s):
                            if (y, x, S) not in treated:
                                treated.append((x, y, S))
                                if dependence_matrix is not None:
                                    test = self._ci_test(x, y, list(S), self._twd, copula_matrix=dependence_matrix, effective_n=effective_n)
                                else:
                                    test = self._ci_test(x, y, list(S), self._twd)
                                self.performed_tests.add((x,y,S))
                                self.nb_ci_tests += 1
                                if self._twd:
                                    data_test = data.dropna(subset=[x, y] + list(S))
                                try:
                                    pval = test.get_pvalue(data_test)
                                except:
                                    print(test, "relies on get_pvalue_by_permutation for p-value estimation")
                                    pval = test.get_pvalue_by_permutation(data_test)
                                print(x, y, S, pval)
                                if pval > self._sparsity:
                                    self.g_hat.remove_undirected_edge(x, y)
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

        # --- Step 1: enforce mandatory edges ---

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.g_hat.get_undirected_edges() and (v, u) not in self.g_hat.get_undirected_edges():
                self.g_hat.add_undirected_edge(u, v)

        # --- Step 2: enforce orientations ---
        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if (u, v) in self.g_hat.get_undirected_edges() or (v, u) in self.g_hat.get_undirected_edges():
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.remove_undirected_edge(v, u)
                self.g_hat.add_directed_edge(u, v)

        # Forbidden orientations
        for u, v in self._bk.get_forbidden_orientations():
            if (u, v) in self.g_hat.get_undirected_edges():
                # Orient as v -> u instead
                self.g_hat.remove_undirected_edge(u, v)
                self.g_hat.add_directed_edge(v, u)
            elif (v, u) in self.g_hat.get_undirected_edges():
                # Orient as u -> v instead
                self.g_hat.remove_undirected_edge(v, u)
                self.g_hat.add_directed_edge(u, v)

    def _uc_rule(self):
        """
        Apply the Unshielded Collider (UC) rule:
        For each unshielded triple x - y - z with x and z not adjacent, 
        orient x -> y <- z if y not in sepset(x, z).
        """
        nodes = self.g_hat.get_vertices()
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        for x in nodes:
            for y in sorted(adj[x]):
                for z in sorted(adj[y]):
                    if z == x or z in sorted(adj[x]):
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.g_hat.remove_undirected_edge(x, y)
                        self.g_hat.remove_undirected_edge(y, z)
                        self.g_hat.add_directed_edges_from([(x, y), (z, y)])

    def _apply_meek_rules(self):
        """
        Apply Meek's orientation rules iteratively using only edge sets:
        Rule 1, Rule 2, Rule 3 for propagating orientations in a CPDAG.
        Returns True if any edge was oriented, False otherwise.
        """
        nodes = self.g_hat.get_vertices()
        changed = False
        adj = {x: self.g_hat.get_adjacencies(x) for x in nodes}
        
        for x in nodes:
            for y in sorted(adj[x]):
                # Rule 1:
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if meek_rule_1(self.g_hat, x, y, z):
                    # if (x,y) in self.g_hat.get_directed_edges() and (y,z) in self.g_hat.get_undirected_edges():
                        changed = True
                        self.g_hat.remove_undirected_edge(z, y)
                        self.g_hat.add_directed_edge(y, z)
                # Rule 2:
                for z in sorted(set(adj[y]) & set(adj[x])):
                    if meek_rule_2(self.g_hat, x, y, z):
                    # if (x,y) in self.g_hat.get_directed_edges() and (y,z) in self.g_hat.get_directed_edges() and (x,z) in self.g_hat.get_undirected_edges():
                        changed = True
                        self.g_hat.remove_undirected_edge(x, z)
                        self.g_hat.add_directed_edge(x, z)
        
        # Rule 3: 
        for x in nodes:
            for y in adj[x]:
                if (x, y) not in self.g_hat.get_directed_edges():
                    continue
                for z in sorted(set(adj[y]) - set(adj[x])):
                    if (z, y) not in self.g_hat.get_directed_edges():
                        continue
                    for w in sorted(set(adj[x]) & set(adj[y]) & set(adj[z])):
                        if meek_rule_3(self.g_hat, x, y, z, w):
                        # undirected_triplet = [(w, y), (w, x), (z, w)]
                        # if all(edge in self.g_hat.get_undirected_edges() for edge in undirected_triplet):
                            changed = True
                            self.g_hat.remove_undirected_edge(w, y)
                            self.g_hat.add_directed_edge(w, y)
        return changed

    def _orientation(self):
        """Orient edges using the UC rule and iterative Meek rules until convergence."""
        self._uc_rule()
        repeat = True
        while repeat:
            # self.g_hat, changed = apply_meek_rules(self.g_hat)
            repeat = self._apply_meek_rules()


class RestPC(PC):
    def __init__(self, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None, twd = False):
        super().__init__(sparsity, ci_test, background_knowledge, twd)
        # Additional attributes for RestPC can be defined here, such as the representation of the rest graph and rules for handling selection bias.

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = 1, dependence_matrix:Optional[pd.DataFrame] = None, effective_n:Optional[pd.DataFrame] = None):
        super()._skeleton(data, max_sepset_size=0, dependence_matrix=dependence_matrix, effective_n=effective_n)  # Run only one iteration of the skeleton phase of PC

    def _orientation(self):
        """Orient edges using the UC rule."""
        self._uc_rule()


class FCI(ConstraintBased):
    def __init__(self, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None, twd = False):
        super().__init__(sparsity, ci_test, background_knowledge, twd)
        # Additional attributes for FCI can be defined here, such as the PAG representation and rules for handling latent confounders and selection bias.

    def _skeleton(self, data: pd.DataFrame = None, max_sepset_size: int = None, dependence_matrix:Optional[pd.DataFrame] = None, effective_n:Optional[pd.DataFrame] = None):
        # TODO
        1


    def _orientation(self):
        # TODO
        1

