from utils.graphs.graphs import Graph
from utils.independence_tests.basics import fisherz_CI_test, gsq_CI_test, chi2_CI_test, kci_CI_test


import pandas as pd
import networkx as nx 
import numpy as np 
from itertools import combinations



######################################################################


class LocPC:
    def __init__(self, data, target_node, sepset=None, CI_test="fisherz", alpha=0.05, forbidden_edges: list = [], display = True):
        """
        Initialize a Local PC (LocPC) algorithm instance for local causal discovery.

        Args:
            data (pd.DataFrame): Dataset containing observed variables.
            target_node (str): Target variable for local discovery.
            sepset (dict, optional): Precomputed separation sets. Keys must be (node1, node2) tuples.
            CI_test (str): Conditional independence test to use. Supported options: "fisherz", "gsq", "kci".
            alpha (float): Significance level for CI tests.
            forbidden_edges (list): List of node pairs (tuples) representing forbidden edges.

        Raises:
            TypeError: If inputs are of incorrect type.
            ValueError: If CI_test is unsupported or if target_node is not in data.
        """
        
        missing_columns = data.columns[data.isnull().any()]
        if not missing_columns.empty and display:
            missing_info = [
                f"{col} (<0.1%)" if data[col].isnull().mean() * 100 < 0.1
                else f"{col} ({data[col].isnull().mean() * 100:.1f}%)"
                for col in missing_columns
                ]
            print("Warning: Missing values found in the following variables:")
            for info in missing_info:
                print(f" - {info}")
            print("Tests are performed by test deletion procedure.")


        # --- Type checks ---
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        if not isinstance(target_node, str):
            raise TypeError("target_node must be a string.")
        if sepset is not None and not isinstance(sepset, dict):
            raise TypeError("sepset must be a dictionary or None.")
        if not isinstance(CI_test, str):
            raise TypeError("CI_test must be a string.")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float.")
        if not isinstance(forbidden_edges, list):
            raise TypeError("forbidden_edges must be a list.")

        if target_node not in data.columns:
            raise ValueError(f"target_node '{target_node}' not found in data columns.")

        self.data = data
        self.target_node = target_node
        self.alpha = alpha
        self.forb = forbidden_edges
        self.sepset = sepset if sepset is not None else {}

        # --- Symmetrize sepset ---
        symmetric_sepset = {}
        for (a, b) in self.sepset:
            symmetric_sepset[(a, b)] = self.sepset[(a, b)]
            if (b, a) not in self.sepset:
                symmetric_sepset[(b, a)] = self.sepset[(a, b)]
        self.sepset = symmetric_sepset

        # --- CI test selection ---
        if CI_test == "fisherz":
            self.CI_test = fisherz_CI_test
        elif CI_test == "gsq":
            self.CI_test = gsq_CI_test
        elif CI_test == "kci":
            self.CI_test = kci_CI_test
        elif CI_test == "chisq" :
            self.CI_test = chi2_CI_test
        else:
            raise ValueError(f"Unsupported CI_test '{CI_test}'. Choose from 'fisherz', 'gsq', 'chisq', or 'kci'.")

        # --- Internal state ---
        self.visited = []
        self.leg = None
        self.neighborhood_h = []
        self.v = list(self.data.columns)
        self.nb_CI_tests = 0
        
        # --- Storage of pvalues and UCs (to orient conflictuals UCs) --- 
        self.pvalues = dict()
        self.UCs = list()
        self.size_tests = list()

    @staticmethod
    def data2observed_data(data: pd.DataFrame, columns=None):
        """
        Filters rows with no missing (NaN) values in selected columns.
        """
        if columns is None:
            columns = data.columns
        return data[data[columns].notna().all(axis=1)]
    
    def _update_skeleton(self, D_set) -> list:
        s = 0
        undirected_edges = self.leg.get_undirected_edges()
        directed_edges = self.leg.get_directed_edges()

        while True:
            cont = False
            
            adj = {}
            
            for D in D_set : # COLOMBO 
                adj[D] = self.leg.get_adjacencies(D)
                
            for D in D_set:
                adj_D = list(adj[D])
                if len(adj_D) >= s:
                    cont = True

                for B in self.leg.get_adjacencies(D):
                    
                    if (D, B) in self.sepset:
                        if self.sepset[(D, B)] is not None : 
                            for edge in [(D, B), (B, D)]:
                                if edge in undirected_edges:
                                    self.leg.remove_undirected_edge(*edge)
                                if edge in directed_edges:
                                    self.leg.remove_directed_edge(*edge)
                        continue
                    else:
                        candidate_neighborhood = list(set(adj_D) - {D, B})

                        for S in combinations(candidate_neighborhood, s):
                            self.nb_CI_tests += 1
                            sel_columns = list(S) + [D,B]
                            test_data = self.data2observed_data(self.data, sel_columns)
                            self.size_tests.append(len(test_data))
                            p_val = self.CI_test(test_data, D, B, list(S), alpha = self.alpha) 
                            if p_val > self.alpha : 
                                self.pvalues[(D,B)] = p_val 
                                self.pvalues[(B,D)] = p_val
                                for edge in [(D, B), (B, D)]:
                                    if edge in undirected_edges:
                                        self.leg.remove_undirected_edge(*edge)
                                    if edge in directed_edges:
                                        self.leg.remove_directed_edge(*edge)
                                self.sepset[(D, B)] = S
                                self.sepset[(B, D)] = S
                                break
            s += 1
            if not cont:
                break
        for D in D_set : 
            for B in [x for x in self.v if x != D] :
                if (D,B) not in self.sepset :
                    self.sepset[(D,B)] = None
                    self.sepset[(B,D)] = None
    
    
    def _find_unshielded_triplets(self):
        """
        Identifies all unshielded triplets in the current graph.

        Returns:
            list of tuples representing unshielded triplets (a, b, c).
        """

        undirected_edges = self.leg.get_undirected_edges()
        unshielded_triplets = []
        for a in self.v :
            for b in self.leg.get_adjacencies(a) :
                for c in self.leg.get_adjacencies(b) : 
                    if c == a : 
                        continue
                    if c not in self.leg.get_adjacencies(a) :
                       unshielded_triplets.append((a,b,c))
        return unshielded_triplets


    def _undirected2collider(self,a,b,c) :
        if (a,b) not in self.forb :
            self.leg.add_directed_edge(a,b)
        if (c,b) not in self.forb :
            self.leg.add_directed_edge(c,b)
        if (a,b) in self.leg.get_undirected_edges() :
            self.leg.remove_undirected_edge(a,b)
        if (b,c) in self.leg.get_undirected_edges() :
            self.leg.remove_undirected_edge(b,c)
        if (b,a) in self.leg.get_undirected_edges() :
            self.leg.remove_undirected_edge(b,a)
        if (c,b) in self.leg.get_undirected_edges() :
            self.leg.remove_undirected_edge(c,b)
        
    def _orient_UCs(self):
        """
        Orients unshielded colliders in the graph according to the
        separation sets and known neighborhood.
        """
        unshielded_triplets = self._find_unshielded_triplets()
        for triplet in unshielded_triplets:
            a, b, c = triplet[0], triplet[1], triplet[2]
            if (c,b,a) in self.UCs : 
                continue 
            if (a in self.neighborhood_h or c in self.neighborhood_h) and (b not in self.sepset[(a, c)]):
                # if (b,c) in self.leg.get_directed_edges() or (b,a) in self.leg.get_directed_edges() :
                #    continue
                self._undirected2collider(a,b,c)
                self.UCs.append((a,b,c))
        
    def _conflicts_max_UCs(self):
        # ref : Improving Accuracy and Scalability of the PC Algorithm by Maximizing P-value1, RAMSEY
        print("Finding unshielded triplets...")
        unshielded_triplets = self._find_unshielded_triplets()
        print(f"Found {len(unshielded_triplets)} unshielded triplets.")

        triplets_pval = dict()
        for triplet in unshielded_triplets:
            a, b, c = triplet
            if (c, b, a) in self.UCs:
                print(f"Skipping symmetric triplet: {(c, b, a)} already in UCs")
                continue
            key = (a, c) if (a, c) in self.pvalues else (c, a)
            if key in self.pvalues:
                triplets_pval[triplet] = self.pvalues[key]
                print(f"Triplet {triplet} has p-value {self.pvalues[key]}")

        sorted_triplets = sorted(triplets_pval, key=lambda x: triplets_pval[x], reverse=True)
        print("Triplets sorted by decreasing p-value.")

        for triplet in sorted_triplets:
            a, b, c = triplet
            print(f"Processing triplet: {triplet}")
            if (a in self.neighborhood_h or c in self.neighborhood_h) and b not in self.sepset.get((a, c), set()):
                print(f"Triplet {triplet} is in local neighborhood and b not in sepset.")
                if ((b, c) in self.leg.get_directed_edges() or (b, a) in self.leg.get_directed_edges()):
                    print(f"Skipping triplet {triplet} due to existing directed edges.")
                    continue
                print(f"Orienting {triplet} as a collider.")
                self._undirected2collider(a, b, c)
                


                
    
                
   

    
    def _meek_rule1(self) -> bool:
        """
        Applies Meek's first rule: orient b—c into b→c when a→b and a is not adjacent to c.

        Returns:
            bool: True if any edge was changed, False otherwise.
        """
        changed = False
        for (Di, Dj) in self.leg.get_directed_edges():
            
            for (a, b) in self.leg.get_undirected_edges():
                if a == Dj:
                    Dk = b
                elif b == Dj:
                    Dk = a
                else:
                    continue  
                
                in_neighborhood = [x not in self.neighborhood_h for x in (Di, Dj, Dk)]
                if in_neighborhood.count(True) > 1:
                    continue

                if Dk not in self.leg.get_adjacencies(Di):
                    undirected_edges = self.leg.get_undirected_edges()
                    for edge in [(Dj, Dk), (Dk, Dj)]:
                        if edge in undirected_edges:
                            self.leg.remove_undirected_edge(*edge)
                            self.leg.add_directed_edge(Dj, Dk)
                            changed = True
                            break  
        return changed


               
    def _meek_rule2(self) -> bool:
        """
        Applies Meek's second rule: orient a—b into a→b if a→c and b→c.

        Returns:
            bool: True if any edge was changed, False otherwise.
        """

        changed = False
        for (Di, Dj) in self.leg.get_directed_edges():
            for Dk in [v for (u, v) in self.leg.get_directed_edges() if u == Dj]:
                # Check if at most one of Di, Dj, Dk is outside the neighborhood
                out_of_neighborhood = [x not in self.neighborhood_h for x in (Di, Dj, Dk)]
                if out_of_neighborhood.count(True) > 1:
                    continue

                if Dk in self.leg.get_adjacencies(Di):
                    if (Di, Dk) in self.leg.get_directed_edges():
                        continue
                    else:
                        if (Di, Dk) in self.leg.get_undirected_edges():
                            self.leg.remove_undirected_edge(Di, Dk)
                            self.leg.add_directed_edge(Di, Dk)
                            changed = True
                        if (Dk, Di) in self.leg.get_undirected_edges():
                            self.leg.remove_undirected_edge(Dk, Di)
                            self.leg.add_directed_edge(Di, Dk)
                            changed = True
        return changed



    def _meek_rule3(self) -> bool:
        """
        Applies Meek's third rule.

        Returns:
            bool: True if any edge was changed, False otherwise.
        """

        changed = False
        
        for Dk in self.neighborhood_h:
            directed_preds = [u for (u, v) in self.leg.get_directed_edges() if v == Dk]
            for i, Dj in enumerate(directed_preds):
                for Dl in directed_preds[i+1:]:
                    if Dj == Dl:
                        continue
                    common_neighbors = set(self.leg.get_adjacencies(Dj)).intersection(
                        self.leg.get_adjacencies(Dl)
                    )
                    for Di in common_neighbors:
                        # Condition "au plus 1 en dehors"
                        nodes_to_check = [Di, Dj, Dl, Dk]
                        out_of_neighborhood = [n not in self.neighborhood_h for n in nodes_to_check]
                        if out_of_neighborhood.count(True) > 1:
                            continue

                        if (
                            ((Di, Dl) in self.leg.get_undirected_edges() or (Dl, Di) in self.leg.get_undirected_edges())
                            and ((Di, Dj) in self.leg.get_undirected_edges() or (Dj, Di) in self.leg.get_undirected_edges())
                        ):
                            if (Di, Dk) in self.leg.get_undirected_edges():
                                self.leg.remove_undirected_edge(Di, Dk)
                                self.leg.add_directed_edge(Di, Dk)
                                changed = True
                            if (Dk, Di) in self.leg.get_undirected_edges():
                                self.leg.remove_undirected_edge(Dk, Di)
                                self.leg.add_directed_edge(Di, Dk)
                                changed = True
        return changed



    def _local_rule(self):
        """
        Applies the local rule to introduce uncertain edges when
        conditions are satisfied for nodes outside the h-hop neighborhood.
        """

        non_neighborhood = set(self.v) - set(self.neighborhood_h)
        if len(non_neighborhood) > 1 :
            for D in self.neighborhood_h :
                neighbors_D = set(self.leg.get_adjacencies(D))
                for A in neighbors_D.intersection(non_neighborhood) :
                    if (A,D) in self.leg.get_directed_edges() or (D,A) in self.leg.get_directed_edges() : 
                        break 
                    LR_cond = True 
                    for W in non_neighborhood - neighbors_D :
                        if A in self.sepset[(D,W)] :
                            LR_cond = False 
                            break 
                    if LR_cond :
                        if (D,A) in self.leg.get_undirected_edges() :
                            self.leg.remove_undirected_edge(D, A) 
                            self.leg.add_uncertain_edge(D, A, edge_type='-||')
                        if (A,D) in self.leg.get_undirected_edges() :
                            self.leg.remove_undirected_edge(A, D) 
                            self.leg.add_uncertain_edge(D, A, edge_type='-||')


    def _apply_meek_rules(self):
        """
        Iteratively applies Meek's rules 1-3 until no more changes occur.
        """

        changed = True
        while changed :
            changed = False
            if self._meek_rule1():
                changed = True
            if self._meek_rule2():
                changed = True
            if self._meek_rule3():
               changed = True

    def runLocPC(self, h, solve_conflicts_max = False):
        """
        Run the Local PC (LocPC) algorithm up to h hops from the target node.

        The algorithm constructs a partially directed graph around the target node
        by iteratively expanding its neighborhood and applying conditional independence tests,
        edge removal, and orientation rules.

        Args:
            h (int): Maximum number of hops from the target node to consider.

        Raises:
            TypeError: If h is not an integer.
            ValueError: If h is negative.
        """
        if not isinstance(h, int):
            raise TypeError("h must be an integer.")
        if h < 0:
            raise ValueError("h must be a non-negative integer.")

        self.leg = Graph()
        self.neighborhood_h = []
        self.leg.add_vertices(self.v)
        
        k = 0
        D_set = {self.target_node}
        self.visited = []

        while k <= h:
            for D in D_set:
                if D not in self.visited:
                    self.visited.append(D)
                    self.neighborhood_h.append(D)
                    for B in [x for x in self.v if x not in self.visited]:
                        # Apply background knowledge (BGK)
                        if (B, D) in self.forb and (D, B) not in self.forb:
                            self.leg.add_directed_edge(D, B)
                        elif (D, B) in self.forb and (B, D) not in self.forb:
                            self.leg.add_directed_edge(B, D)
                        elif (D, B) in self.forb and (B, D) in self.forb:
                            self.sepset[(D,B)] = ["background knowledge"]
                            self.sepset[(B,D)] = ["background knowledge"]
                        else :
                            self.leg.add_undirected_edge(B, D)

            self._update_skeleton(D_set)

            new_D_set = set()
            for D in D_set:
                for x in self.leg.get_adjacencies(D):
                    if x in self.visited or (D,x) in self.leg.get_directed_edges() or (x,D) in self.leg.get_directed_edges() :
                        continue
                    new_D_set.add(str(x))
            D_set = new_D_set
            k += 1

        if not solve_conflicts_max : 
            self._orient_UCs()
        else : 
            self._conflicts_max_UCs()
        self._apply_meek_rules()
        self._local_rule()


    def find_subset_NOC(self):
        """
        Finds a subset of nodes that could satisfy the non-orientability criterion.

        Returns:
            list: subset of nodes connected to the target by undirected edges.
        """

        subset = set([self.target_node])
        queue = [self.target_node]

        while queue:
            current = queue.pop(0)
            neighbors = set(self.leg.get_adjacencies(current)).intersection(self.visited)
            for neighbor in neighbors:
                if ((current, neighbor) not in self.leg.get_directed_edges() and 
                    (neighbor, current) not in self.leg.get_directed_edges()):
                    if neighbor not in subset:
                        subset.add(neighbor)
                        queue.append(neighbor)
        return list(subset)

    def non_orientability_criterion(self, subset):
        """
        Checks whether the given subset satisfies the non-orientability criterion
        in the LEG L^{Y,h}, as defined below:

        A subset D of Neighborhood(Y, h, G) satisfies the non-orientability criterion if,
        for every node D in the subset:
        There is no node A outside of D such that D - A is an undirected edge.

        Returns:
            True if the subset satisfies the criterion, False otherwise.
        """
        for D in subset:
            # Condition 1:
            set_cond_2 = []
            for A in set(self.leg.get_adjacencies(D)) - set(subset):
                if (D, A) in self.leg.get_undirected_edges() or (A, D) in self.leg.get_undirected_edges():
                    return False
                if (D,A) in self.leg.get_uncertain_edges() : 
                    set_cond_2.append(A)
                    if len(set_cond_2) > 1 :
                        return False
        return True
    
    
    
    

def runLocPC_CDE(data, treatment, outcome, alpha=0.05, CI_test="fisherz", linear_estimation=False, known_sepset=None, return_leg=True, forbidden_edges=[], display = True):
    """
    Run the Local PC algorithm and estimate the Causal Direct Effect (CDE) if identifiable.

    This function performs local causal discovery around the outcome variable using LocPC,
    checks for identifiability of the CDE from treatment to outcome, and optionally estimates
    the effect using linear regression.

    Args:
        data (pd.DataFrame): Observational dataset.
        treatment (str): Name of the treatment variable.
        outcome (str): Name of the outcome variable.
        alpha (float): Significance level for conditional independence tests.
        CI_test (str): Conditional independence test to use. Options: "fisherz", "gsq", "kci".
        linear_estimation (bool): If True, estimate the direct effect using linear regression.
        known_sepset (dict, optional): Optional prior knowledge of separation sets.
        return_leg (bool): If True, include the learned LEG graph in the result.
        forbidden_edges (list): List of directed edges (format : (node1, node2) ) that are not allowed.

    Returns:
        dict: {
            'identifiability' (bool),
            'adjustment_set' (list or None),
            'estimated_linear_DE' (float or None),
            'NOC' (bool),
            'leg' (Graph object, optional),
            'exec_time' (float),
            'nb_CI_tests' (int),
            'discovered_hop' (int)
        }

    Raises:
        TypeError: For incorrect input types.
        ValueError: If required variables are missing from the data.
    """
    import pandas as pd
    import time
    from sklearn.linear_model import LinearRegression

    # --- Input checks ---
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if not isinstance(treatment, str) or not isinstance(outcome, str):
        raise TypeError("treatment and outcome must be strings.")
    if treatment not in data.columns:
        raise ValueError(f"treatment '{treatment}' not found in data columns.")
    if outcome not in data.columns:
        raise ValueError(f"outcome '{outcome}' not found in data columns.")
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float.")
    if not isinstance(CI_test, str):
        raise TypeError("CI_test must be a string.")
    if known_sepset is not None and not isinstance(known_sepset, dict):
        raise TypeError("known_sepset must be a dictionary or None.")
    if not isinstance(forbidden_edges, list):
        raise TypeError("forbidden_edges must be a list.")
    if not isinstance(linear_estimation, bool):
        raise TypeError("linear_estimation must be a boolean.")
    if not isinstance(return_leg, bool):
        raise TypeError("return_leg must be a boolean.")

    # --- Initialization ---
    start_time = time.time()
    Y = outcome
    X = treatment
    locpc = LocPC(data=data, target_node=Y, CI_test=CI_test, sepset=known_sepset, alpha=alpha, forbidden_edges=forbidden_edges, display = display)
    identifiability = False
    h = 0
    result = {'NOC': False}

    # --- Run LocPC up to full discovery or identifiability ---
    while True:
        locpc.runLocPC(h)
        if (Y, X) in locpc.leg.get_directed_edges() or X not in locpc.leg.get_adjacencies(Y):
            identifiability = True
            break
        D = locpc.find_subset_NOC()
        if locpc.non_orientability_criterion(D):
            result['NOC'] = True
            break
        if len(locpc.neighborhood_h) == len(locpc.v):
            break
        h += 1

    # --- Final identifiability check (fully oriented) ---
    if not identifiability:
        for n in locpc.leg.get_adjacencies(Y):
            if (n, Y) not in locpc.leg.get_directed_edges() and (Y, n) not in locpc.leg.get_directed_edges():
                break
        else:
            identifiability = True

    # --- Adjustment set ---
    if identifiability:
        adjustment_set = [p for (p, x) in locpc.leg.get_directed_edges() if x == Y]
    else:
        adjustment_set = None
    result['identifiability'] = identifiability
    result['adjustment_set'] = adjustment_set

    # --- Linear estimation ---
    if linear_estimation:
        if identifiability:
            if X not in adjustment_set:
                estimated_linear_DE = 0.0
            else:
                model = LinearRegression()
                X_adjust = data[adjustment_set]
                y_target = data[Y]
                model.fit(X_adjust, y_target)
                estimated_linear_DE = model.coef_[adjustment_set.index(X)]
        else:
            estimated_linear_DE = None
        result['estimated_linear_DE'] = estimated_linear_DE

    # --- Final result ---
    if return_leg:
        result['leg'] = locpc.leg

    result['exec_time'] = time.time() - start_time
    result['nb_CI_tests'] = locpc.nb_CI_tests
    result['discovered_hop'] = h

    return result


