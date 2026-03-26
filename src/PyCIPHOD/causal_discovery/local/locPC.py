import pandas as pd
from itertools import combinations

from utils.graphs.partially_specified_graphs import LocalEssentialGraph
from utils.independence_tests.basic import CiTests, FisherZ
from utils.background_knowledge.background_knowledge import BackgroundKnowledge


class LocPC:
    def __init__(self, data: pd.DataFrame, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge = BackgroundKnowledge(), twd = False):
        self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge
        self._knowntests = {}
        self._twd = twd
        self._nodes = list(data.columns)
        
        self.performed_tests = set()
        
        bk_nd = background_knowledge.get_non_descendants()
        self._non_descendants = {node: bk_nd.get(node, set()) for node in self._nodes}
        
        self.nb_ci_tests = 0
        self.sepset = dict()
    
    def _update_skeleton(self, D_set):
        for d in D_set: 
            for b in [x for x in self._nodes if (x not in self._visited) and x!=d]: 
                self.leg.add_undirected_edge(d,b)
        data_test = self._data
        s = 0
        repeat = True 
        while repeat: 
            repeat = False       
            adj = {x: self.leg.get_adjacencies(x) for x in self._nodes}
            for d in D_set: 
                self._visited.add(d)
                if len(adj[d]) - 1 >= s:
                    repeat = True
                    for b in [x for x in adj[d] if not (x in self._visited and d in self._non_descendants[x])]:
                        for S in combinations([a for a in adj[d] if a != b], s):
                            if (d,b,S) in self._knowntests:
                                p_val = self._knowntests[(d,b,S)]
                            else:
                                test = self._ci_test(d,b,list(S))
                                self.performed_tests.add((d,b,S))
                                self.nb_ci_tests += 1
                                if self._twd:
                                    data_test = self._data.dropna(subset=[d, b] + list(S))
                                p_val = test.get_pvalue(data_test)
                                self._knowntests[(d,b,S)] = self._knowntests[(b,d,S)] = p_val
                            if p_val > self._sparsity: 
                                self.leg.remove_undirected_edge(d,b)
                                self.sepset[(d, b)] = self.sepset[(b, d)] = S
                                break
            s += 1
        
        D_new = {
            n
            for d in D_set
            for n in self.leg.get_adjacencies(d)
        }

        return D_new - self._visited
    
    def _apply_background_knowledge(self):
        """
        Apply background knowledge constraints to the leg:
        1) Remove forbidden edges and add mandatory edges in the skeleton.
        2) Orient edges according to mandatory and forbidden orientations if present.
        """
        if not self._bk:
            return 

        # --- Step 1: enforce mandatory and forbidden edges ---
        # Remove forbidden edges if present
        # for u, v in self._bk.get_forbidden_edges():
        #     if (u, v) in self.leg.get_undirected_edges() or (v, u) in self.leg.get_undirected_edges():
        #         self.leg.remove_undirected_edge(u, v)
        #         self.leg.remove_undirected_edge(v, u)

        # Add mandatory edges if missing
        for u, v in self._bk.get_mandatory_edges():
            if (u, v) not in self.leg.get_undirected_edges() and (v, u) not in self.leg.get_undirected_edges():
                self.leg.add_undirected_edge(u, v)

        # --- Step 2: enforce orientations ---
        # Mandatory orientations
        for u, v in self._bk.get_mandatory_orientations():
            if (u, v) in self.leg.get_undirected_edges() or (v, u) in self.leg.get_undirected_edges():
                self.leg.remove_undirected_edge(u, v)
                self.leg.remove_undirected_edge(v, u)
                self.leg.add_directed_edge(u, v)

        # Forbidden orientations
        for u, v in self._bk.get_forbidden_orientations():
            if (u, v) in self.leg.get_undirected_edges():
                # Orient as v -> u instead
                self.leg.remove_undirected_edge(u, v)
                self.leg.add_directed_edge(v, u)
            elif (v, u) in self.leg.get_undirected_edges():
                # Orient as u -> v instead
                self.leg.remove_undirected_edge(v, u)
                self.leg.add_directed_edge(u, v)
            

    def _uc_rule(self):
        adj = {x: self.leg.get_adjacencies(x) for x in self._nodes}
        for x in self._nodes:
            for y in adj[x]:
                for z in adj[y]:
                    if z == x or z in adj[x] or any(node not in self._visited for node in (x, y, z)):
                        continue
                    if y not in self.sepset.get((x, z), []):
                        self.leg.remove_undirected_edge(x, y)
                        self.leg.remove_undirected_edge(y, z)
                        self.leg.add_directed_edges_from([(x, y), (z, y)])
                        
    
    def _meek_rules(self):
        changed = False
        adj = {x: self.leg.get_adjacencies(x) for x in self._nodes}
        
        for x in self._visited:
            for y in set(adj[x]) & self._visited:
                # Rule 1:
                for z in (set(adj[y]) - set(adj[x])) & self._visited:
                    if any(node not in self._visited for node in (x, y, z)):
                        continue
                    if (x,y) in self.leg.get_directed_edges() and (y,z) in self.leg.get_undirected_edges():
                        changed = True
                        self.leg.remove_undirected_edge(z, y)
                        self.leg.add_directed_edge(y, z)
                # Rule 2:
                for z in set(adj[y]) & set(adj[x]) & self._visited:
                    if any(node not in self._visited for node in (x, y, z)):
                        continue
                    if (x,y) in self.leg.get_directed_edges() and (y,z) in self.leg.get_directed_edges() and (x,z) in self.leg.get_undirected_edges():
                        changed = True
                        self.leg.remove_undirected_edge(x, z)
                        self.leg.add_directed_edge(x, z)
        
        # Rule 3: 
        for x in self._visited:
            for y in set(adj[x]) & self._visited:
                if (x, y) not in self.leg.get_directed_edges():
                    continue
                for z in (set(adj[y]) - set(adj[x])) & self._visited:
                    if (z, y) not in self.leg.get_directed_edges():
                        continue
                    for w in set(adj[x]) & set(adj[y]) & set(adj[z]) & self._visited:
                        if any(node not in self._visited for node in (x, y, z, w)):
                            continue
                        undirected_triplet = [(w, y), (w, x), (z, w)]
                        if all(edge in self.leg.get_undirected_edges() for edge in undirected_triplet):
                            changed = True
                            self.leg.remove_undirected_edge(w, y)
                            self.leg.add_directed_edge(w, y)

        return changed
    
    def _nnc_rule(self):
        for d in self._visited:
            for a in set(self.leg.get_adjacencies(d)) - self._visited:
                if (d,a) not in self.leg.get_undirected_edges():
                    continue 
                nnc = True
                outside_nodes = set(self._nodes) - self._visited - set(self.leg.get_adjacencies(d))
                if len(outside_nodes) == 0:
                    continue
                for w in outside_nodes:
                    if a in self.sepset[(d,w)]:
                        nnc = False
                        break
                if nnc:
                    self.leg.remove_undirected_edge(d,a)
                    self.leg.add_uncertain_edge(d,a,'-||')
                    
    def _orientation(self):
        self._uc_rule()
        repeat = True
        while repeat:
            repeat = self._meek_rules()
        self._nnc_rule()
        
    def run(self, target, hop: int = 0):
        target = [target]
        self.leg = LocalEssentialGraph()
        self.leg.add_vertices(self._nodes)
        self._visited = set()
        self._target = target
        k = 0
        D_set = set(self._target)
        while k <= hop:
            D_new = self._update_skeleton(D_set)
            if self._visited == set(self._nodes):
                break
            D_set = D_new
            k+=1
        self._apply_background_knowledge()
        self._orientation()
        
    
    ### LOCPC-CDE PART 
    
    def _find_subset_NOC(self):
        # Add assert if no target to run the algorithm before
        subset = set(self._target)
        queue = self._target
        while queue:
            current = queue.pop(0)
            neighbors = set(self.leg.get_adjacencies(current)) & self._visited
            for neighbor in neighbors:
                if ((current, neighbor) not in self.leg.get_directed_edges() and 
                    (neighbor, current) not in self.leg.get_directed_edges()):
                    if neighbor not in subset:
                        subset.add(neighbor)
                        queue.append(neighbor)
        return set(subset)
    
        
    def _non_orientability_criterion(self, subset):
        for d in subset: 
            set_condition_2 = []
            for a in set(self.leg.get_adjacencies(d)) - subset:
                # Condition 1:
                if (d,a) in self.leg.get_undirected_edges():
                    return False
                # Condition 2:
                if (d,a) in self.leg.get_uncertain_edges():
                    set_condition_2.append(a)
                    if len(set_condition_2) > 1:
                        return False
        return True   
    
    
    def run_locPC_CDE(self, treatment: str, outcome: str) -> dict:
        h = 0
        identifiability_CDE = False
        
        # Run the algorithm incrementally with hops
        while h <= len(self._nodes):
            self.run(outcome, hop=h)
            subset_noc = self._find_subset_NOC()
            if self._non_orientability_criterion(subset_noc):
                break # Stop if non-orientability criterion met
            if (outcome, treatment) in self.leg.get_directed_edges() or treatment not in self.leg.get_adjacencies(outcome):
                identifiability_CDE = True
                break # Check on the relation between treatment and outcome
            if len(self._visited) == len(self._nodes):
                break # Stop if all nodes visited
            h += 1
        
        if not identifiability_CDE: # Extra identifiability check if not flagged
            for n in self.leg.get_adjacencies(outcome):
                if (n, outcome) not in self.leg.get_directed_edges() and (outcome, n) not in self.leg.get_directed_edges():
                    break
            else:
                identifiability_CDE = True # All adjacencies oriented, CDE identifiable
        
        adjustment_set = [p for (p, x) in self.leg.get_directed_edges() if x == outcome] if identifiability_CDE else None # Compute adjustment set if identifiable
        
        return {
            "treatment": treatment,
            "outcome": outcome,
            "identifiability": identifiability_CDE,
            "adjustment_set": adjustment_set
    }