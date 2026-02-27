import pandas as pd
from itertools import combinations

from utils.graphs.partially_specified_graphs import LocalEssentialGraph
from utils.independence_tests.basic import CiTests, FisherZ
from utils.background_knowledge.background_knowledge import BackgroundKnowledge


class LocPC:
    def __init__(self, data: pd.DataFrame, sparsity: float = 0.05, ci_test: CiTests = FisherZ, background_knowledge: BackgroundKnowledge = None):
        self._data = data
        self._sparsity = sparsity
        self._ci_test = ci_test
        self._bk = background_knowledge
        self._knowntests = {}
        
        self._nodes = list(data.columns)

        self.nb_ci_tests = 0
        self.sepset = dict()
    
    def _update_skeleton(self, D_set):
        for d in D_set: 
            for b in [x for x in self._nodes if (x not in self._visited) and x!=d]: 
                self.leg.add_undirected_edge(d,b)
        s = 0
        repeat = True 
        while repeat: 
            repeat = False       
            adj = {x: self.leg.get_adjacencies(x) for x in self._nodes}
            for d in D_set: 
                self._visited.add(d)
                if len(adj[d]) - 1 >= s:
                    repeat = True
                    for b in adj[d]:
                        for S in combinations([a for a in adj[d] if a != b], s):
                            if (d,b,S) in self._knowntests:
                                p_val = self._knowntests[(d,b,S)]
                            else:
                                test = self._ci_test(d,b,list(S))
                                self.nb_ci_tests += 1
                                p_val = test.get_pvalue(self._data)
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
                for w in set(self._nodes) - self._visited - set(self.leg.get_adjacencies(d)):
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
        self._orientation()