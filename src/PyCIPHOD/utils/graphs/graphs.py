import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 


class Graph:
    def __init__(self):
        # super().__init__()
        self._directed_g = nx.DiGraph()
        self._confounded_g = nx.Graph()
        self._undirected_g = nx.Graph()
        self._uncertain_g = nx.DiGraph()

        self._g = nx.MultiDiGraph()

        self._directed_g.nodes = self._g.nodes
        self._confounded_g.nodes = self._g.nodes
        self._undirected_g.nodes = self._g.nodes
        self._uncertain_g.nodes = self._g.nodes


        self._list_certain_edge_types = ['<->', '->', '-']
        self._list_uncertain_edge_types = ['*-o', '*->', '*-', '--', '-->', '-||']

    def add_vertex(self, vertex: str) -> None:
        if vertex not in self._g.nodes:
            self._g.add_node(vertex)

    def add_vertices(self, vertices_list: list) -> None:
        for vertex in vertices_list:
            self.add_vertex(vertex)

    def get_vertices(self):
        return set(self._g.nodes)

    def add_directed_edge(self, vertex_i: str, vertex_j: str) -> None:
        # raise NotImplementedError("Please Implement this method")
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        self._directed_g.add_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_i, vertex_j, key='->')
        # if (vertex_i, vertex_j) not in self.directed_edges:
        #     self.add_vertices([vertex_i, vertex_j])
        #     self._g.add_edge(vertex_i, vertex_j, type='->')
        #     self.directed_edges.append((vertex_i, vertex_j))
        #     # self.directed_edges = [(vertex_i, vertex_j) for vertex_i, vertex_j, attrs in self.edges(data=True) if attrs.get("type") == '->']
        # else:
        #     print("Warning: Edge already exists")

    def add_directed_edges_from(self, edge_list: list) -> None:
        for (vertex_i, vertex_j) in edge_list:
            self.add_directed_edge(vertex_i, vertex_j)

    def remove_directed_edge(self, vertex_i: str, vertex_j: str) -> None:
        if self._directed_g.has_edge(vertex_i, vertex_j):
            self._directed_g.remove_edge(vertex_i, vertex_j)
        if self._g.has_edge(vertex_i, vertex_j, key='->'):
            self._g.remove_edge(vertex_i, vertex_j, key='->')

    def add_confounded_edge(self, vertex_i: str, vertex_j: str) -> None:
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        self._confounded_g.add_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_i, vertex_j, key='<->')
        self._g.add_edge(vertex_j, vertex_i, key='<->')

    def add_confounded_edges_from(self, edge_list: list) -> None:
        for (vertex_i, vertex_j) in edge_list:
            self.add_confounded_edge(vertex_i, vertex_j)

    def remove_confounded_edge(self, vertex_i: str, vertex_j: str) -> None:
        if self._confounded_g.has_edge(vertex_i, vertex_j):
            self._confounded_g.remove_edge(vertex_i, vertex_j)
        
        if self._g.has_edge(vertex_i, vertex_j, key='<->'):
            self._g.remove_edge(vertex_i, vertex_j, key='<->')
        if self._g.has_edge(vertex_j, vertex_i, key='<->'):
            self._g.remove_edge(vertex_j, vertex_i, key='<->')

    def add_undirected_edge(self, vertex_i: str, vertex_j: str) -> None:
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        self._undirected_g.add_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_i, vertex_j, key='-')
        self._g.add_edge(vertex_j, vertex_i, key='-')

    def add_undirected_edges_from(self, edge_list: list) -> None:
        for (vertex_i, vertex_j) in edge_list:
            self.add_undirected_edge(vertex_i, vertex_j)

    def remove_undirected_edge(self, vertex_i: str, vertex_j: str) -> None:
        if self._undirected_g.has_edge(vertex_i, vertex_j):
            self._undirected_g.remove_edge(vertex_i, vertex_j)
        if self._g.has_edge(vertex_i, vertex_j, key='-'):
            self._g.remove_edge(vertex_i, vertex_j, key='-')
        if self._g.has_edge(vertex_j, vertex_i, key='-'):
            self._g.remove_edge(vertex_j, vertex_i, key='-')

    def add_uncertain_edge(self, vertex_i: str, vertex_j: str, edge_type='*-o') -> None:
        """

        :param vertex_i:
        :param vertex_j:
        :param edge_type: '*-o' or '*->' or '*-' or '--' or '-->' or '-||'
        :return:
        """
        self.add_vertex(vertex_i)
        self.add_vertex(vertex_j)
        assert edge_type in self._list_uncertain_edge_types
        self._uncertain_g.add_edge(vertex_i, vertex_j, type=edge_type)
        self._g.add_edge(vertex_i, vertex_j, key=edge_type)

    def remove_uncertain_edge(self, vertex_i: str, vertex_j: str) -> None:
        if self._uncertain_g.has_edge(vertex_i, vertex_j):
            self._uncertain_g.remove_edge(vertex_i, vertex_j)
        if self._uncertain_g.has_edge(vertex_j, vertex_i):
            self._uncertain_g.remove_edge(vertex_j, vertex_i)

        for edge_type in self._list_uncertain_edge_types:
            if self._g.has_edge(vertex_i, vertex_j, key=edge_type):
                self._g.remove_edge(vertex_i, vertex_j, key=edge_type)
            if self._g.has_edge(vertex_j, vertex_i, key=edge_type):
                self._g.remove_edge(vertex_j, vertex_i, key=edge_type)

    def update_uncertain_edge(self, vertex_i, vertex_j, edge_type='*-o') -> None:
        assert edge_type in self._list_uncertain_edge_types
        self._uncertain_g.add_edge(vertex_i, vertex_j, type=edge_type)
        for edge_type in self._list_uncertain_edge_types:
            self._g.remove_edge(vertex_i, vertex_j, key=edge_type)
        self._g.add_edge(vertex_j, vertex_i, key=edge_type)

    def uncertain_to_certain_edge(self, vertex_i: str, vertex_j: str, edge_type="->") -> None:
        assert edge_type in self._list_certain_edge_types
        self.remove_uncertain_edge(vertex_i, vertex_j)
        if edge_type == "->":
            self.add_directed_edge(vertex_i, vertex_j)
        elif edge_type == "<->":
            self.add_confounded_edge(vertex_i, vertex_j)
        self._g.add_edge(vertex_j, vertex_i, key=edge_type)

    def get_edges(self):
        return set(self._g.edges(keys=True))

    def get_directed_edges(self):
        return set(self._directed_g.edges)

    def get_confounded_edges(self):
        return set(self._confounded_g.edges)

    # def get_undirected_edges(self):
    #     return set(self._undirected_g.edges)
    
    def get_undirected_edges(self): # Corrected to have the symmetry
        return set(e for edge in self._undirected_g.edges() for e in [edge, edge[::-1]])
    
    def get_uncertain_edges(self):
        return set(self._uncertain_g.edges)

    def get_edge_types(self, vertex_i: str, vertex_j: str):
        return set(list(self._g.get_edge_data(vertex_i, vertex_j).keys()))

    def is_acyclic(self):
        return nx.is_directed_acyclic_graph(self._directed_g)

    def get_parents(self, vertex: str) -> set:
        """
        :param vertex:
        :return:
        """
        return set(self._directed_g.predecessors(vertex))

    def get_children(self, vertex: str) -> set:
        return set(self._directed_g.successors(vertex))

    # def get_adjacencies(self, vertex: str) -> set:
    #     parents = self.get_parents(vertex)
    #     children = self.get_children(vertex)
    #     adjacencies = parents.union(children)
    #     return adjacencies
    
    def get_adjacencies(self, vertex: str) -> set:
        adjacents = set()
        if vertex in self._directed_g:
            adjacents.update(self._directed_g.predecessors(vertex))
            adjacents.update(self._directed_g.successors(vertex))
        if vertex in self._confounded_g:
            adjacents.update(self._confounded_g.neighbors(vertex))
        if vertex in self._undirected_g:
            adjacents.update(self._undirected_g.neighbors(vertex))
        if vertex in self._uncertain_g:
            adjacents.update(self._uncertain_g.predecessors(vertex))
            adjacents.update(self._uncertain_g.successors(vertex))
        return adjacents

    # def get_unshielded_triples(self):
    #     """
    #     Returns all unshielded triples in the graph.
    #     An unshielded triple is a triple (X, Z, Y) where:
    #         - X and Z are adjacent
    #         - Y and Z are adjacent
    #         - X and Y are NOT adjacent
    #     """
    #     triples = []
    #     adj = {v: self.get_adjacencies(v) for v in self.get_vertices()}

    #     for z in self.get_vertices():
    #         neighbors = list(adj[z])
    #         if len(neighbors) < 2:
    #             continue
    #         # on parcourt toutes les paires de voisins
    #         for i in range(len(neighbors)):
    #             x = neighbors[i]
    #             for j in range(i + 1, len(neighbors)):
    #                 y = neighbors[j]
    #                 if y not in adj[x]: 
    #                     triples.append((x, z, y))
    #     return triples


    def get_ancestors(self, vertex: str) -> set:
        """
        sds
        :param vertex:
        :return:
        """
        def ancestor_recursive(vertex_i: str, sublist: list):
            sublist.append(vertex_i)
            if self.get_parents(vertex_i):
                for parent in self.get_parents(vertex_i):
                    if parent not in sublist:
                        return sublist + ancestor_recursive(parent, sublist)
                    else:
                        return sublist
            else:
                return sublist
        return set(ancestor_recursive(vertex, []))

    def get_descendants(self, vertex: str) -> set:
        """
        sds
        :param vertex:
        :return:
        """
        def descendant_recursive(vertex_i: str, sublist: list):
            sublist.append(vertex_i)
            if self.get_children(vertex_i):
                for child in self.get_children(vertex_i):
                    if child not in sublist:
                        return sublist + descendant_recursive(child, sublist)
                    else:
                        return sublist
            else:
                return sublist
        return set(descendant_recursive(vertex, []))

    def get_non_descendants(self, vertex: str) -> set:
        return self.get_vertices().difference(self.get_descendants(vertex))

    def get_confounded_adjacencies(self, vertex: str) -> list:
        """
        sdsds
        :param vertex:
        :return:
        """
        return set(self._confounded_g.adj)
        # confounded_adjacencies = []
        # for potential_adj in self._g.pred[vertex]:
        #     for idx in self._g.pred[vertex][potential_adj]:
        #         if self._g.pred[vertex][potential_adj][idx]["type"] == '<->':
        #             if potential_adj not in confounded_adjacencies:
        #                 confounded_adjacencies.append(potential_adj)
        # The symmetry of the confounded adjacencies is ensured by adding edges from both vertex_i to vertex_j and
        # from vertex_j to vertex_i in the function add_bidirectional_edge.
        # for potential_adj in self._g.succ[vertex]:
        #     for idx in self._g.succ[vertex][potential_adj]:
        #         if self._g.succ[vertex][potential_adj][idx]["type"] == '<->':
        #             if potential_adj not in confounded_adjacencies:
        #                 confounded_adjacencies.append(potential_adj)
        # return confounded_adjacencies

    def all_paths(self, vertex_i: str, vertex_j: str):
        1

    def all_confounded_paths(self, vertex_i: str, vertex_j: str):
        1

    def counfounded_components(self):
        1

    def draw_graph(self, treatment: set = None, outcome: set = None):
        vertex_color = "lightblue"
        font_color = "black"
        directed_edge_color = "gray"
        confounded_edge_color = "black"
        undirected_edge_color = "black"
        uncertain_edge_color = "#F7B617"
        treatment_color = "#c82804"
        outcome_color = "#4851a1"

        all_nodes = list(self._g.nodes)
        outcome = set(outcome or [])
        treatment = set(treatment or [])
        non_outcome_nodes = [n for n in all_nodes if n not in outcome]

        # Layout
        circular_pos = nx.circular_layout(self._g.subgraph(non_outcome_nodes))
        center = np.array([0.0, 0.0])
        pos = {n: center for n in outcome}
        pos.update(circular_pos)

        fig, ax = plt.subplots()

        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(treatment), node_color=treatment_color)
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(outcome), node_color=outcome_color)

        set_vertices = set(self._g.nodes) - treatment - outcome
        nx.draw_networkx_nodes(self._g, pos, ax=ax, nodelist=list(set_vertices), node_color=vertex_color)
        nx.draw_networkx_labels(self._g, pos, ax=ax, font_color=font_color)

        acyclic_edges = [edge for edge in self.get_directed_edges() if (edge[1], edge[0]) not in self.get_directed_edges()]
        cyclic_edges = [edge for edge in self.get_directed_edges() if (edge[1], edge[0]) in self.get_directed_edges()]
        confounded_edges = self.get_confounded_edges()
        undirected_edges = self.get_undirected_edges()

        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=acyclic_edges, edge_color=directed_edge_color, arrowstyle='->')
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=cyclic_edges, edge_color=directed_edge_color, arrowstyle='->')
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=confounded_edges, arrowstyle='<|-|>', edge_color=confounded_edge_color)
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=undirected_edges, arrowstyle='-', edge_color=undirected_edge_color)

        dashed_arrow = [(u, v) for (u, v, t) in self.get_edges() if t == '-->']
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=dashed_arrow, edge_color=uncertain_edge_color,
                            style='dashed', arrowstyle='->')

        arrow_double_bar = [(u, v) for (u, v, t) in self.get_edges() if t == '-||']
        nx.draw_networkx_edges(self._g, pos, ax=ax, edgelist=arrow_double_bar, edge_color="red",
                            style='solid', arrowstyle='-[')

        plt.axis('off')
    plt.show()



    



    
    



class FullySpecifiedGraph(Graph):
    def __init__(self):
        super(Graph, self).__init__()
        self.add_directed_edge = self.add_directed_edge if self._remain_acyclic() else 1


class DirectedMixedGraph(Graph):
    def add_undirected_edge(self, vertex_i: str, vertex_j: str) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)

    def remove_undirected_edge(self, vertex_i: str, vertex_j: str) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)


class AcyclicDirectedMixedGraph(DirectedMixedGraph):
    1


class DirectedAcyclicGraph(AcyclicDirectedMixedGraph):
    def __init__(self):
        super(AcyclicDirectedMixedGraph, self).__init__()

    def add_confounded_edge(self, vertex_i: str, vertex_j: str) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)

    def remove_confounded_edge(self, vertex_i: str, vertex_j: str) -> None:
        raise NotImplementedError("This function is not available for " + self.__class__.__name__)





