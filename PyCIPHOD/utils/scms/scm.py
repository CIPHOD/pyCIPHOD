from graphs import AcyclicDirectedMixedGraph

class SCM:
    def __init__(self, v, u, f, u_dist):
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
        self._list_uncertain_edge_types = ['*-o', '*->', '*-']

    def generate_data(self):
        # TODO

    def induced_graph(self):
        #TODO

class LinearSCM(SCM):
    def __init__(self):
        super(SCM, self).__init__()
        self.add_directed_edge = self.add_directed_edge if self._remain_acyclic() else 1


    def random_coefficients(self, method="gaussian"):
        self.g

    def compute_total_effect_from_scm(self, X, Y, coefficients):
        """
        Compute the total causal effect of X on Y in a linear SCM.

        Parameters:
        - G: A directed graph (DiGraph) where edges have weights representing direct effects.
        - X: The source node.
        - Y: The target node.
        - coefficients

        Returns:
        - total_effect: The sum of all causal path effects from X to Y.
        """
        total_effect = 0

        # Find all directed paths from X to Y
        all_paths = list(nx.all_simple_paths(G, source=X, target=Y))

        # Compute the effect along each path
        for path in all_paths:
            path_effect = 1  # Start with multiplicative identity
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_effect *= coefficients[(u, v)]  # Multiply the edge weights along the path
            total_effect += path_effect  # Sum up contributions from all paths
        return total_effect

def create_random_scm():
    1

def create_random_linear_scm_from_dag(g):
    scm = LinearSCM()
    coefficients = {edge: np.random.uniform(0, 1) for edge in g.edges}  # Random weights
    