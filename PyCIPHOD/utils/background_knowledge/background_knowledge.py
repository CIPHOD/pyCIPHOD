class BackgroundKnowledge:
    """
    Stores background knowledge for causal discovery algorithms.
    Supports mandatory/forbidden edges and orientations.
    """

    def __init__(self):
        # Edges are represented as tuples (u, v)
        self._mandatory_edges = set()
        self._forbidden_edges = set()
        # Orientations are also tuples (u, v) meaning u -> v
        self._mandatory_orientations = set()
        self._forbidden_orientations = set()

    # -------------------- Mandatory edges --------------------
    def add_mandatory_edge(self, u, v):
        """Add a single mandatory edge u-v (undirected)."""
        self._mandatory_edges.add((u, v))

    def add_mandatory_edges_from(self, edges):
        """Add multiple mandatory edges from a list of tuples [(u,v), ...]."""
        self._mandatory_edges.update(edges)

    def remove_mandatory_edge(self, u, v):
        """Remove a mandatory edge if it exists."""
        self._mandatory_edges.discard((u, v))

    def get_mandatory_edges(self):
        """Return all mandatory edges as a set of tuples."""
        return self._mandatory_edges.copy()

    # -------------------- Forbidden edges --------------------
    def add_forbidden_edge(self, u, v):
        self._forbidden_edges.add((u, v))

    def add_forbidden_edges_from(self, edges):
        self._forbidden_edges.update(edges)

    def remove_forbidden_edge(self, u, v):
        self._forbidden_edges.discard((u, v))

    def get_forbidden_edges(self):
        return self._forbidden_edges.copy()

    # -------------------- Mandatory orientations --------------------
    def add_mandatory_orientation(self, u, v):
        """Add a mandatory orientation u -> v."""
        self._mandatory_orientations.add((u, v))

    def add_mandatory_orientations_from(self, edges):
        self._mandatory_orientations.update(edges)

    def remove_mandatory_orientation(self, u, v):
        self._mandatory_orientations.discard((u, v))

    def get_mandatory_orientations(self):
        return self._mandatory_orientations.copy()

    # -------------------- Forbidden orientations --------------------
    def add_forbidden_orientation(self, u, v):
        """Add a forbidden orientation u -> v."""
        self._forbidden_orientations.add((u, v))

    def add_forbidden_orientations_from(self, edges):
        self._forbidden_orientations.update(edges)

    def remove_forbidden_orientation(self, u, v):
        self._forbidden_orientations.discard((u, v))

    def get_forbidden_orientations(self):
        return self._forbidden_orientations.copy()