class BackgroundKnowledge:
    """
    Stores background knowledge for causal discovery algorithms.

    Supports:
    - Mandatory / forbidden undirected edges
    - Mandatory / forbidden edge orientations
    - Non-descendant constraints

    Non-descendant constraints are stored as a dictionary:
        {D: set_of_nodes}

    where each node in set_of_nodes is constrained to NOT be a descendant of D.
    """

    def __init__(self):
        # Edges are represented as tuples (u, v)
        self._mandatory_edges = set()
        self._forbidden_edges = set()

        # Orientations are tuples (u, v) meaning u -> v
        self._mandatory_orientations = set()
        self._forbidden_orientations = set()

        # Non-descendant constraints: {node: set(non_descendants)}
        self._non_descendants = dict()

    # -------------------- Mandatory edges --------------------
    def add_mandatory_edge(self, u, v):
        """Add a single mandatory edge u-v (undirected)."""
        self._mandatory_edges.add((u, v))

    def add_mandatory_edges_from(self, edges):
        """Add multiple mandatory edges from [(u,v), ...]."""
        self._mandatory_edges.update(edges)

    def remove_mandatory_edge(self, u, v):
        """Remove a mandatory edge if it exists."""
        self._mandatory_edges.discard((u, v))

    def get_mandatory_edges(self):
        """Return all mandatory edges."""
        return self._mandatory_edges.copy()

    # -------------------- Forbidden edges --------------------
    def add_forbidden_edge(self, u, v):
        """Add a forbidden undirected edge u-v."""
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

    # -------------------- Non-descendant constraints --------------------
    def add_non_descendant(self, node, non_descendant):
        """
        Add a non-descendant constraint.

        Parameters
        ----------
        node : hashable
            The reference node D.
        non_descendant : hashable
            A node that cannot be a descendant of D.
        """
        if node not in self._non_descendants:
            self._non_descendants[node] = set()
        self._non_descendants[node].add(non_descendant)

    def add_non_descendants_from(self, node, nodes):
        """
        Add multiple non-descendant constraints for a given node.

        Parameters
        ----------
        node : hashable
            The reference node D.
        nodes : iterable
            Nodes that cannot be descendants of D.
        """
        if node not in self._non_descendants:
            self._non_descendants[node] = set()
        self._non_descendants[node].update(nodes)

    def remove_non_descendant(self, node, non_descendant):
        """Remove a non-descendant constraint if it exists."""
        if node in self._non_descendants:
            self._non_descendants[node].discard(non_descendant)
            if not self._non_descendants[node]:
                del self._non_descendants[node]

    def get_non_descendants(self):
        """
        Return the non-descendant constraints as a dictionary:
        {node: set(non_descendants)}.
        """
        return {k: v.copy() for k, v in self._non_descendants.items()}