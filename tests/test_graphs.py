import pytest

from pyciphod.utils.graphs.graphs import Graph, AcyclicDirectedMixedGraph, DirectedAcyclicGraph, create_random_admg, create_random_dag


@pytest.fixture
def graph():
    """Fixture returning a fresh Graph instance."""
    return Graph()


def test_add_vertices_and_edges(graph):
    g = graph
    # ajouter sommets et différents types d'arêtes
    g.add_vertices(["A", "B", "X", "Y", "C", "D", "E", "F"])

    # directed X -> Y
    g.add_directed_edge("X", "Y")
    # undirected A - B
    g.add_undirected_edge("A", "B")
    # confounded C <-> D
    g.add_confounded_edge("C", "D")
    # uncertain E *-o F
    g.add_uncertain_edge("E", "F", edge_type='*-o')

    verts = g.get_vertices()
    assert set(["A", "B", "X", "Y", "C", "D", "E", "F"]).issubset(verts)

    # directed edges
    directed = g.get_directed_edges()
    assert ("X", "Y") in directed

    # undirected edges should contain both orientations
    undirected = g.get_undirected_edges()
    assert ("A", "B") in undirected and ("B", "A") in undirected

    # confounded: check either order present
    conf = g.get_confounded_edges()
    assert any(e == ("C", "D") or e == ("D", "C") for e in conf)

    # uncertain edges recorded in uncertain graph (directed representation)
    uncertain = g.get_uncertain_edges()
    assert ("E", "F") in uncertain

    # edge types for uncertain should include '*-o'
    edge_types = g.get_edge_types("E", "F")
    assert '*-o' in edge_types

    # parents/children
    assert "X" in g.get_parents("Y")
    assert "Y" in g.get_children("X")

    # adjacencies include neighbors from all edge types
    adj_A = g.get_adjacencies("A")
    assert "B" in adj_A
    adj_C = g.get_adjacencies("C")
    assert "D" in adj_C


def test_remove_edges_and_uncertain_update(graph):
    g = graph
    g.add_vertices(["U", "V"])
    g.add_undirected_edge("U", "V")
    assert ("U", "V") in g.get_undirected_edges()

    g.remove_undirected_edge("U", "V")
    assert ("U", "V") not in g.get_undirected_edges() and ("V", "U") not in g.get_undirected_edges()

    # uncertain update: add then remove
    g.add_uncertain_edge("P", "Q", edge_type='*-o')
    assert ("P", "Q") in g.get_uncertain_edges()
    g.remove_uncertain_edge("P", "Q")
    assert ("P", "Q") not in g.get_uncertain_edges()


def test_acyclicity_detection():
    g = Graph()
    g.add_vertices(["X", "Y", "Z"])
    g.add_directed_edge("X", "Y")
    g.add_directed_edge("Y", "Z")
    assert g.is_acyclic() is True

    # add back edge to create cycle
    g.add_directed_edge("Z", "X")
    assert g.is_acyclic() is False


def test_get_adjacencies_mixed_edges():
    g = Graph()
    g.add_vertices(["A", "B", "C", "D"])
    g.add_directed_edge("A", "B")
    g.add_undirected_edge("A", "C")
    g.add_confounded_edge("A", "D")

    adj = g.get_adjacencies("A")
    # adjacency should include B (child), C (undirected), D (confounded)
    assert {"B", "C", "D"}.issubset(adj)


def test_directed_acyclic_graph_prevents_cycles():
    dag = DirectedAcyclicGraph()
    dag.add_vertices(["A", "B", "C"])
    # add A->B and B->C should be fine
    dag.add_directed_edge("A", "B")
    dag.add_directed_edge("B", "C")
    assert dag.is_acyclic()

    # adding C->A should raise ValueError and not be present
    with pytest.raises(ValueError):
        dag.add_directed_edge("C", "A")
    assert ("C", "A") not in dag.get_directed_edges()


def test_acmg_allows_confounded_and_directed_edges_and_is_acyclic():
    admg = AcyclicDirectedMixedGraph()
    admg.add_vertices(["X", "Y", "Z"])
    admg.add_directed_edge("X", "Y")
    admg.add_confounded_edge("Y", "Z")
    # Directed edges should be acyclic
    assert admg.is_acyclic()
    # adjacencies should include confounded neighbor
    assert "Z" in admg.get_adjacencies("Y")


def test_create_random_admg_and_dag_have_vertices_and_no_directed_cycles():
    admg = create_random_admg(num_v=5, p_edge=0.8, seed=42)
    assert len(admg.get_vertices()) == 5
    assert admg.is_acyclic()

    dag = create_random_dag(num_v=6, p_edge=0.5, seed=7)
    assert len(dag.get_vertices()) == 6
    assert dag.is_acyclic()
