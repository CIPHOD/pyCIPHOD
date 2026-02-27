from graphs import Graph
from graphs import DirectedMixedGraph
from graphs import AcyclicDirectedMixedGraph


class PartiallySpecifiedGraph(Graph):
    def __init__(self, fully_specified_graph: FullySpecifiedGraph):
        assert fully_specified_graph.is_acyclic()


class ClusterAcyclicDirectedMixedGraph(PartiallySpecifiedGraph, AcyclicDirectedMixedGraph):
    def __init__(self):
        1


class ClusterDirectedMixedGraph(PartiallySpecifiedGraph, DirectedMixedGraph):
    def __init__(self):
        1


class SummaryCausalGraph(ClusterDirectedMixedGraph):
    def __init__(self):
        1


class PartiallyDirectedGraphs(Graph):
    def __init__(self):
        1


class CompletedPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        1


class PartialAncestralGraphs(PartiallyDirectedGraphs):
    def __init__(self):
        1