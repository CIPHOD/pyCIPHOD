from utils.graphs.graphs import Graph
from utils.graphs.graphs import DirectedMixedGraph
from utils.graphs.graphs import AcyclicDirectedMixedGraph
from utils.graphs.graphs import FullySpecifiedGraph


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
        super().__init__()


class CompletedPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()
        
        
class TemporalPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()
        


class LocalEssentialGraph(Graph):
    def __init__(self):
        super().__init__()

class PartialAncestralGraphs(PartiallyDirectedGraphs):
    def __init__(self):
        1