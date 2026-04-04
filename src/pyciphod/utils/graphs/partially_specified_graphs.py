from pyciphod.utils.graphs.graphs import Graph, DirectedMixedGraph, AcyclicDirectedMixedGraph, FullySpecifiedGraph


class PartiallySpecifiedGraph(Graph):
    def __init__(self):
        super().__init__()


class ClusterAcyclicDirectedMixedGraph(PartiallySpecifiedGraph, AcyclicDirectedMixedGraph):
    def __init__(self):
        super().__init__()


class ClusterDirectedMixedGraph(PartiallySpecifiedGraph, DirectedMixedGraph):
    def __init__(self):
        super().__init__()


class SummaryCausalGraph(ClusterDirectedMixedGraph):
    def __init__(self):
        super().__init__()


class DifferenceGraph(PartiallySpecifiedGraph, DirectedMixedGraph):
    def __init__(self):
        super().__init__()



class PartiallyDirectedGraphs(Graph):
    def __init__(self):
        super().__init__()


class CompletedPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()
        
        
class TemporalPartiallyDirectedAcyclicGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()
        


class LocalEssentialGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()

class PartialAncestralGraphs(PartiallyDirectedGraphs):
    def __init__(self):
        1


class CompletedPartiallyDirectedAcyclicDifferenceGraph(PartiallyDirectedGraphs):
    def __init__(self):
        super().__init__()

