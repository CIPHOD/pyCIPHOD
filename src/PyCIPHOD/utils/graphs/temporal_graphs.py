from graphs import DirectedAcyclicGraph
from graphs import AcyclicDirectedMixedGraph


class FTDirectedAcyclicGraph(DirectedAcyclicGraph):
    def __init__(self):
        super(DirectedAcyclicGraph, self).__init__()
        self.t_min = 0
        self.t_max = 0

    def add_directed_edges_from(self):
        raise NotImplementedError("This function is not available for FTCGs")

    def add_bidirectional_edges_from(self, edge_list):
        raise NotImplementedError("This function is not available for FTCGs")


class FTAcyclicDirectedMixedGraph(AcyclicDirectedMixedGraph):
    1

