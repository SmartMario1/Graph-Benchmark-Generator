import numpy as np
import random
from string import ascii_lowercase, ascii_uppercase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

class TypedGraph:

    graph : nx.Graph = None
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    def __init__(self, v, p, t=1, seed = None, mode="erdos-renyi"):
        self.randomgen = random.Random(seed)
        if mode == "erdos-renyi":
            self.graph = nx.erdos_renyi_graph(v, p, seed=seed)

        for i, node in enumerate(self.graph):
            self.graph.nodes[i]['type'] = self.randomgen.randint(0, t - 1)

    def view(self):
        colors = [self.colors[x[1]["type"]] for x in self.graph.nodes.data()]
        nx.draw(self.graph, node_color=colors)

        plt.show()




tg = TypedGraph(100, 0.03, 2)
tg.view()
