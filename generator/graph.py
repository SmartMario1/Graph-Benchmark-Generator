import numpy as np
import random
from string import ascii_lowercase, ascii_uppercase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import typing

GLOB_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

class TypedGraph:
    graph : nx.Graph = None
    # todo: dynamic?


    def __init__(self, n, p, t=1, k=1, seed = None, mode="erdos-renyi"):
        self.randomgen = random.Random(seed)
        if mode == "erdos-renyi":
            self.graph = nx.erdos_renyi_graph(n, p, seed=seed)
        elif mode == "watts-strogatz":
            self.graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
        elif mode == "barabasi-albert":
            self.graph = nx.barabasi_albert_graph(n, p)
        elif mode == "internet":
            self.graph = nx.random_internet_as_graph(n, seed=seed)

        for i, node in enumerate(self.graph):
            self.graph.nodes[i]['type'] = self.randomgen.randint(0, t - 1)

    def view(self):
        colors = [GLOB_COLORS[x[1]["type"]] for x in self.graph.nodes.data()]
        nx.draw(self.graph, node_color=colors, node_size=100)
        plt.show()

class GraphTransformation:
    graph_prec : nx.Graph = None
    graph_opps = None

    def __init__(self, graph_prec):
        self.graph_prec = graph_prec
        self.graph_opps = lambda x : x

    def apply(self, graph: nx.Graph):
        gen = nx.isomorphism.GraphMatcher(graph, self.graph_prec, node_match=lambda x, y: x['type'] == y['type'])
        i = 0
        for x in gen.subgraph_monomorphisms_iter():
            if (i == 0):
                i += 1
                print(x)
                continue
            print(x)
            mapping = {r:k for k, r in x.items()}
            nx.add_cycle(graph, [mapping[i] for i in range(4)])
            return

    def view(self):
        colors = [GLOB_COLORS[x[1]["type"]] for x in self.graph_prec.nodes.data()]
        nx.draw(self.graph_prec, node_color=colors, node_size=100)
        plt.show()



tg = TypedGraph(10, 2, t=3, mode="barabasi-albert")
tg.view()
print(list(tg.graph.nodes)[1:5])
tr = GraphTransformation(nx.subgraph(tg.graph, [0,1,2,3]))
tr.view()
tr.apply(tg.graph)
tg.view()
