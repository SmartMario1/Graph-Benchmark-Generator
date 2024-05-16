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

    randomgen : random.Random = None

    def __init__(self, n, p=0.5, t=1, k=1, seed = None, mode="erdos-renyi"):
        self.randomgen = random.Random(seed)
        if mode == "erdos-renyi":
            self.graph = nx.erdos_renyi_graph(n, p, seed=seed)
        elif mode == "watts-strogatz":
            self.graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
        elif mode == "barabasi-albert":
            self.graph = nx.barabasi_albert_graph(n, k, seed=seed)
        elif mode == "internet":
            self.graph = nx.random_internet_as_graph(n, seed=seed)
        else:
            print(f"Unknown generation mode \"{mode}\".")
            exit(1)

        for i, node in enumerate(self.graph):
            self.graph.nodes[i]['type'] = self.randomgen.randint(0, t - 1)

    def view(self):
        colors = [GLOB_COLORS[x[1]["type"]] for x in self.graph.nodes.data()]
        nx.draw(self.graph, node_color=colors, node_size=100)
        plt.show()

class GraphTransformation:
    graph_prec : nx.Graph = None
    graph_post : nx.Graph = None
    randomgen : random.Random = None

    # When making a graph transformation, we assume at least <REMOVE> edges and <ADD> nodes are present in the pattern.
    def __init__(self, graph_prec, graph_post = None, seed=None, mode = "random", add=2, remove=2):
        self.randomgen = random.Random(seed)
        self.graph_prec = graph_prec
        if (not graph_post):
            self.graph_post = nx.Graph(graph_prec)
            if (mode == "random"):
                for i in range(remove):
                    # print(self.graph_post.edges)
                    try:
                        u, v = self.randomgen.choice(list(self.graph_post.edges))
                    except IndexError as e:
                        # There are already no edges in the graph anymore.
                        break
                    # print("removing:", u, "-", v)
                    self.graph_post.remove_edge(u, v)
                # print(self.graph_post.edges)
                for i in range(add):
                    e1 = self.randomgen.choice(list(self.graph_post.nodes))
                    e2 = self.randomgen.choice(list(self.graph_post.nodes))
                    # ensure that it doesn't try to add a self edge
                    while e2 == e1:
                        e2 = self.randomgen.choice(list(self.graph_post.nodes))
                    # print(self.graph_post.nodes, e1, e2)
                    self.graph_post.add_edge(e1, e2)
                    # print(self.graph_post.edges)
                    pass

    def apply(self, graph: nx.Graph, view = False) -> bool:
        if view:
            colors = [GLOB_COLORS[x[1]["type"]] for x in graph.nodes.data()]
            sub1 = plt.subplot(121)
            nx.draw(graph, with_labels=True, node_color=colors, node_size=100)

        gen = nx.isomorphism.GraphMatcher(graph, self.graph_prec, node_match=lambda x, y: x['type'] == y['type'])

        maps = []
        for i, inv_mapping in enumerate(gen.subgraph_monomorphisms_iter()):
            maps.append({v:k for k,v in inv_mapping.items()})
            if i > 1000:
                break

        if not maps:
            print("no mappings found")
            plt.clf()
            return False

        mapping = self.randomgen.choice(maps)
        # print(maps)
        transformed_graph = graph
        for edge in nx.Graph(self.graph_prec).edges:
            transformed_graph.remove_edge(mapping[edge[0]], mapping[edge[1]])
        for edge in self.graph_post.edges:
            transformed_graph.add_edge(mapping[edge[0]], mapping[edge[1]])

        if view:
            sub2 = plt.subplot(122)
            nx.draw(graph, with_labels=True, node_color=colors, node_size=100)
            plt.show()

        return True


    def view(self):
        colors = [GLOB_COLORS[x[1]["type"]] for x in self.graph_prec.nodes.data()]
        sub1 = plt.subplot(121)
        nx.draw(self.graph_prec, with_labels = True, node_color=colors, node_size=100)
        sub2 = plt.subplot(122)
        nx.draw(self.graph_post, with_labels = True, node_color=colors, node_size=100)
        plt.show()
