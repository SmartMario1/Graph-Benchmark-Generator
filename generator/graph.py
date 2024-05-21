import numpy as np
import random
from string import ascii_lowercase, ascii_uppercase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import typing

GLOB_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 1], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]

class TypedGraph:
    graph : nx.Graph = None

    randomgen : random.Random = None

    def __init__(self, n, p=0.5, t=1, k=1, seed = None, mode="erdos-renyi", type_mode = "default"):
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
            if type_mode == "default":
                self.graph.nodes[i]['type'] = self.randomgen.randint(0, t - 1)
            elif type_mode == "degree":
                self.graph.nodes[i]['type'] = self.graph.degree(i)

    def view(self):
        colors = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0, 0, 0] for x in self.graph.nodes.data()]
        nx.draw(self.graph, node_color=colors, node_size=100)
        plt.show()

class GraphTransformation:
    graph_prec : nx.Graph = None
    graph_post : nx.Graph = None
    randomgen : random.Random = None
    type_mode = "default"

    # When making a graph transformation, we assume at least <REMOVE> edges and <ADD> nodes are present in the pattern.
    def __init__(self, graph_prec, graph_post = None, seed=None, mode = "random", add=2, remove=2, type_mode = "default", keep_degree = False):
        self.randomgen = random.Random(seed)
        self.graph_prec = graph_prec
        self.type_mode = type_mode
        if (not graph_post):
            self.graph_post = nx.Graph(graph_prec)
            if (mode == "random"):
                if keep_degree:
                    j = 0
                    k = 0
                    while j < add and k < 1000:
                        k += 1
                        u1, v1 = self.randomgen.choice(list(self.graph_post.edges))
                        u2, v2 = self.randomgen.choice(list(self.graph_post.edges))
                        t = self.randomgen.randint(0,1)
                        # Try random order so both permutations can occur.
                        if t:
                            if (u1 != u2 and v1 != v2 and not self.graph_post.has_edge(u1, u2) and not self.graph_post.has_edge(v1, v2)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, u2)
                                self.graph_post.add_edge(v1, v2)
                                j += 2
                            elif (u1 != v2 and u2 != v1 and not self.graph_post.has_edge(u1, v2) and not self.graph_post.has_edge(u2, v1)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, v2)
                                self.graph_post.add_edge(u2, v1)
                                j += 2
                        else:
                            if (u1 != v2 and u2 != v1 and not self.graph_post.has_edge(u1, v2) and not self.graph_post.has_edge(u2, v1)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, v2)
                                self.graph_post.add_edge(u2, v1)
                                j += 2
                            elif (u1 != u2 and v1 != v2 and not self.graph_post.has_edge(u1, u2) and not self.graph_post.has_edge(v1, v2)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, u2)
                                self.graph_post.add_edge(v1, v2)
                                j += 2
                    return
                for i in range(remove):
                    try:
                        u, v = self.randomgen.choice(list(self.graph_post.edges))
                    except IndexError as e:
                        # There are already no edges in the graph anymore.
                        break

                    self.graph_post.remove_edge(u, v)
                    if self.type_mode == "degree":
                        self.graph_post.nodes[u]["type"] -= 1
                        self.graph_post.nodes[v]["type"] -= 1
                for i in range(add):
                    e1 = self.randomgen.choice(list(self.graph_post.nodes))
                    e2 = self.randomgen.choice(list(self.graph_post.nodes))
                    # ensure that it doesn't try to add a self edge
                    while e2 == e1:
                        e2 = self.randomgen.choice(list(self.graph_post.nodes))
                    if self.type_mode == "degree" and not (self.graph_post.has_edge(e1, e2) or self.graph_post.has_edge(e2, e1)):
                        self.graph_post.nodes[e1]["type"] += 1
                        self.graph_post.nodes[e2]["type"] += 1
                    self.graph_post.add_edge(e1, e2)

    def apply(self, graph: nx.Graph, view = False, verbose = False) -> bool:
        if view:
            colors = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0, 0, 0] for x in graph.nodes.data()]
            sub1 = plt.subplot(121)
            nx.draw(graph, with_labels=True, node_color=colors, node_size=100)

        gen = nx.isomorphism.GraphMatcher(graph, self.graph_prec, node_match=lambda x, y: x['type'] == y['type'])

        maps = []
        if verbose: print("begin_mapping")

        for i, inv_mapping in enumerate(gen.subgraph_monomorphisms_iter()):
            maps.append({v:k for k,v in inv_mapping.items()})
            if verbose: print(f"found {i} map")
            if i > 10:
                break

        if verbose: print("done mapping")

        if not maps:
            print("no mappings found")
            plt.clf()
            return False

        mapping = self.randomgen.choice(maps)
        # print(maps)
        transformed_graph = graph
        for edge in nx.Graph(self.graph_prec).edges:
            transformed_graph.remove_edge(mapping[edge[0]], mapping[edge[1]])
            transformed_graph.nodes[mapping[edge[0]]]["type"] = self.graph_post.nodes[edge[0]]["type"]
            transformed_graph.nodes[mapping[edge[1]]]["type"] = self.graph_post.nodes[edge[1]]["type"]
        for edge in self.graph_post.edges:
            transformed_graph.add_edge(mapping[edge[0]], mapping[edge[1]])
            transformed_graph.nodes[mapping[edge[0]]]["type"] = self.graph_post.nodes[edge[0]]["type"]
            transformed_graph.nodes[mapping[edge[1]]]["type"] = self.graph_post.nodes[edge[1]]["type"]

        if view:
            colors = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0, 0, 0] for x in graph.nodes.data()]
            sub2 = plt.subplot(122)
            nx.draw(graph, with_labels=True, node_color=colors, node_size=100)
            plt.show()

        return True


    def view(self):
        colors_prec = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0.1, 0.1, 0.1] for x in self.graph_prec.nodes.data()]
        sub1 = plt.subplot(121)
        nx.draw(self.graph_prec, with_labels = True, node_color=colors_prec, node_size=100)
        colors_post = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0.1, 0.1, 0.1] for x in self.graph_post.nodes.data()]
        sub2 = plt.subplot(122)
        nx.draw(self.graph_post, with_labels = True, node_color=colors_post, node_size=100)
        plt.show()
