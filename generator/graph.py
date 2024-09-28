import numpy as np
import random
from string import ascii_lowercase, ascii_uppercase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import smilesreader

GLOB_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 1], [0.8, 0.8, 0.8], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]]

class TypedGraph:
    graph : nx.Graph = None

    randomgen : random.Random = None

    def __init__(self, n, p=0.5, t=1, k=1, seed = None, mode="erdos-renyi", type_mode = "default", graph_parts = 1, normal = False, n_range = 0, smiles_dir = None, use_distribution = False):
        self.randomgen = random.Random(seed)
        if normal:
            nodes = round(self.randomgen.normalvariate(n, n_range))
        else:
            nodes = self.randomgen.randint(round(n), round(n+n_range))
        rem = nodes % graph_parts
        nodes = nodes // graph_parts
        if mode == "erdos-renyi":
            graph = nx.Graph()
            for i in range(graph_parts):
                graph = nx.disjoint_union(graph, nx.erdos_renyi_graph(nodes + int(rem >= 1), p, seed=seed))
                rem -= 1
            self.graph = graph
        elif mode == "watts-strogatz":
            graph = nx.Graph()
            for i in range(graph_parts):
                graph = nx.disjoint_union(graph, nx.watts_strogatz_graph(nodes + int(rem >= 1), k, p, seed=seed))
                rem -= 1
            self.graph = graph
        elif mode == "barabasi-albert":
            graph = nx.Graph()
            for i in range(graph_parts):
                graph = nx.disjoint_union(graph, nx.barabasi_albert_graph(nodes + int(rem >= 1), k, seed=seed))
                rem -= 1
            self.graph = graph
        elif mode == "smiles":
            self.graph = smilesreader.create_smiles_graph(smiles_dir, n, use_distribution)

        else:
            print(f"Unknown generation mode \"{mode}\".")
            exit(1)

        types = []
        for i, node in enumerate(self.graph):
            if type_mode == "default":
                self.graph.nodes[i]['type'] = self.randomgen.randint(0, t - 1)

            elif type_mode == "degree":
                if mode == "smiles":
                    if self.graph.nodes[i]['element'] not in types:
                        types.append(self.graph.nodes[i]['element'])
                    self.graph.nodes[i]['type'] = types.index(self.graph.nodes[i]['element'])

                else:
                    self.graph.nodes[i]['type'] = self.graph.degree(i)
        for e in self.graph.edges:
            if 'order' not in self.graph.edges[e]:
                # If we are not using double.triple bonds, all edges are single weight
                self.graph.edges[e]['order'] = 1

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
                    # Keep going untill the amount of changed edges is higher than add, or 1000 tries.
                    while len(list(filter(lambda x: not x in self.graph_prec.edges, self.graph_post.edges))) < add and k < 1000:
                        k += 1
                        u1, v1, order1 = self.randomgen.choice(list(self.graph_post.edges.data('order')))
                        u2, v2, order2 = self.randomgen.choice([x for x in list(self.graph_post.edges.data('order')) if x[2] == order1])
                        t = self.randomgen.randint(0,1)
                        # Try random order so both permutations can occur.
                        if t:
                            if (u1 != u2 and v1 != v2 and not self.graph_post.has_edge(u1, u2) and not self.graph_post.has_edge(v1, v2)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, u2, order = order1)
                                self.graph_post.add_edge(v1, v2, order = order2)
                                j += 2
                            elif (u1 != v2 and u2 != v1 and not self.graph_post.has_edge(u1, v2) and not self.graph_post.has_edge(u2, v1)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, v2, order = order1)
                                self.graph_post.add_edge(u2, v1, order = order2)
                                j += 2
                        else:
                            if (u1 != v2 and u2 != v1 and not self.graph_post.has_edge(u1, v2) and not self.graph_post.has_edge(u2, v1)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, v2, order = order1)
                                self.graph_post.add_edge(u2, v1, order = order2)
                                j += 2
                            elif (u1 != u2 and v1 != v2 and not self.graph_post.has_edge(u1, u2) and not self.graph_post.has_edge(v1, v2)):
                                self.graph_post.remove_edge(u1, v1)
                                self.graph_post.remove_edge(u2, v2)
                                self.graph_post.add_edge(u1, u2, order = order1)
                                self.graph_post.add_edge(v1, v2, order = order2)
                                j += 2
                    return
                for i in range(remove):
                    try:
                        u, v = self.randomgen.choice(list(self.graph_post.edges))
                    except IndexError as e:
                        # There are already no edges in the graph anymore.
                        break

                    self.graph_post.remove_edge(u, v)
                for i in range(add):
                    e1 = self.randomgen.choice(list(self.graph_post.nodes))
                    e2 = self.randomgen.choice(list(self.graph_post.nodes))

                    # ensure that it doesn't try to add a self edge
                    i = 0
                    while e2 == e1:
                        e2 = self.randomgen.choice(list(self.graph_post.nodes))
                        i += 1
                        if i > 50:
                            break
                    if i > 50:
                        continue

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
            return None

        mapping = self.randomgen.choice(maps)

        # print(maps)
        transformed_graph = graph
        for edge in nx.Graph(self.graph_prec).edges:
            transformed_graph.remove_edge(mapping[edge[0]], mapping[edge[1]])
            transformed_graph.nodes[mapping[edge[0]]]["type"] = self.graph_post.nodes[edge[0]]["type"]
            transformed_graph.nodes[mapping[edge[1]]]["type"] = self.graph_post.nodes[edge[1]]["type"]
        for edge in self.graph_post.edges.data():
            transformed_graph.add_edge(mapping[edge[0]], mapping[edge[1]], order=edge[2]['order'])
            transformed_graph.nodes[mapping[edge[0]]]["type"] = self.graph_post.nodes[edge[0]]["type"]
            transformed_graph.nodes[mapping[edge[1]]]["type"] = self.graph_post.nodes[edge[1]]["type"]

        if view:
            colors = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0, 0, 0] for x in graph.nodes.data()]
            sub2 = plt.subplot(122)
            nx.draw(graph, with_labels=True, node_color=colors, node_size=100)
            plt.show()

        return mapping


    def view(self):
        colors_prec = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0.1, 0.1, 0.1] for x in self.graph_prec.nodes.data()]
        sub1 = plt.subplot(121)
        nx.draw(self.graph_prec, with_labels = True, node_color=colors_prec, node_size=100)
        colors_post = [GLOB_COLORS[x[1]["type"]] if x[1]["type"] < len(GLOB_COLORS) else [0.1, 0.1, 0.1] for x in self.graph_post.nodes.data()]
        sub2 = plt.subplot(122)
        nx.draw(self.graph_post, with_labels = True, node_color=colors_post, node_size=100)
        plt.show()
