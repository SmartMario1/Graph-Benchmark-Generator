import networkx as nx
import to_pddl
import graph

seed = 44

tg = graph.TypedGraph(10, 1, t=3, mode="barabasi-albert", seed=seed)
tg.view()
print(list(tg.graph.nodes)[1:5])
tr = graph.GraphTransformation(nx.Graph(nx.subgraph(tg.graph, [0,1,5,3,4])), add=4, remove=4, seed=seed)
tr.view()
tr.apply(tg.graph, view=True)
tr.apply(tg.graph, view=True)

print(to_pddl.graph_transformation_to_pddl_action(tr, 0))