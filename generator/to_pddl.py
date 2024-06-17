import graph
import networkx as nx

def typed_graph_to_pddl_state(graph: nx.Graph, action = False, inv = False, post = False):
    out = ""
    q = ""
    t = "\t"
    if action:
        q += '?'
        t += '\t'

    # we assume every node in the graph has the "type" attribute
    for e in graph.edges:
        n0 = q + f"n{e[0]}t{graph.nodes.data()[e[0]]['type']}"
        n1 = q + f"n{e[1]}t{graph.nodes.data()[e[1]]['type']}"
        if not inv:
            out += t + f"(link {n0} {n1})\n"
            out += t + f"(link {n1} {n0})\n"
        else:
            out += t + f"(not (link {n0} {n1}))\n"
            out += t + f"(not (link {n1} {n0}))\n"

    # Actions need the precondition that arguments are not the same node.
    if action and not post:
        nodes = list(graph.nodes)
        for i, n0 in enumerate(nodes):
            typen0 = graph.nodes.data()[n0]['type']
            for n1 in nodes[i+1:]:
                typen1 = graph.nodes.data()[n1]['type']
                if typen0 == typen1:
                    out += t + f"(not (= ?n{n0}t{typen0} ?n{n1}t{typen1}))\n"
                pass

    return out

def typed_graph_to_pddl_objects(graph: nx.Graph):
    out = ""
    # print([t for t in graph.nodes.data()])
    typedict = {t[1]['type']:[] for t in graph.nodes.data()}
    for n in graph.nodes:
        type = graph.nodes.data()[n]['type']
        typedict[type].append(n)

    for k in typedict:
        out += "\t"
        for n in typedict[k]:
            out += f"n{n}t{k} "
        out += f"- type{k}\n"
    return out

def typed_graph_to_pddl_action_args(graph: nx.Graph):
    out = ""
    for n in sorted(graph.nodes):
        type = graph.nodes.data()[n]['type']
        out += f"?n{n}t{type} - type{type} "
    return out

def graph_transformation_to_pddl_action(grtr: graph.GraphTransformation, id: int):
    out = f"(:action transformation{id}\n"
    out += f"\t:parameters ({typed_graph_to_pddl_action_args(grtr.graph_prec)}\t)\n"
    out += f"\t:precondition (and\n{typed_graph_to_pddl_state(grtr.graph_prec, action=True)}\t)\n"
    out += f"\t:effect (and\n{typed_graph_to_pddl_state(grtr.graph_prec, action=True, inv=True, post=True)}{typed_graph_to_pddl_state(grtr.graph_post, action=True, post=True)}\t)\n"
    out += ")\n\n"
    return out
