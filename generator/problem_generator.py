import networkx as nx
import to_pddl
import graph
import typing
import os
import sys
import argparse

TYPES_DEFAULT = 2

def generate_pddl_problem(graph_init, graph_post, name, domain):
    out = f"(define (problem graph-{name}) (:domain graph-{domain})\n"
    out += "(:objects\n"
    out += to_pddl.typed_graph_to_pddl_objects(graph_init)
    out += ")\n"
    out += "(:init\n"
    out += to_pddl.typed_graph_to_pddl_state(graph_init)
    out += ")\n"
    out += "(:goal (and\n"
    out += to_pddl.typed_graph_to_pddl_state(graph_post)
    out += "))\n"
    out += ")"
    return out

def generate_pddl_domain(actions: typing.List[graph.GraphTransformation], name, amount_types):
    out = f"(define (domain graph-{name})\n\n"
    out += "(:requirements :strips :typing :equality :negative-preconditions)\n\n"
    out += "(:types\n\t"
    for i in range(amount_types):
        out += f"type{i} "
    out += "- node\n\t"
    out += ")\n\n"

    out += "(:predicates\n\t(link ?n0 - node ?n1 - node)\n\t)\n\n"

    for i, a in enumerate(actions):
        out += to_pddl.graph_transformation_to_pddl_action(a, i)

    out += ")"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a generator for generic graph problems in PDDL. \
        The problems generated with this generator seek to emulate the structure of the Organic Synthesis benchmark.")

    parser.add_argument('-n', '--nodes', dest='nodes', type=int, required=True,
                        help = "Amount of nodes to generate in each graph")

    parser.add_argument('-l', '--length', dest='length', type=int, required=True,
                        help = "Length of the upperbound plan generated for the graph problem.")

    parser.add_argument('-lr', '--length-range', dest='length-range', type=int, default=0,
                        help = "Amount of variance in plan length. Range starts from -length. Actual plan length gets sampled at random.")

    parser.add_argument('--mode', dest='mode', type=str, default="barabasi-albert",
                        help = "The way to generate each graph. Currently the following are supported:\
                        \n- barabasi-albert (DEFAULT)\n- erdos-renyi\n- watts-strogatz\n- internet")

    parser.add_argument('-t', '--types', dest='types', type=int, default=TYPES_DEFAULT,
                        help = f"Amount of node types to add to the graphs. Default is {TYPES_DEFAULT}")

    parser.add_argument('-p', dest='p', type=float, default=0.5,
                        help="Chance parameter used by some graph generation methods")

    parser.add_argument('-k', dest='k', type=int, default=1,
                        help="Discrete parameter used by some graph generation methods")

    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None,
                        help="Random generation seed")

    parser.add_argument('--view', action="store_true", help="DEBUG show intermediate graphs")

    args = parser.parse_args()

    tg = graph.TypedGraph(args.nodes, p=args.p, k=args.k, t=args.types, mode="barabasi-albert", seed=args.seed)
    init = nx.Graph(tg.graph)
    tr = graph.GraphTransformation(nx.Graph(nx.subgraph(tg.graph, list([1, 10, 15, 19]))), add=4, remove=4, seed=args.seed)
    if args.view:
        tr.view()
    tr.apply(tg.graph, view=args.view)
    tr.apply(tg.graph, view=args.view)

    file = open("domain.pddl", 'w')
    file.write(generate_pddl_domain([tr], "test", args.types))
    file.close()
    file = open("testproblem.pddl", 'w')
    file.write(generate_pddl_problem(init, tg.graph, "test1", "test"))
    file.close()