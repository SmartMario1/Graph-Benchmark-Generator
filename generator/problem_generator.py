import networkx as nx
import to_pddl
import graph
import typing
import os
import sys
import argparse
import random

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
    out += "node - object\n\t"
    out += ")\n\n"

    out += "(:predicates\n\t(link ?n0 - node ?n1 - node)\n\t)\n\n"

    for i, a in enumerate(actions):
        out += to_pddl.graph_transformation_to_pddl_action(a, i)

    out += ")"
    return out

def generate_connected_sample(args, tg, islands = False):
    sample = []
    to_add = args.size - 1
    start = randomgen.randint(0, args.nodes)
    sample.append(start)

    len_isl = args.size // args.isl_amt
    rem = args.size % args.isl_amt
    isl_start = to_add + 1

    while (to_add > 0):
        tmp = 0
        if rem:
            tmp = 1

        # If this is true, it is time for a new island
        if (islands and to_add <= (isl_start - len_isl - tmp)):
            start = randomgen.randint(0, args.nodes)
            n = 0
            while start in sample:
                n += 1
                start = randomgen.randint(0, args.nodes)
                if n > 200:
                    # We probably have the whole graph as a precondition at this point.
                    return sample
            sample.append(start)
            isl_start = to_add
            to_add -= 1
            if rem:
                rem -= 1
            continue


        added = False
        for j in reversed(range(len(sample))):
            edges = list(tg.graph.edges(sample[j]))
            while edges:
                new = randomgen.choice(edges)
                if not new[1] in sample:
                    sample.append(new[1])
                    added = True
                    break
                edges.remove(new)
            if added:
                break
        else:
            # If this executes, there was no new node to connect anywhere in our list.
            # We are forced to generate a new starting point.
            start = randomgen.randint(0, args.nodes)
            n = 0
            while start in sample:
                n += 1
                start = randomgen.randint(0, args.nodes)
                if n > 200:
                    # We probably have the whole graph as a precondition at this point.
                    return sample
            sample.append(start)
        to_add -= 1
    return sample

def generate_sample(args, tg : graph.TypedGraph):
    sample = []
    if args.sample_mode == "random-sequential":
        start = randomgen.randint(0, args.nodes - args.size - 1)
        sample = list(range(start, start + args.size))
    if args.sample_mode == "start-sequential":
        sample = list(range(args.size))
    if args.sample_mode == "random":
        sample = randomgen.sample(list(range(args.nodes)), args.size)
    if args.sample_mode == "random-islands":
        len_isl = args.size // args.isl_amt
        rem = args.size % args.isl_amt
        print(len_isl, rem)
        for _i in range(args.isl_amt):
            tmp = 0
            if rem:
                tmp = 1
                rem -= 1

            start = randomgen.randint(0, args.nodes - args.size - 1)
            # Make sure the islands are not overlapping
            n = 0
            while (start + len_isl + tmp) in sample:
                n += 1
                start = randomgen.randint(0, args.nodes - args.size - 1)
                if (n > 100):
                    # Assume it is impossible to add another disconnected island
                    print("Islands overlapping, action size is too big for disconnected islands.")
                    break

            isl = list(range(start, start + len_isl + tmp))
            sample += isl

    if args.sample_mode == "connected":
        sample = generate_connected_sample(args, tg)
    if args.sample_mode == "connected-islands":
        sample = generate_connected_sample(args, tg, islands=True)

    # Remove duplicates before returning (may reduce action precondition size)
    return list(set(sample))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a generator for generic graph problems in PDDL. \
        The problems generated with this generator seek to emulate the structure of the Organic Synthesis benchmark.")

    parser.add_argument('-n', '--nodes', dest='nodes', type=int, required=True,
                        help = "Amount of nodes to generate in each graph")

    parser.add_argument('-l', '--length', dest='length', type=int, required=True,
                        help = "Length of the upperbound plan generated for the graph problem.")

    parser.add_argument('-lr', '--length-range', dest='length_range', type=int, default=0,
                        help = "Amount of variance in plan length. Range starts from -length. Actual plan length gets sampled at random.")

    parser.add_argument('--mode', dest='mode', type=str, default="barabasi-albert", choices=['barabasi-albert', 'erdos-renyi', 'watts-strogatz', 'internet'],
                        help = "The way to generate each graph. Currently the following are supported:\
                        \n- barabasi-albert (DEFAULT)\n- erdos-renyi\n- watts-strogatz\n- internet")

    parser.add_argument('--action-size', dest='size', type=int, default=4,
                        help="The amount of arguments to be generated for actions. A larger size means\
                        the precondition of actions becomes larger and thus stricter.")

    parser.add_argument('--action-add', dest='add_amt', type=int, default=2,
                        help="The amount of (random) edges an action adds to the given subgraph. When using degree mode this should be at least 2.")

    parser.add_argument('--action-rm', dest='rm_amt', type=int, default=2,
                        help="The amount of (random) edges an action removes. If no edges can be removed anymore continues to adding.")

    parser.add_argument('-t', '--types', dest='types', type=int, default=TYPES_DEFAULT,
                        help = f"Amount of node types to add to the graphs. Default is {TYPES_DEFAULT}")

    parser.add_argument('-p', dest='p', type=float, default=0.5,
                        help="Chance parameter used by some graph generation methods")

    parser.add_argument('-k', dest='k', type=int, default=1,
                        help="Discrete parameter used by some graph generation methods")

    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None,
                        help="Random generation seed")

    parser.add_argument('--name', dest='name', type=str, default=None,
                        help="The name of the batch. Saves the domain to domain_<name>.pddl and the problems to p-<n>_<name>.pddl")

    parser.add_argument('--num_problems', dest='prb_amt', type=int, default=1,
                        help="How many problems to generate from a given initial graph. \
                            More problems means more actions get generated for these problems, so the search will become more difficult.")

    parser.add_argument('--min_diff_actions', dest='mda_amt', type=int, default=1,
                        help="The minimum amount of different actions needed per problem generated.\
                            This guarantees that at least <min_diff_action> actions are used in the upperbound plan.\
                            This does not guarantee there is no plan with less actions, or that there aren't more actions in the upperbound plan.")

    parser.add_argument('--action_sample_mode', dest='sample_mode', choices=['random-sequential', 'start-sequential', 'random', 'random-islands', 'connected', 'connected-islands'], default='random-sequential',
                        help="The way to sample the current graph to generate preconditions for the actions. This can have a big effect on the preconditions of actions depending on graph generation method.")

    parser.add_argument('--num_islands', dest='isl_amt', type=int, default=2,
                        help="The amount of seperate islands to generate when choosing an island action sample mode. Islands are of equal size (when possible).")

    parser.add_argument('--view', action="store_true", help="DEBUG show intermediate graphs")
    parser.add_argument('--verbose', action="store_true", help="DEBUG print more")

    args = parser.parse_args()

    randomgen = random.Random(args.seed)
    tg = graph.TypedGraph(args.nodes, p=args.p, k=args.k, t=args.types, mode=args.mode, seed=args.seed)
    # Save the initial graph for adding it to the pddl problems later.
    init = nx.Graph(tg.graph)

    name = ""
    if args.name:
        name += f"_{args.name}"

    length = random.randint(args.length, args.length + args.length_range)
    actions = []
    for j in range(args.prb_amt):
        i = 0
        sample = generate_sample(args, tg)
        actions.append(graph.GraphTransformation(nx.Graph(nx.subgraph(tg.graph, sample)), add=args.add_amt, remove=args.rm_amt, seed=args.seed))
        if args.view:
            actions[-1].view()
        while(i < args.length):
            # This applies the action, if it returns false, it didn't apply and we need to create a new action.
            if (args.length - i < args.mda_amt or not actions[-1].apply(tg.graph, view=args.view, verbose=args.verbose)):
                sample = generate_sample(args, tg)
                actions.append(graph.GraphTransformation(nx.Graph(nx.subgraph(tg.graph, sample)), add=args.add_amt, remove=args.rm_amt, seed=args.seed))
                if args.view:
                    actions[-1].view()
                actions[-1].apply(tg.graph, view=args.view, verbose=args.verbose)
                # Decrease the amount of different actions we still need to make
                args.mda_amt -= 1
            i += 1
        file = open(f"p-{j + 1}" + name + ".pddl", 'w')
        file.write(generate_pddl_problem(init, tg.graph, f"{j + 1}", args.name))
        file.close()
        tg.graph = nx.Graph(init)

    file = open("domain" + name + ".pddl", 'w')
    file.write(generate_pddl_domain(actions, args.name, args.types))
    file.close()