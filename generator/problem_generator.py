import networkx as nx
import to_pddl
import graph
import typing
import os
import sys
import argparse
import random

TYPES_DEFAULT = 2

def generate_pddl_problem(graph_init, graph_post, name, domain, degree_mode = False):
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

def generate_pddl_domain(actions: typing.List[graph.GraphTransformation], name, amount_types, degree_mode = False):
    out = f"(define (domain graph-{name})\n\n"
    out += "(:requirements :strips :typing :equality :negative-preconditions)\n\n"
    out += "(:types\n\t"
    for i in range(amount_types):
        out += f"type{i} "
    out += "- node\n\t"
    out += "node - object\n\t"
    out += ")\n\n"

    out += "(:predicates\n\t(link ?n0 - node ?n1 - node)\n\t(double-link ?n0 - node ?n1 - node)\n\t(triple-link ?n0 - node ?n1 - node)\n\t(aromatic-link ?n0 - node ?n1 - node)\n\t)\n\n"

    for i, a in enumerate(actions):
        out += to_pddl.graph_transformation_to_pddl_action(a, i)

    out += ")"
    return out

def generate_connected_sample(args, tg : graph.TypedGraph, islands = False):
    sample = []
    nodes = len(tg.graph.nodes) - 1
    if args.normal:
        size = round(randomgen.normalvariate(args.size, args.action_range))
    else:
        size = randomgen.randint(round(args.size), round(args.size + args.action_range))
    if args.tp_mode == "degree" and size < 3:
        size = 3
    to_add = size - 1
    start = randomgen.randint(0, nodes)
    sample.append(start)

    len_isl = size // args.isl_amt
    rem = size % args.isl_amt
    isl_start = to_add + 1

    while (to_add > 0):
        tmp = 0
        if rem:
            tmp = 1

        # If this is true, it is time for a new island
        if (islands and to_add <= (isl_start - len_isl - tmp)):
            start = randomgen.randint(0, nodes)
            n = 0
            while start in sample:
                n += 1
                start = randomgen.randint(0, nodes)
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
            start = randomgen.randint(0, nodes)
            n = 0
            while start in sample:
                n += 1
                start = randomgen.randint(0, nodes)
                if n > 200:
                    # We probably have the whole graph as a precondition at this point.
                    return sample
            sample.append(start)
        to_add -= 1
    return sample

def generate_sample(args, tg : graph.TypedGraph):
    sample = []
    nodes = len(tg.graph.nodes) - 1
    if (args.normal):
        size = round(randomgen.normalvariate(args.size, args.action_range))
    else:
        size = randomgen.randint(round(args.size), round(args.size + args.action_range))
    if args.tp_mode == "degree" and size < 3:
        size = 3
    if args.sample_mode == "random-sequential":
        start = randomgen.randint(0, nodes)
        sample = list(range(start, start + size))
    if args.sample_mode == "start-sequential":
        sample = list(range(size))
    if args.sample_mode == "random":
        sample = randomgen.sample(list(range(nodes + 1)), size)
    if args.sample_mode == "random-islands":
        len_isl = size // args.isl_amt
        rem = size % args.isl_amt
        print(len_isl, rem)
        for _i in range(args.isl_amt):
            tmp = 0
            if rem:
                tmp = 1
                rem -= 1

            start = randomgen.randint(0, nodes - size - 1)
            # Make sure the islands are not overlapping
            n = 0
            while (start + len_isl + tmp) in sample:
                n += 1
                start = randomgen.randint(0, nodes - size - 1)
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

def generate_pddl_plan(plan, tg: graph.TypedGraph):
    out = ""
    i = 0
    for t, mapping in plan:
        i += 1
        out += f"{i}: (transformation{t} "
        for n in sorted(tg.graph.nodes):
            if n in mapping:
                out += f"n{mapping[n]}t{tg.graph.nodes.data()[n]['type']} "
        out += ")\n"
    out += f"; cost = {i}\n"
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a generator for generic graph problems in PDDL. \
        The problems generated with this generator seek to emulate the structure of the Organic Synthesis benchmark.")

    parser.add_argument('-n', '--nodes', dest='nodes', type=float, required=True,
                        help = "Amount of nodes to generate in each graph")

    parser.add_argument('-nr', '--node_range', dest="node_range", type=float, default = 0,
                        help = "Amount of variance in node amount. Range starts from --nodes. Actual plan length gets sampled at random.")

    parser.add_argument('-l', '--length', dest='length', type=float, required=True,
                        help = "Length of the upperbound plan generated for the graph problem.")

    parser.add_argument('-lr', '--length-range', dest='length_range', type=float, default=0,
                        help = "Amount of variance in plan length. Range starts from --length. Actual plan length gets sampled at random.")

    parser.add_argument('--mode', dest='mode', type=str, default="barabasi-albert", choices=['barabasi-albert', 'erdos-renyi', 'watts-strogatz', 'smiles'],
                        help = "The way to generate each graph. Currently the following are supported:\
                        \n- barabasi-albert (DEFAULT)\n- erdos-renyi\n- watts-strogatz\n- smiles")

    parser.add_argument('--smiles', dest="smiles", type=str, default=None, help="Directory where smiles files are stored if using smiles method")

    parser.add_argument('--max-degree', dest="max_d", type=int, default=None, help="Max degree of nodes when generating using the barabasi albert method.")

    parser.add_argument('--action-size', dest='size', type=float, default=4,
                        help="The amount of arguments to be generated for actions. A larger size means\
                        the precondition of actions becomes larger and thus stricter.")

    parser.add_argument('--action-range', dest='action_range', type=float, default=0,
                        help="Amount of variance in action size. Range starts from --action-size. Actual action size gets sampled at random.")

    parser.add_argument('--action-add', dest='add_amt', type=int, default=2,
                        help="The amount of (random) edges an action adds to the given subgraph. When using degree mode this instead means the amount of edges that should be different after the action is applied.")

    parser.add_argument('--action-rm', dest='rm_amt', type=int, default=2,
                        help="The amount of (random) edges an action removes. If no edges can be removed anymore continues to adding.")

    parser.add_argument('-t', '--types', dest='types', type=int, default=TYPES_DEFAULT,
                        help = f"Amount of node types to add to the graphs. Default is {TYPES_DEFAULT}")

    parser.add_argument('-p', dest='p', type=float, default=0,
                        help="Chance parameter used by some graph generation methods")

    parser.add_argument('-k', dest='k', type=int, default=1,
                        help="Discrete parameter used by some graph generation methods")

    parser.add_argument('-s', '--seed', dest="seed", type=int, default=None,
                        help="Random generation seed")

    parser.add_argument('--name', dest='name', type=str, default=None,
                        help="The name of the batch. Saves the domain to domain_<name>.pddl and the problems to p-<n>_<name>.pddl")

    parser.add_argument('--num_problems', dest='prb_amt', type=int, default=1,
                        help="How many problems to generate from a given initial graph. More problems means that the search will become more\
                        difficult, as each problem gets its own actions generated.")

    parser.add_argument('--min_diff_actions', dest='mda_amt', type=int, default=1,
                        help="The minimum amount of different actions needed per problem generated.\
                            This guarantees that at least <min_diff_action> actions are used in the upperbound plan.\
                            This does not guarantee there is no plan with less actions, or that there aren't more actions in the upperbound plan.")

    parser.add_argument('--action_sample_mode', dest='sample_mode', choices=['random-sequential', 'start-sequential', 'random', 'random-islands', 'connected', 'connected-islands'], default='random-sequential',
                        help="The way to sample the current graph to generate preconditions for the actions. This can have a big effect on the preconditions of actions depending on graph generation method.")

    parser.add_argument('--num_islands', dest='isl_amt', type=int, default=2,
                        help="The amount of seperate islands to generate when choosing an island action sample mode. Islands are of equal size (when possible).")

    parser.add_argument('--type_mode', dest='tp_mode', choices=['default', 'degree'], default='default',
                        help="Change the type mode between default (random type assigned from -t different types) and degree (a node is categorized by its degree)")

    parser.add_argument('--graph_parts', type=int, default=1, help="How many seperate graph parts to generate and combine into one graph.\
        More graph parts means more disconnected parts that are internally generated by the given method.")

    parser.add_argument('--normal', action='store_true', help="Switch from a uniform distribution to a normal distribution for the ranges.")

    parser.add_argument('--view', action="store_true", help="DEBUG show intermediate graphs")
    parser.add_argument('--verbose', action="store_true", help="DEBUG print more")

    parser.add_argument('--plan', action="store_true", help="Generate the upperbound plan and store it in an associated plan file")

    parser.add_argument('--same_start', action="store_true", help="If this flag is enabled, each problem generated will start from the same starting graph.")

    parser.add_argument('--dir_split', action="store_true", help="Whether or not to split every problem into a seperate dir with its own domain file or to output all problems and 1 domain file flatly.")

    parser.add_argument('--use_dist', action="store_true", help="When using smiles, if this is true, use the distribution defined in smilesreader.py to select molecule sizes.")

    args = parser.parse_args()

    randomgen = random.Random(args.seed)

    name = ""
    if args.name:
        name += f"_{args.name}"

    actions = []
    problems = []

    amount_types = args.types - 1
    t = -1

    if args.same_start:
        tg = graph.TypedGraph(args.nodes, p=args.p, k=args.k, t=args.types, mode=args.mode, seed=randomgen.randint(0, 65536), type_mode=args.tp_mode, graph_parts = args.graph_parts, normal=args.normal, n_range=args.node_range, smiles_dir=args.smiles, max_d=args.max_d, use_distribution=args.use_dist)
        init = nx.Graph(tg.graph)

    for j in range(args.prb_amt):
        # If same_start is true, we don't want to generate a new graph each time.
        if not args.same_start:
            tg = graph.TypedGraph(args.nodes, p=args.p, k=args.k, t=args.types, mode=args.mode, seed=randomgen.randint(0, 65536), type_mode=args.tp_mode, graph_parts = args.graph_parts, normal=args.normal, n_range=args.node_range, smiles_dir=args.smiles, max_d=args.max_d, use_distribution=args.use_dist)
            init = nx.Graph(tg.graph)

        plan = []
        if args.normal:
            length = round(randomgen.normalvariate(args.length, args.length_range))
        else:
            length = randomgen.randint(round(args.length), round(args.length + args.length_range))
        i = 0
        sample = generate_sample(args, tg)
        mda_amt = args.mda_amt
        if (not (length - i < mda_amt)):
            actions.append(graph.GraphTransformation(nx.Graph(nx.subgraph(tg.graph, sample)), add=args.add_amt, remove=args.rm_amt, seed=args.seed, type_mode=args.tp_mode, keep_degree=(args.tp_mode == "degree")))
            t += 1
            if args.view:
                actions[-1].view()
        while(i < length):
            if (args.tp_mode == "degree"):
                max_degree = max(list(tg.graph.degree()), key=lambda x: x[1])
                if max_degree[1] > amount_types:
                    amount_types = max_degree[1]
            # This applies the action, if it returns false, it didn't apply and we need to create a new action.
            if (not (length - i < mda_amt)):
                mapping = actions[-1].apply(tg.graph, view=args.view, verbose=args.verbose)
            if (length - i < mda_amt or not mapping):
                sample = generate_sample(args, tg)
                actions.append(graph.GraphTransformation(nx.Graph(nx.subgraph(tg.graph, sample)), add=args.add_amt, remove=args.rm_amt, seed=args.seed, type_mode=args.tp_mode, keep_degree=(args.tp_mode == "degree")))
                if args.view:
                    actions[-1].view()
                mapping = actions[-1].apply(tg.graph, view=args.view, verbose=args.verbose)
                # Decrease the amount of different actions we still need to make
                mda_amt -= 1
                t += 1
            plan.append((t, mapping))
            i += 1
        if (args.tp_mode == "degree"):
            max_degree = max(list(tg.graph.degree()), key= lambda x: x[1])
            if max_degree[1] > amount_types:
                amount_types = max_degree[1]
        if not args.dir_split:
            file = open(f"p-{j + 1}" + name + ".pddl", 'w')
            file.write(generate_pddl_problem(init, tg.graph, f"{j + 1}", args.name, degree_mode = (args.tp_mode == "degree")))
            file.close()
            if args.plan:
                file = open(f"plan-{j + 1}" + name + ".plan", 'w')
                file.write(generate_pddl_plan(plan, tg))
                file.close()
        else:
            problems.append([generate_pddl_problem(init, tg.graph, f"{j + 1}", args.name, degree_mode = (args.tp_mode == "degree")), generate_pddl_plan(plan, tg)])
        if args.same_start:
            tg.graph = nx.Graph(init)

    if not args.dir_split:
        # If we don't split by dir, just write the final domain in the current folder.
        file = open("domain" + name + ".pddl", 'w')
        file.write(generate_pddl_domain(actions, args.name, amount_types=amount_types + 1, degree_mode = (args.tp_mode == "degree")))
        file.close()
    else:
        domain = generate_pddl_domain(actions, args.name, amount_types=amount_types + 1, degree_mode = (args.tp_mode == "degree"))
        for i, problem in enumerate(problems):
            try:
                os.mkdir(f"./p-{i+1}{name}/")
            except FileExistsError:
                pass
            file = open(f"./p-{i+1}{name}/p-{i + 1}" + name + ".pddl", 'w')
            file.write(problem[0])
            file.close()
            if args.plan:
                file = open(f"./p-{i+1}{name}/plan-{i + 1}" + name + ".plan", 'w')
                file.write(problem[1])
                file.close()
            file = open(f"./p-{i+1}{name}/domain.pddl", 'w')
            file.write(domain)
            file.close()