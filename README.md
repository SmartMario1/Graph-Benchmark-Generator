# Graph Benchmark Generator
 This is a generator for Hard To Ground (HTG) PDDL graph problems. To use it call the python file problem_generator.py.

## Arguments
In this section all possible arguments are explained, sometimes in more detail than in the --help message.

  - -h, --help          Show a compressed variant of this section and exit.
  - -n NODES, --nodes NODES
                        Amount of nodes to generate in the starting graph. If used with --graph_parts, the graph gets generated so that there are \<NODES\> nodes in total, distributing them as equal as possible among the graph parts.
  - -nr NODE_RANGE, --node_range NODE_RANGE
                        Amount of variance in node amount. Range starts from --nodes. Actual plan length gets sampled at random.
  - -l LENGTH, --length LENGTH
                        Length of the upperbound plan generated for the graph problem. It is not guaranteed that there is no shorter plan, but it is guaranteed that the plan is at most \<LENGTH\> long.
  - -lr LENGTH_RANGE, --length-range LENGTH_RANGE
                        Amount of variance in plan length. Range starts from --length. Actual plan length gets sampled at random per problem. A uniform distribution is used.
  - --mode {barabasi-albert,erdos-renyi,watts-strogatz,internet}
                        The way to generate each graph. Currently the following are supported: - barabasi-albert (DEFAULT) - erdos-renyi - watts-strogatz. If multiple graph_parts are used, each part gets internally generated using this method.
  - --action-size SIZE    The amount of arguments to be generated for actions. A larger size means the precondition of actions becomes larger and thus stricter. It also means that there should be more valid permutations of nodes and thus that the problem is harder to ground.
  - --action-range ACTION_RANGE
                        Amount of variance in action size. Range starts from --action-size. Actual action size gets sampled at random using a uniform distribution. Each action generated pulls a seperate value from the range.
  - --action-add ADD_AMT  The amount of (random) edges an action adds to the given subgraph. When using degree mode this instead means the amount of edges that should be different after the action is applied.
  - --action-rm RM_AMT    The amount of (random) edges an action removes before adding. If no edges can be removed anymore continues to adding. If using type_mode degree, this value is ignored.
  - -t TYPES, --types TYPES
                        Amount of node types to add to the graphs. Default is 2. This value gets ignored when using type_mode degree.
  - -p P                  Chance (float) parameter used by some graph generation methods.
  - -k K                  Discrete parameter used by some graph generation methods.
  - -s SEED, --seed SEED  Random generation seed.
  - --name NAME           The name of the batch. Saves the domain to domain_\<name>.pddl and the problems to p-\<n>_\<name>.pddl
  - --num_problems PRB_AMT
                        How many problems to generate from a given initial graph. More problems means that the search will become more
                        difficult, as each problem gets its own actions generated.
  - --min_diff_actions MDA_AMT
                        The minimum amount of different actions needed per problem generated. This guarantees that at least \<min_diff_action> actions are used in the upperbound plan.
                        This does not guarantee there is no plan with less actions, or that there aren't more actions in the upperbound plan.
  - --action_sample_mode {random-sequential,start-sequential,random,random-islands,connected,connected-islands}
                        The way to sample the current graph to generate preconditions for the actions. This can have a big effect on the preconditions of actions depending on graph
                        generation method.
    - random-sequential: Start from a random starting node and add the next \<action-size> nodes in generation order
    - start-sequential: Take the first \<action-size> nodes in generation order.
    - random: Take \<action-size> random nodes.
    - random-islands: Take \<action-size> / \<num_island> nodes sequentially in generation order, then generate a new starting point and repeat \<num_island> times.
    - connected: Start from a random node and follow an edge and add that node \<action-size> - 1 times.
    We always try to follow the edge of the most recent node to a new node. If there are no nodes that we have not already added attached to the most recent node, continue back in order of recency. If there are no nodes that can be added, generate a new starting point.
    - connected-islands: Same as connected, except we generate a new starting point after \<action-size> / \<num_islands> nodes are added.
  - --num_islands ISL_AMT
                        The amount of seperate islands to generate when choosing an island action sample mode. Islands are of equal size (when possible).
  - --type_mode {default,degree}
                        Change the type mode between default (random type assigned from -t different types) and degree (a node is categorized by its degree). When using degree mode, the way actions are generated changes to a method that performs edge swaps until \<action-add> edges are different than when we started. This way the degree of nodes gets preserved between actions.
  - --graph_parts GRAPH_PARTS
                        How many seperate graph parts to generate and combine into one graph. More graph parts means more disconnected parts that are internally generated by the given
                        method.
  - --normal              Switch from a uniform distribution to a normal distribution for all ranges specified. Actual value becomes mean and range becomes standard deviation.
  - --plan                Generate the upperbound plan of every problem and store it in an associated plan file
  - --same_start          If this flag is enabled, each problem generated will start from the same starting graph.
  - --view                DEBUG show intermediate graphs in interactive matplotlib windows
  - --verbose             DEBUG print when a mapping is found or not
