from pysmiles import read_smiles
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pysmiles
import typing
import linecache

# If you want to change the distribution of molecule sizes, edit this distribution. Make sure the distribution has the same lenght as the amount of files in your smiles folder.
distribution = [0.1] * 10

def create_smiles_graph(smiles_dir, nodes, seed=None, use_distribution = False):
    np.random.seed(seed)

    # We need to read the data from a smiles directory
    if not smiles_dir:
        raise Exception("Can't use the smiles generation type without a directory of smiles files")

    files = [smiles_dir + x for x in sorted(os.listdir(smiles_dir))]
    line_counts = [sum([1 for _ in open(file)]) for file in files]

    graph = nx.Graph()
    # Now we know where we can read the smiles lines, choose a molecule from a file untill the graph is big enough.
    n = 0
    while n < nodes:
        if use_distribution:
            file_path = np.random.choice(files, p=distribution)
        else:
            file_path = np.random.choice(files)

        n += files.index(file_path) + 1
        f = open(file_path, 'r')
        i = 0
        line_n = np.random.randint(0, line_counts[files.index(file_path)])
        for line in f:
            if i == line_n:
                molecule = line.split('\t')[0]
            i += 1
        f.close()

        mol_graph = pysmiles.read_smiles(molecule, explicit_hydrogen=True)
        graph = nx.disjoint_union(graph, mol_graph)

    return graph