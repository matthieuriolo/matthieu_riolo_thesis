#!/usr/bin/env python3

import itertools
from itertools import permutations
import os

# file locations
FILE_EXECUTED_TESTS = 'executed_tests.txt'


# define the different test cases
K_ALGO = 'algorithm'
V_ALGO_BERT = 'bert'
V_ALGO_INCEPTION = 'inception'

K_PCI_PERC = 'pci'
K_CPU_PERC = 'cpu'
K_DATA_SIZE_PERC = 'data_size'


CONTAINER_STRUCTURES = dict([
    (K_ALGO, [V_ALGO_INCEPTION, V_ALGO_BERT]),
    (K_PCI_PERC, [33, 66, 100]),
    (K_CPU_PERC, [33, 66, 100]),
    (K_DATA_SIZE_PERC, [2, 5]),
])











def structure_enumerate(container_structures):
    """
    Creates a generator with all permutation of the dictionary's values.
    The keys of the dictionary will stay intact
    """
    container_values = list(container_structures.values())
    for product_values in itertools.product(*container_values):
        yield dict(zip(container_structures.keys(), product_values))


def structure_name(structure):
    """
    returns a human readable name for the test structure
    """
    return "-".join([str(x) for x in structure.values()])

def structure_tested(structure):
    """
    returns true if the structure has been used for a performance test
    """
    if not os.path.isfile(FILE_EXECUTED_TESTS):
        return False
    name = structure_name(structure)
    with open(FILE_EXECUTED_TESTS) as f:
        return name in [s.strip() for s in f.readlines()]

def structure_ran(structure):
    """
    adds the structure name to the list of successfully ran tests
    """
    with open(FILE_EXECUTED_TESTS, 'a') as f:
        f.write(structure_name(structure) + "\n")

for structure in structure_enumerate(CONTAINER_STRUCTURES):
    structure_ran(structure)
    print(structure_tested(structure))