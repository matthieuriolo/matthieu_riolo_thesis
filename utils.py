import config

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
    with open(config.FILE_EXECUTED_TESTS) as f:
        return name in [s.strip() for s in f.readlines()]

def structure_ran_successfully(structure):
    """
    adds the structure name to the list of successfully ran tests
    """
    with open(config.FILE_EXECUTED_TESTS, 'a') as f:
        f.write(structure_name(structure) + "\n")
