import config
import utils


for structure in config.structure_enumerate(config.CONTAINER_STRUCTURES):
    # ignore already executed test cases
    if utils.structure_tested(structure):
        continue
    name = utils.structure_name(structure)

    print(f'start test case {name}')
    
    test_case = utils.build_test(structure)
    test_case.pre()
    test_case.run()
    test_case.post()
    
    print(f'end test case {name}')