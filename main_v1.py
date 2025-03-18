import random
from typing import List

import numpy as np


from heuristic.alns import ALNS
from heuristic.operator import select_operator
from heuristic.random_initialization import random_initialization
from heuristic.solution import Solution
from heuristic.ext_utils import visualize_solution
from problem.cvrpptpl import read_from_file


def prepare_l1_destroy_operators():
    from heuristic.l1_destroy_operator import ShawDestinationsRemoval, WorstDestinationsRemoval, RandomDestinationsRemoval, RandomRouteSegmentRemoval
    op1 = ShawDestinationsRemoval(1,5)
    op2 = WorstDestinationsRemoval(1,5)
    op3 = RandomDestinationsRemoval(1,5)
    op4 = RandomRouteSegmentRemoval(3, 10)
    return [op1, op2, op3, op4]

def prepare_l2_destroy_operators():
    from heuristic.l2_destroy_operator import WorstLockersRemoval, RandomCustomersRemoval, RandomLockersRemoval, WorstCustomersRemoval
    op1 = WorstLockersRemoval(1,3)
    op2 = RandomCustomersRemoval(1,5)
    op3 = RandomLockersRemoval(1,3)
    op4 = WorstCustomersRemoval(1,5)
    return [op1, op2, op3, op4]

def prepare_reassignment_operators(problem):
    from heuristic.reassignment_operator import BestFirstFitReassignment, RandomFirstFitReassignment
    op1 = BestFirstFitReassignment(problem)
    op2 = RandomFirstFitReassignment(problem)
    return [op1, op2]

def prepare_reinsertion_operators(problem):
    from heuristic.reinsertion_operator import RandomOrderBestPosition, HighestRegretBestPosition
    op1 = RandomOrderBestPosition(problem)
    op2 = HighestRegretBestPosition(problem)
    return [op1, op2]
    
    
    
def main():
    cvrp_instance_name = "A-n32-k5"
    cvrpptpl_filename = f"{cvrp_instance_name}_idx_0.txt"
    problem = read_from_file(cvrpptpl_filename)
    l1_d_operators = prepare_l1_destroy_operators()
    l2_d_operators = prepare_l2_destroy_operators()
    ra_operators = prepare_reassignment_operators(problem)
    ri_operators = prepare_reinsertion_operators(problem)
    solver = ALNS(problem,
                  l1_d_operators,
                  l2_d_operators,
                  ra_operators,
                  ri_operators,
                  initial_temp=1000,
                  cooling_rate=0.95
                  )
    best_solution = solver.solve()
    visualize_solution(problem, best_solution)
    
    
if __name__ == "__main__":
    import pickle
    # fixed random seed
    # seed = 1123
    # random.seed(seed)
    # np.random.seed(seed)
    with open("rng_state.pkl", "rb") as f:
        python_rng_state, numpy_rng_state = pickle.load(f)
        random.setstate(python_rng_state)   
        np.random.set_state(numpy_rng_state)
    main()
    # for i in range(1000):
    #     numpy_rng_state = np.random.get_state()
    #     python_rng_state = random.getstate()
    #     with open("rng_state.pkl", "wb") as f:
    #         pickle.dump((python_rng_state, numpy_rng_state), f)
    #     print("test ",i)
    #     main()