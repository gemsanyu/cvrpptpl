import random

import numpy as np

from heuristic.reassignment_operator import BestCustomerBestFirstFitReassignment
from heuristic.l2_destroy_operator import WorstCustomersRemoval, WorstLockersRemoval
from heuristic.l1_destroy_operator import ShawDestinationsRemoval, RandomDestinationsRemoval, WorstDestinationsRemoval
from heuristic.reinsertion_operator import RandomOrderBestPosition, HighestRegretBestPosition
from heuristic.ext_utils import visualize_solution
from heuristic.random_initialization import random_initialization
from heuristic.solution import Solution
from problem.cvrpptpl import read_from_file

def main():
    cvrp_instance_name = "A-n32-k5"
    cvrpptpl_filename = f"{cvrp_instance_name}_idx_0.txt"
    problem = read_from_file(cvrpptpl_filename)
    solution = random_initialization(problem)
    solution.check_validity()
    d_op = WorstCustomersRemoval(1,3)
    # d_op = WorstDestinationsRemoval(2,5)
    reassignment_op = BestCustomerBestFirstFitReassignment(problem)
    reinsertion_op = HighestRegretBestPosition(problem)
    best_total_cost = solution.total_cost
    for i in range(1):
        # print(f"iteration: {i}")
        modified_solution: Solution = solution.copy()
        d_op.apply_with_check(problem, modified_solution)
        return
        reassignment_op.apply_with_check(problem, modified_solution)
        reinsertion_op.apply_with_check(problem, modified_solution)
        if modified_solution.total_cost < best_total_cost:
            best_total_cost = modified_solution.total_cost
            solution = modified_solution
        elif random.random()<0.05:
            solution = modified_solution
        print(solution.total_cost)
        print(modified_solution.total_cost)
        print(best_total_cost)
        print("----------------")
    visualize_solution(problem, solution)
if __name__ == "__main__":
    # fixed random seed
    # seed = 1123
    # random.seed(seed)
    # np.random.seed(seed)
    import pickle
    with open("rng_state.pkl", "rb") as f:
        python_rng_state, numpy_rng_state = pickle.load(f)
        random.setstate(python_rng_state)   
        np.random.set_state(numpy_rng_state)
    main()
    # for i in range(1):
        # numpy_rng_state = np.random.get_state()
        # python_rng_state = random.getstate()
        # with open("rng_state.pkl", "wb") as f:
        #     pickle.dump((python_rng_state, numpy_rng_state), f)
        # print("test ",i)
        # main()