import random

import numpy as np

from heuristic.l1_destroy_operator import WorstDestinationsRemoval
from heuristic.reinsertion_operator import HighestRegretBestPosition
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
    d_op = WorstDestinationsRemoval(2,5)
    reinsertion_op = HighestRegretBestPosition(problem)
    best_total_cost = solution.total_cost
    for i in range(100):
        # print(f"iteration: {i}")
        modified_solution: Solution = solution.copy()
        d_op.apply_with_check(problem, modified_solution)
        reinsertion_op.apply_with_check(problem, modified_solution)
        if modified_solution.total_cost < best_total_cost:
            best_total_cost = modified_solution.total_cost
            solution = modified_solution
        elif random.random()<0.05:
            solution = modified_solution
        # print(solution.total_cost)
        # print(modified_solution.total_cost)
    print(best_total_cost)
        # print("----------------")
    # visualize_solution(problem, solution)
if __name__ == "__main__":
    # fixed random seed
    # seed = 1123
    # random.seed(seed)
    # np.random.seed(seed)
    for i in range(100):
        print("test",i)
        main()