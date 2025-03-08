import random

import numpy as np

from heuristic.l1_destroy_operator import RandomRouteSegmentRemoval
from heuristic.reinsertion_operator import RandomOrderBestPosition
from heuristic.ext_utils import visualize_solution
from heuristic.random_initialization import random_initialization
from heuristic.solution import Solution
from problem.cvrpptpl import read_from_file

def main():
    cvrp_instance_name = "A-n32-k5"
    cvrpptpl_filename = f"{cvrp_instance_name}_idx_0.txt"
    problem = read_from_file(cvrpptpl_filename)
    solution = random_initialization(problem)
    # visualize_solution(problem, solution)
    d_op = RandomRouteSegmentRemoval(2,5)
    r_op = RandomOrderBestPosition()
    for i in range(100):
        modified_solution: Solution = solution.copy()
        d_op.apply(problem, modified_solution)
        # r_op.apply(problem, modified_solution)
        # if modified_solution.total_cost < solution.total_cost:
        #     solution = modified_solution
        # print(solution.total_cost)
if __name__ == "__main__":
    # fixed random seed
    random.seed(1)
    np.random.seed(1)
    main()