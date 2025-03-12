import random

import numpy as np

from heuristic.reassignment_operator import RandomOrderBestReassignment
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
    visualize_solution(problem, solution)
    # print(solution.total_cost)
    # visualize_solution(problem, solution)
    # destroy_operators = [RandomRouteSegmentRemoval(1,3), RandomRouteSegmentRemoval(5,10)]
    # reinsertion_operators = [RandomOrderBestPosition(problem), HighestRegretBestPosition(problem)]
    d_op = WorstLockersRemoval(1,3)
    # d_op = WorstDestinationsRemoval(1,5)
    reassignment_op = RandomOrderBestReassignment(problem)
    reinsertion_op = HighestRegretBestPosition(problem)
    best_total_cost = solution.total_cost
    for i in range(10000):
        modified_solution: Solution = solution.copy()
        # d_op = random.choice(destroy_operators)
        # r_op = random.choice(reinsertion_operators)
        d_op.apply(problem, modified_solution)
        reassignment_op.apply(problem, modified_solution)
        visualize_solution(problem, modified_solution)
        exit()
        # r_op.apply(problem, modified_solution)
        # if modified_solution.total_cost < best_total_cost:
        #     best_total_cost = modified_solution.total_cost
        #     solution = modified_solution
        # elif random.random()<0.05:
        #     solution = modified_solution
        # print(solution.total_cost)
        # print(best_total_cost)
    visualize_solution(problem, solution)
if __name__ == "__main__":
    # fixed random seed
    seed = 1123
    random.seed(seed)
    np.random.seed(seed)
    main()