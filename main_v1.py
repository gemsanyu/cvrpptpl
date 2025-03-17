import random
from typing import List

import numpy as np

from heuristic.operator import Operator
from heuristic.l1_destroy_operator import DestroyStatus
from heuristic.reinsertion_operator import HighestRegretBestPosition, ReinsertionStatus
from heuristic.ext_utils import visualize_solution
from heuristic.random_initialization import random_initialization
from heuristic.solution import Solution
from problem.cvrpptpl import read_from_file

def select_operator(operators: List[Operator], weights: np.ndarray):
    probs = weights/weights.sum()
    selected_operator = np.random.choice(operators, p=probs, size=1)[0]
    return selected_operator

def prepare_l1_destroy_operators()->List[Operator]:
    from heuristic.l1_destroy_operator import ShawDestinationsRemoval, WorstDestinationsRemoval, RandomDestinationsRemoval, RandomRouteSegmentRemoval
    op1 = ShawDestinationsRemoval(1,5)
    op2 = WorstDestinationsRemoval(1,5)
    op3 = RandomDestinationsRemoval(1,5)
    op4 = RandomRouteSegmentRemoval(3, 10)
    return [op1, op2, op3, op4]

def prepare_reinsertion_operators(problem):
    from heuristic.reinsertion_operator import RandomOrderBestPosition, HighestRegretBestPosition
    op1 = RandomOrderBestPosition(problem)
    op2 = HighestRegretBestPosition(problem)
    return [op1, op2]

def main():
    cvrp_instance_name = "A-n32-k5"
    cvrpptpl_filename = f"{cvrp_instance_name}_idx_0.txt"
    problem = read_from_file(cvrpptpl_filename)
    solution = random_initialization(problem)
    solution.check_validity()
    l1_destroy_operators = prepare_l1_destroy_operators()
    l1_d_op_scores = np.asanyarray([0.1]*len(l1_destroy_operators))
    reinsertion_operators = prepare_reinsertion_operators(problem)
    ri_op_scores = np.asanyarray([0.1]*len(reinsertion_operators))
    best_total_cost = solution.total_cost
    for i in range(1000):
        # print(f"iteration: {i}")
        modified_solution: Solution = solution.copy()
        d_op = select_operator(l1_destroy_operators, l1_d_op_scores)
        ri_op = select_operator(reinsertion_operators, ri_op_scores)
        d_status = d_op.apply_with_check(problem, modified_solution)
        if d_status is not DestroyStatus.SUCCESS:
            continue
        r_status = ri_op.apply_with_check(problem, modified_solution)
        if r_status is not ReinsertionStatus.SUCCESS:
            continue
        if modified_solution.total_cost < best_total_cost:
            best_total_cost = modified_solution.total_cost
            solution = modified_solution
        elif random.random()<0.005:
            solution = modified_solution
        # print(solution.total_cost)
        # print(modified_solution.total_cost)
    print(best_total_cost)
        # print("----------------")
    # visualize_solution(problem, solution)
if __name__ == "__main__":
    import pickle
    # fixed random seed
    # seed = 1123
    # random.seed(seed)
    # np.random.seed(seed)
    # with open("rng_state.pkl", "rb") as f:
    #     python_rng_state, numpy_rng_state = pickle.load(f)
    #     random.setstate(python_rng_state)   
    #     np.random.set_state(numpy_rng_state)
    # main()
    for i in range(1000):
        numpy_rng_state = np.random.get_state()
        python_rng_state = random.getstate()
        with open("rng_state.pkl", "wb") as f:
            pickle.dump((python_rng_state, numpy_rng_state), f)
        print("test ",i)
        main()