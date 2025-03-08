from abc import ABC, abstractmethod
from random import randint

import numpy as np

from heuristic.operator import Operator
from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

class RandomRouteSegmentRemoval(Operator):
    # Level 1 Operator
    def __init__(self, min_to_remove, max_to_remove):
        super().__init__()
        self.min_to_remove = min_to_remove
        self.max_to_remove = max_to_remove
    
    def apply(self, problem, solution):
        route_lengths = np.asanyarray([len(route) for route in solution.routes], int)
        chosen_vehicle_idx = np.random.choice(np.where(route_lengths>1)[0])
        chosen_route_length = route_lengths[chosen_vehicle_idx]
        min_to_remove = min(self.min_to_remove, chosen_route_length-1)
        max_to_remove = min(self.max_to_remove, chosen_route_length-1)
        segment_len = randint(min_to_remove, max_to_remove)
        start_idx = randint(1, chosen_route_length-segment_len)
        end_idx = start_idx + segment_len
        remove_segment(solution, chosen_vehicle_idx, start_idx, end_idx)

def remove_segment(solution: Solution,
                    vehicle_idx: int,
                    start_idx: int,
                    end_idx: int):
    problem = solution.problem
    dests_to_remove = solution.routes[vehicle_idx][start_idx:end_idx]
    # adjust vehicle charge
    start_dest_idx, end_dest_idx = dests_to_remove[0], dests_to_remove[-1]
    prev_dest_idx = solution.routes[vehicle_idx][start_idx-1]
    next_dest_idx = solution.routes[vehicle_idx][end_idx%len(solution.routes[vehicle_idx])]
    vehicle_costs = problem.distance_matrix[[prev_dest_idx, end_dest_idx, prev_dest_idx],[start_dest_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[vehicle_idx]
    prev_to_start_cost, end_to_next_cost, prev_to_next_cost = vehicle_costs
    solution.total_vehicle_charge = solution.total_vehicle_charge - (prev_to_start_cost + end_to_next_cost) + prev_to_next_cost
    # remove from route
    solution.destination_vehicle_assignmests[dests_to_remove] = -1
    # solution.routes[vehicle_idx] = solution.routes[vehicle_idx][:start_idx] + solution.routes[vehicle_idx][end_idx:]
    solution.routes[vehicle_idx] = np.concatenate([solution.routes[vehicle_idx][:start_idx], solution.routes[vehicle_idx][end_idx:]])
    print(solution.routes[vehicle_idx])
    print(type(solution.routes[vehicle_idx]))
    exit()