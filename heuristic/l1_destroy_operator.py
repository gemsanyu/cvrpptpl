from random import randint

import numpy as np

from heuristic.operator import Operator
from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

class RandomDestinationsRemoval(Operator):
    def __init__(self, min_to_remove, max_to_remove):
        super().__init__()
        self.min_to_remove = min_to_remove
        self.max_to_remove = max_to_remove
    
    def apply(self, problem, solution):
        dests_in_routes = np.where(solution.destination_vehicle_assignmests > -1)[0]
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(dests_in_routes))
        # pick randomly
        dests_to_remove = np.random.choice(dests_in_routes, num_to_remove, replace=False)
        for dest_idx in dests_to_remove:
            remove_a_destination(solution, dest_idx)
             
class WorstDestinationsRemoval(Operator):
    def __init__(self, min_to_remove, max_to_remove):
        super().__init__()
        self.min_to_remove = min_to_remove
        self.max_to_remove = max_to_remove
    
    def apply(self, problem, solution):
        dests_in_routes = np.where(solution.destination_vehicle_assignmests > -1)[0]
        removal_d_costs = compute_removal_d_costs(problem, solution, dests_in_routes)
        sorted_idx = np.argsort(removal_d_costs)
        dests_in_routes = dests_in_routes[sorted_idx]

        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(dests_in_routes))
        dests_in_routes = dests_in_routes[:num_to_remove]
        # pick randomly
        dests_to_remove = np.random.choice(dests_in_routes, num_to_remove, replace=False)
        for dest_idx in dests_to_remove:
             remove_a_destination(solution, dest_idx)

def compute_removal_d_costs(problem:Cvrpptpl, 
                           solution: Solution, 
                           dests_in_routes: np.ndarray):
    removal_d_costs = np.zeros_like(dests_in_routes, dtype=float)
    for i, dest_idx in enumerate(dests_in_routes):
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        pos = solution.routes[v_idx].index(dest_idx)
        prev_dest_idx = solution.routes[v_idx][pos-1]
        next_dest_idx = solution.routes[v_idx][(pos+1)%len(solution.routes[v_idx])]
        related_arc_costs = problem.distance_matrix[[prev_dest_idx, dest_idx, prev_dest_idx],[dest_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[v_idx]
        prev_to_dest_cost, pos_to_dest_cost, prev_to_next_cost = related_arc_costs
        d_cost = prev_to_next_cost -(prev_to_dest_cost + pos_to_dest_cost)
        removal_d_costs[i] = d_cost
    return removal_d_costs
    
        
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

def remove_a_destination(solution: Solution,
                         dest_idx: int):
    problem = solution.problem
    v_idx = solution.destination_vehicle_assignmests[dest_idx]
    pos = solution.routes[v_idx].index(dest_idx)
    prev_dest_idx = solution.routes[v_idx][pos-1]
    next_dest_idx = solution.routes[v_idx][(pos+1)%len(solution.routes[v_idx])]
    
    related_arc_costs = problem.distance_matrix[[prev_dest_idx, dest_idx, prev_dest_idx],[dest_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[v_idx]
    prev_to_dest_cost, pos_to_dest_cost, prev_to_next_cost = related_arc_costs
    d_cost = prev_to_next_cost -(prev_to_dest_cost + pos_to_dest_cost)
    solution.total_vehicle_charge = solution.total_vehicle_charge + d_cost
    # remove from route
    solution.destination_vehicle_assignmests[dest_idx] = -1
    solution.routes[v_idx] = solution.routes[v_idx][:pos] + solution.routes[v_idx][pos+1:]
    solution.vehicle_loads[v_idx] -= solution.destination_total_demands[dest_idx]

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
    solution.routes[vehicle_idx] = solution.routes[vehicle_idx][:start_idx] + solution.routes[vehicle_idx][end_idx:]
    solution.vehicle_loads[vehicle_idx] -= np.sum(solution.destination_total_demands[dests_to_remove])