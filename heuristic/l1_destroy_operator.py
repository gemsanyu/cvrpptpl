from random import randint

import numpy as np

from heuristic.d_op_utils import remove_a_destination, remove_segment, compute_removal_d_costs
from heuristic.operator import Operator
from heuristic.solution import Solution, NO_DESTINATION, NO_VEHICLE
from problem.cvrpptpl import Cvrpptpl


class L1DestroyOperator(Operator):
    def __init__(self, min_to_remove, max_to_remove):
        super().__init__()
        self.min_to_remove = min_to_remove
        self.max_to_remove = max_to_remove

class ShawDestinationsRemoval(L1DestroyOperator):
    def apply(self, problem, solution):
        dests_in_routes = np.where(solution.destination_vehicle_assignmests != NO_VEHICLE)[0]
        seed_pos = np.random.choice(len(dests_in_routes), 1)
        seed_dest_idx = dests_in_routes[seed_pos]
        demand_diff = np.abs(solution.destination_total_demands[dests_in_routes] - solution.destination_total_demands[seed_dest_idx])
        dist_from_seed = problem.distance_matrix[seed_dest_idx, dests_in_routes]
        is_dest_locker = (dests_in_routes <= problem.num_customers)
        is_seed_locker = is_dest_locker[seed_pos]
        is_same_type = (is_dest_locker == is_seed_locker).astype(float)
        type_factor = (1-is_same_type)*1.5
        vec_assignment_idx = solution.destination_vehicle_assignmests[dests_in_routes]
        seed_vec_idx = vec_assignment_idx[seed_pos]
        is_same_route = (vec_assignment_idx==seed_vec_idx)
        route_factor = (1-is_same_route)*0.5
        demand_diff = (demand_diff-demand_diff.mean())/demand_diff.std()
        dist_from_seed = (dist_from_seed-dist_from_seed.mean())/dist_from_seed.std()
        similarity = demand_diff + dist_from_seed + type_factor + route_factor
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(dests_in_routes))    
        sorted_idx = np.argsort(similarity)
        dests_to_remove = dests_in_routes[sorted_idx][:num_to_remove]
        for dest_idx in dests_to_remove:
            remove_a_destination(solution, dest_idx)
        

class RandomDestinationsRemoval(L1DestroyOperator):
    
    def apply(self, problem, solution):
        dests_in_routes = np.where(solution.destination_vehicle_assignmests != NO_VEHICLE)[0]
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(dests_in_routes))
        # pick randomly
        dests_to_remove = np.random.choice(dests_in_routes, num_to_remove, replace=False)
        for dest_idx in dests_to_remove:
            remove_a_destination(solution, dest_idx)
             
class WorstDestinationsRemoval(L1DestroyOperator):
    
    def apply(self, problem, solution):
        dests_in_routes = np.where(solution.destination_vehicle_assignmests != NO_VEHICLE)[0]
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
        
class RandomRouteSegmentRemoval(L1DestroyOperator):
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
