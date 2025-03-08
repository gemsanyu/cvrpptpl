from random import shuffle

import numpy as np

from heuristic.operator import Operator
from heuristic.solution import Solution


class RandomOrderBestPosition(Operator):
    def apply(self, problem, solution):
        dests_to_reinsert = get_destinations_to_reinsert(solution)
        shuffle(dests_to_reinsert)
        for dest_idx in dests_to_reinsert:
            demand = solution.destination_total_demands[dest_idx]
            best_d_cost = np.inf
            best_position = None
            best_v_idx = None
            for v_idx, route in enumerate(solution.routes):                
                if solution.vehicle_loads[v_idx] + demand > problem.vehicle_capacities[v_idx]:
                    continue
                route_ = route[1:] + route[:1]
                dist_from = problem.distance_matrix[route, dest_idx]
                dist_to = problem.distance_matrix[dest_idx, route_]
                dist_in_route = problem.distance_matrix[route, route_]
                d_cost = dist_from + dist_to - dist_in_route
                best_r_pos = np.argmin(d_cost)
                best_r_d_cost = d_cost[best_r_pos]
                if best_r_d_cost < best_d_cost:
                    best_d_cost = best_r_d_cost
                    best_position = best_r_pos
                    best_v_idx = v_idx
            # insert and commit
            solution.routes[best_v_idx] = solution.routes[best_v_idx][:best_position+1] + [dest_idx] + solution.routes[best_v_idx][best_position+1:]
            solution.destination_vehicle_assignmests[dest_idx] = best_v_idx
            solution.vehicle_loads[best_v_idx] += demand
            solution.total_vehicle_charge += best_d_cost
            
class HighestRegretBestPosition(Operator):
    def apply(self, problem, solution):
        dests_to_reinsert = get_destinations_to_reinsert(solution)
        shuffle(dests_to_reinsert)
        for dest_idx in dests_to_reinsert:
            demand = solution.destination_total_demands[dest_idx]
            best_d_cost = np.inf
            best_position = None
            best_v_idx = None
            for v_idx, route in enumerate(solution.routes):                
                if solution.vehicle_loads[v_idx] + demand > problem.vehicle_capacities[v_idx]:
                    continue
                route_ = route[1:] + route[:1]
                dist_from = problem.distance_matrix[route, dest_idx]
                dist_to = problem.distance_matrix[dest_idx, route_]
                dist_in_route = problem.distance_matrix[route, route_]
                d_cost = dist_from + dist_to - dist_in_route
                best_r_pos = np.argmin(d_cost)
                best_r_d_cost = d_cost[best_r_pos]
                if best_r_d_cost < best_d_cost:
                    best_d_cost = best_r_d_cost
                    best_position = best_r_pos
                    best_v_idx = v_idx
            # insert and commit
            solution.routes[best_v_idx] = solution.routes[best_v_idx][:best_position+1] + [dest_idx] + solution.routes[best_v_idx][best_position+1:]
            solution.destination_vehicle_assignmests[dest_idx] = best_v_idx
            solution.vehicle_loads[best_v_idx] += demand
            solution.total_vehicle_charge += best_d_cost

def get_destinations_to_reinsert(solution: Solution):
    is_dest_need_visit = np.logical_and(solution.destination_total_demands > 0, solution.destination_vehicle_assignmests == -1)
    dests_to_reinsert = np.where(is_dest_need_visit)[0]
    return dests_to_reinsert