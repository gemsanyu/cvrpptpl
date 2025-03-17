from enum import Enum
from random import shuffle
from typing import List

import numpy as np

from heuristic.operator import Operator
from heuristic.r_op_utils import find_best_insertion_pos, get_destinations_to_reinsert, compute_regrets
from heuristic.solution import Solution, NO_VEHICLE, NO_DESTINATION
from problem.cvrpptpl import Cvrpptpl

class ReinsertionStatus(Enum):
    SUCCESS = 1
    INFEASIBLE = 2

class BestPositionReinsertionOperator(Operator):
    def __init__(self, problem: Cvrpptpl):
        self.find_best_insertion_pos = self.prepare_find_best_insertion_pos(problem)
        super().__init__()
    
    def prepare_find_best_insertion_pos(self, problem: Cvrpptpl):
        num_nodes:int  = problem.num_nodes
        num_vehicles:int = problem.num_vehicles
        route_tmp_arr: np.ndarray = np.zeros([num_vehicles, num_nodes+2], dtype=int)
        routes_len_tmp_arr: np.ndarray = np.empty([num_vehicles,], dtype=int)
        r_cost_tmp_arr: np.ndarray = np.empty([num_vehicles,], dtype=float)
        pos_tmp_arr: np.ndarray = np.empty([num_vehicles,], dtype=int)
        def find_best_insertion_pos_s(routes: List[List[int]],
                                    dest_idx: int,
                                    demand: int,
                                    vehicle_loads: np.ndarray,
                                    vehicle_capacities: np.ndarray,
                                    vehicle_costs: np.ndarray,
                                    distance_matrix: np.ndarray):
            for v_idx, route in enumerate(routes):
                route_len = len(route)
                routes_len_tmp_arr[v_idx] = route_len
                route_tmp_arr[v_idx, :route_len] = route
                route_tmp_arr[v_idx, route_len] = 0
            result = find_best_insertion_pos(route_tmp_arr,
                                             routes_len_tmp_arr,
                                             dest_idx,
                                             demand,
                                             vehicle_loads,
                                             vehicle_capacities,
                                             vehicle_costs,
                                             distance_matrix,
                                             r_cost_tmp_arr,
                                             pos_tmp_arr)
            best_d_cost, best_pos, best_v_idx = result
            return best_d_cost, best_pos, best_v_idx
        return find_best_insertion_pos_s

    def reinsert_dests_to_best_position(self, problem: Cvrpptpl, solution: Solution, dests_to_reinsert: np.ndarray):
        for dest_idx in dests_to_reinsert:
            demand = solution.destination_total_demands[dest_idx]
            best_d_cost, best_position, best_v_idx = self.find_best_insertion_pos(
                solution.routes,
                dest_idx,
                solution.destination_total_demands[dest_idx],
                solution.vehicle_loads,
                problem.vehicle_capacities,
                problem.vehicle_costs,
                problem.distance_matrix                            
            )
            if best_d_cost > 99999:
                return ReinsertionStatus.INFEASIBLE
            solution.routes[best_v_idx] = solution.routes[best_v_idx][:best_position+1] + [dest_idx] + solution.routes[best_v_idx][best_position+1:]
            solution.destination_vehicle_assignmests[dest_idx] = best_v_idx
            solution.vehicle_loads[best_v_idx] += demand
            solution.total_vehicle_charge += best_d_cost
        return ReinsertionStatus.SUCCESS
class RandomOrderBestPosition(BestPositionReinsertionOperator):    
    def apply(self, problem, solution):
        dests_to_reinsert = get_destinations_to_reinsert(solution)
        shuffle(dests_to_reinsert)
        status = self.reinsert_dests_to_best_position(problem, solution, dests_to_reinsert)
        return status

class HighestRegretBestPosition(BestPositionReinsertionOperator):
    def apply(self, problem, solution):
        """
        order by highest cost regret
        insert into best position, 
        static update->dont update regret table after every insertion, 
        too costly

        Args:
            problem (_type_): _description_
            solution (_type_): _description_
        """
        dests_to_reinsert = get_destinations_to_reinsert(solution)
        dests_regret = compute_regrets(solution.routes,
                                       dests_to_reinsert,
                                       solution.destination_total_demands,
                                       solution.vehicle_loads,
                                       problem.vehicle_capacities,
                                       problem.vehicle_costs,
                                       problem.distance_matrix)
        sorted_idx = np.argsort(dests_regret)
        dests_to_reinsert = dests_to_reinsert[sorted_idx]
        status = self.reinsert_dests_to_best_position(problem, solution, dests_to_reinsert)
        return status
