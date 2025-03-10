from typing import List

import numba as nb
import numpy as np

from heuristic.solution import Solution

def get_destinations_to_reinsert(solution: Solution):
    return get_destinations_to_reinsert_with_nb(solution.destination_total_demands, solution.destination_vehicle_assignmests)

@nb.jit(nb.int64[:](nb.int64[:],nb.int64[:]),nopython=True,cache=True)
def get_destinations_to_reinsert_with_nb(destination_total_demands:np.ndarray, destination_vehicle_assignmests):
    is_dest_need_visit = np.logical_and(destination_total_demands > 0, destination_vehicle_assignmests == -1)
    dests_to_reinsert = np.where(is_dest_need_visit)[0]
    return dests_to_reinsert

@nb.jit(nb.types.UniTuple(nb.float64,2)(nb.int64[:],nb.int64,nb.float64[:,:],nb.float64),nopython=True,cache=True)
def find_best_insertion_in_route(route: np.ndarray,
                                dest_idx: int,
                                distance_matrix: np.ndarray,
                                vehicle_cost: float):
    best_d_cost: float = 999999
    best_position: int = -1
    route_len = len(route)
    for i in range(route_len):
        prev_dest_idx = route[i]
        next_dest_idx = route[(i+1)%route_len]
        dist_from = distance_matrix[prev_dest_idx, dest_idx]
        dist_to = distance_matrix[dest_idx, next_dest_idx]
        dist_in_route = distance_matrix[prev_dest_idx, next_dest_idx]
        d_cost = (dist_from + dist_to - dist_in_route)*vehicle_cost
        if d_cost<best_d_cost:
            best_d_cost = d_cost
            best_position = i
    return best_d_cost, best_position

@nb.jit(nb.types.Tuple((nb.float64,nb.int64,nb.int64))(nb.int64[:,:],nb.int64[:],nb.int64,nb.int64,nb.int64[:],nb.int64[:],nb.float64[:],nb.float64[:,:],nb.float64[:],nb.int64[:]),nopython=True,cache=True,parallel=True)
def find_best_insertion_pos(routes: np.ndarray,
                            routes_len: np.ndarray,
                            dest_idx: int,
                            demand: int,
                            vehicle_loads: np.ndarray,
                            vehicle_capacities: np.ndarray,
                            vehicle_costs: np.ndarray,
                            distance_matrix: np.ndarray,
                            r_cost_tmp_arr: np.ndarray,
                            pos_tmp_arr: np.ndarray):
    n_vehicle = len(routes)
    for v_idx in nb.prange(n_vehicle):
        if vehicle_loads[v_idx] + demand > vehicle_capacities[v_idx]:
            r_cost_tmp_arr[v_idx] = 999999
            continue
        best_r_cost, best_r_pos = find_best_insertion_in_route(routes[v_idx, :routes_len[v_idx]],
                                                               dest_idx,
                                                               distance_matrix,
                                                               vehicle_costs[v_idx])
        r_cost_tmp_arr[v_idx] = best_r_cost
        pos_tmp_arr[v_idx] = best_r_pos
    
    best_v_idx = np.argmin(r_cost_tmp_arr)
    best_d_cost = r_cost_tmp_arr[best_v_idx]
    best_pos = pos_tmp_arr[best_v_idx]
    return best_d_cost, best_pos, best_v_idx
    
    
def compute_regret_of_dest_reinsertion(routes: List[List[int]],
                   dest_idx: int,
                   demand: int,
                   vehicle_loads: np.ndarray,
                   vehicle_capacities: np.ndarray,
                   vehicle_costs: np.ndarray,
                   distance_matrix: np.ndarray)->float:
    best_d_cost = 999999
    best_d_cost2 = 999999
    for v_idx, route in enumerate(routes):
        if demand + vehicle_loads[v_idx] > vehicle_capacities[v_idx]:
            continue
        route_len = len(route)
        for i in range(route_len):
            prev_dest_idx = route[i]
            next_dest_idx = route[(i+1)%route_len]
            dist_from = distance_matrix[prev_dest_idx, dest_idx]
            dist_to = distance_matrix[dest_idx, next_dest_idx]
            dist_in_route = distance_matrix[prev_dest_idx, next_dest_idx]
            d_cost = (dist_from + dist_to - dist_in_route)*vehicle_costs[v_idx]
            if d_cost<best_d_cost2:
                if d_cost<best_d_cost:
                    best_d_cost2 = best_d_cost
                    best_d_cost = d_cost
                else:
                    best_d_cost2 = d_cost
    return best_d_cost2-best_d_cost

def compute_regrets(routes: List[List[int]],
                    dests_idx: np.ndarray,
                    destination_total_demands: np.ndarray,
                    vehicle_loads: np.ndarray,
                    vehicle_capacities: np.ndarray,
                    vehicle_costs: np.ndarray,
                    distance_matrix: np.ndarray,
                    ):
    regrets = np.zeros([len(dests_idx),], dtype=float)
    for i, dest_idx in enumerate(dests_idx):
        regrets[i] = compute_regret_of_dest_reinsertion(routes,
                                                        dest_idx,
                                                        destination_total_demands[dest_idx],
                                                        vehicle_loads,
                                                        vehicle_capacities,
                                                        vehicle_costs,
                                                        distance_matrix)
    return regrets