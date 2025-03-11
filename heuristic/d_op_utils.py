import numpy as np

from problem.cvrpptpl import Cvrpptpl
from heuristic.solution import Solution

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