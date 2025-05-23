import math

import numpy as np

from problem.cvrpptpl import Cvrpptpl
from heuristic.solution import Solution, NO_VEHICLE, NO_DESTINATION

def complete_customers_removal(problem: Cvrpptpl, solution: Solution, custs_idx: np.ndarray):
    for cust_idx in custs_idx:
        demand = problem.demands[cust_idx]
        dest_idx = solution.package_destinations[cust_idx]
        if dest_idx == cust_idx: #home delivery
            remove_a_destination(solution, dest_idx)
            if problem.customers[cust_idx-1].is_flexible:
                solution.destination_total_demands[cust_idx] = 0
                solution.package_destinations[cust_idx] = NO_DESTINATION
            continue
        # if self pickup -> do not remove destination yet, it's a little bit
        # more complicated
        # remove demand from locker
        solution.locker_loads[dest_idx] -= demand
        solution.total_locker_charge -= solution.locker_costs[dest_idx]
        solution.package_destinations[cust_idx] = NO_DESTINATION
        # remove from mrt line
        incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[dest_idx]
        using_mrt = incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]
        if using_mrt:
            start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
            solution.destination_total_demands[start_station_idx] -= demand
            solution.mrt_loads[incoming_mrt_line_idx] -= demand
            solution.total_mrt_charge -= problem.mrt_line_costs[incoming_mrt_line_idx]*demand
            v_idx = solution.destination_vehicle_assignmests[start_station_idx]
            if v_idx != NO_VEHICLE:
                solution.vehicle_loads[v_idx] -= demand
        else:
            solution.destination_total_demands[dest_idx]-=demand
            v_idx = solution.destination_vehicle_assignmests[dest_idx]
            if v_idx != NO_VEHICLE:
                solution.vehicle_loads[v_idx] -= demand
        solution.check_validity()
    
    # remove useless node (0 demand from that destination)
    dests_in_routes = np.where(solution.destination_vehicle_assignmests != NO_VEHICLE)[0]
    lockers_in_routes = dests_in_routes[np.where(dests_in_routes>problem.num_customers)[0]]
    for locker_idx in lockers_in_routes:
        if solution.destination_total_demands[locker_idx]>0:
            continue
        remove_a_destination(solution, locker_idx)
    solution.check_validity()

def compute_customer_removal_d_costs(problem: Cvrpptpl, solution: Solution):
    cust_d_costs: np.ndarray = np.zeros([problem.num_customers+1,], dtype=float)
    for customer in problem.customers:
        cust_idx = customer.idx
        demand = customer.demand
        # check if it is home delivery, then also calculate its 
        # removal_d_cost from the route
        v_idx = solution.destination_vehicle_assignmests[cust_idx]
        if v_idx != NO_VEHICLE:
            pos = solution.routes[v_idx].index(cust_idx)
            prev_dest_idx = solution.routes[v_idx][pos-1]
            next_dest_idx = solution.routes[v_idx][(pos+1)%len(solution.routes[v_idx])]
            related_arc_costs = problem.distance_matrix[[prev_dest_idx, cust_idx, prev_dest_idx],[cust_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[v_idx]
            prev_to_dest_cost, pos_to_dest_cost, prev_to_next_cost = related_arc_costs
            d_cost = prev_to_next_cost -(prev_to_dest_cost + pos_to_dest_cost)
            cust_d_costs[cust_idx] += d_cost
            continue
        
        # if it is self pickup:
        # compute its locker cost (demand-related)
        # if the locker use mrt line
        # also compute mrt line usage cost (demand-related)
        locker_idx = solution.package_destinations[cust_idx]
        locker_load_cost = solution.locker_costs[locker_idx]
        cust_d_costs[cust_idx] += locker_load_cost
        incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[locker_idx]
        is_using_mrt = incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]
        if is_using_mrt:
            mrt_load_cost = solution.mrt_line_costs[incoming_mrt_line_idx]*demand
            cust_d_costs[cust_idx] += mrt_load_cost
        # also consider the estimated cost (whats this called)?
        # of removing this customer, also leads to destination (locker or mrt start station)
        # removal from the routes
        dest_idx = locker_idx
        if is_using_mrt:
            dest_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
            v_idx = solution.destination_vehicle_assignmests[dest_idx]
            if v_idx == NO_VEHICLE:
                continue
            pos = solution.routes[v_idx].index(dest_idx)
            prev_dest_idx = solution.routes[v_idx][pos-1]
            next_dest_idx = solution.routes[v_idx][(pos+1)%len(solution.routes[v_idx])]
            related_arc_costs = problem.distance_matrix[[prev_dest_idx, dest_idx, prev_dest_idx],[dest_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[v_idx]
            prev_to_dest_cost, pos_to_dest_cost, prev_to_next_cost = related_arc_costs
            d_cost = prev_to_next_cost -(prev_to_dest_cost + pos_to_dest_cost)
            demand_ratio = demand/solution.destination_total_demands[dest_idx]
            cust_d_costs[cust_idx] += demand_ratio*d_cost
    return cust_d_costs[1:]

def remove_a_destination(solution: Solution,
                         dest_idx: int):
    problem = solution.problem
    v_idx = solution.destination_vehicle_assignmests[dest_idx]
    # print(dest_idx, solution.locker_loads[dest_idx], solution.destination_total_demands[dest_idx])
    pos = solution.routes[v_idx].index(dest_idx)
    prev_dest_idx = solution.routes[v_idx][pos-1]
    next_dest_idx = solution.routes[v_idx][(pos+1)%len(solution.routes[v_idx])]
    
    
    related_arc_costs = problem.distance_matrix[[prev_dest_idx, dest_idx, prev_dest_idx],[dest_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[v_idx]
    prev_to_dest_cost, pos_to_dest_cost, prev_to_next_cost = related_arc_costs
    d_cost = prev_to_next_cost -(prev_to_dest_cost + pos_to_dest_cost)
    
    
    solution.total_vehicle_charge = solution.total_vehicle_charge + d_cost
    # remove from route
    solution.destination_vehicle_assignmests[dest_idx] = NO_VEHICLE
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
    prev_dest_idx = solution.routes[vehicle_idx][start_idx-1]
    next_dest_idx = solution.routes[vehicle_idx][end_idx%len(solution.routes[vehicle_idx])]
    r = [prev_dest_idx] + dests_to_remove + [next_dest_idx]
    d_cost = -problem.distance_matrix[r[:-1], r[1:]].sum()*problem.vehicle_costs[vehicle_idx] + problem.distance_matrix[prev_dest_idx, next_dest_idx]
    solution.total_vehicle_charge = solution.total_vehicle_charge + d_cost
    solution.destination_vehicle_assignmests[dests_to_remove] = NO_VEHICLE
    # remove from route
    solution.routes[vehicle_idx] = solution.routes[vehicle_idx][:start_idx] + solution.routes[vehicle_idx][end_idx:]
    solution.vehicle_loads[vehicle_idx] -= np.sum(solution.destination_total_demands[dests_to_remove])

def compute_locker_removal_d_costs(problem: Cvrpptpl, solution: Solution, lockers_idx: np.ndarray):
    locker_removal_d_costs: np.ndarray = np.zeros([len(lockers_idx),], dtype=float)
    customers_removal_d_costs: np.ndarray = compute_customer_removal_d_costs(problem, solution)
    for i, locker_idx in enumerate(lockers_idx):
        locker_custs_idx = np.where(solution.package_destinations == locker_idx)[0]
        cust_removal_d_costs = customers_removal_d_costs[locker_custs_idx-1]
        locker_removal_d_costs[i] = np.sum(cust_removal_d_costs)
    return locker_removal_d_costs


def complete_locker_removal(problem: Cvrpptpl, solution: Solution, locker_idx: np.ndarray):
    locker_custs_idx = np.where(solution.package_destinations==locker_idx)[0]
    complete_customers_removal(problem, solution, locker_custs_idx)
    