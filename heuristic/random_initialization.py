from random import shuffle, random

import numpy as np

from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

def first_fit_destination_assignment(node_idx: int, problem: Cvrpptpl, solution:Solution)->bool:
    feasible_assignment_found: bool = False
    if node_idx>problem.num_customers:
        return True
    alternatives = problem.destination_alternatives[node_idx]
    shuffle(alternatives)
    for dest_idx in alternatives:
        # if the considered alternative is to use a locker
        if dest_idx != node_idx:
            if solution.locker_loads[dest_idx] + problem.demands[node_idx] > problem.locker_capacities[dest_idx]:
                continue
            solution.locker_loads[dest_idx] += problem.demands[node_idx]
            feasible_assignment_found = first_fit_destination_assignment(node_idx+1, problem, solution) 
            if feasible_assignment_found:
                # commit other changes
                solution.package_destinations[node_idx]=dest_idx
                solution.total_locker_charge += problem.demands[node_idx]*problem.locker_costs[dest_idx]
                return True
            else:
                # revert load change
                solution.locker_loads[dest_idx] -= problem.demands[node_idx]
            continue
        # if home delivery
        solution.destination_total_demands[node_idx] += problem.demands[node_idx]
        feasible_assignment_found = first_fit_destination_assignment(node_idx+1, problem, solution)
        if feasible_assignment_found:
            # commit other changes
            solution.package_destinations[node_idx]=node_idx
            return True
        else:
            # revert load change
            solution.destination_total_demands[node_idx] -= problem.demands[node_idx]
    return False

def randomize_mrt_line_usage(problem: Cvrpptpl, solution:Solution):
    """if locker has no mrt lines
    then set the destination total demands to its loads
    else check if we want to use the mrt line
    if we want to use the mrt line
    then apply mrt line cost
    and add the locker loads to the start station destination total demands

    Args:
        problem (Cvrpptpl): _description_
        solution (Solution): _description_
    """
    for locker in problem.lockers:
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[locker.idx]
        if incoming_mrt_line_idx is None:
            solution.destination_total_demands[locker.idx] = solution.locker_loads[locker.idx]
            continue
        # 1/2 chance for trying using the mrt line
        # remember 1 line only serve 1 locker as end station
        if solution.locker_loads[locker.idx] > problem.mrt_line_capacities[incoming_mrt_line_idx]:
            continue
        if random() <= 0.5:
            solution.mrt_usage_masks[incoming_mrt_line_idx] = True
            solution.mrt_loads[incoming_mrt_line_idx] = solution.locker_loads[locker.idx]
            start_station_idx = solution.problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
            solution.destination_total_demands[start_station_idx] += solution.locker_loads[locker.idx]
            solution.mrt_line_costs += solution.mrt_loads[incoming_mrt_line_idx]*problem.mrt_line_costs[incoming_mrt_line_idx]
        else:
            solution.destination_total_demands[locker.idx] += solution.locker_loads[locker.idx]

def greedy_route_insertion(problem: Cvrpptpl, solution: Solution):
    required_destinations = np.nonzero(solution.destination_total_demands)[0]
    # for locker in problem.lockers:
    #     dest_idx = locker.idx
    #     print(dest_idx, solution.locker_loads[dest_idx], solution.destination_total_demands[dest_idx])
    
    
    # exit()
    # sort them based on their distance to depot
    required_destinations_distance_to_depot = problem.distance_matrix[required_destinations, 0]
    sorted_idx = np.argsort(required_destinations_distance_to_depot)
    required_destinations = required_destinations[sorted_idx]
    # now we know no-one is in route yet, so no further checking needed
    vehicle_current_locations: np.ndarray = np.zeros([problem.num_vehicles,], dtype=int)
    for dest_idx in required_destinations:
        distance_from_vehicles = problem.distance_matrix[vehicle_current_locations, dest_idx]
        cost_from_vehicles = distance_from_vehicles*problem.vehicle_costs
        is_vehicle_feasible = solution.vehicle_loads + problem.demands[dest_idx] < problem.vehicle_capacities
        if not np.any(is_vehicle_feasible):
            return False
        closest_vehicle_idx = np.where(is_vehicle_feasible, distance_from_vehicles, np.inf).argmin()
        # assigne to closest vehicle
        solution.destination_vehicle_assignmests[dest_idx] = closest_vehicle_idx
        solution.vehicle_loads[closest_vehicle_idx] += problem.demands[dest_idx]
        solution.total_vehicle_charge += cost_from_vehicles[closest_vehicle_idx]
        solution.routes[closest_vehicle_idx] += [dest_idx]
        vehicle_current_locations[closest_vehicle_idx] = dest_idx
    # add cost to depot for all route
    for v_idx, route in enumerate(solution.routes):
        last_dest_idx = route[-1]
        last_to_depot_dist = problem.distance_matrix[last_dest_idx, 0]
        solution.total_vehicle_charge += last_to_depot_dist*problem.vehicle_costs[v_idx]   
    return True

def random_initialization(problem: Cvrpptpl)->Solution:
    new_solution: Solution
    feasible_solution_found: bool = False
    while not feasible_solution_found:
        new_solution = Solution(problem)
        feasible_assignment_found = first_fit_destination_assignment(1, problem, new_solution)
        if not feasible_assignment_found:
            raise ValueError("cannot find any feasible destination assignment")
        randomize_mrt_line_usage(problem, new_solution)
        feasible_route_found = greedy_route_insertion(problem, new_solution)
        if not feasible_route_found:
            continue
        feasible_solution_found = True
    return new_solution