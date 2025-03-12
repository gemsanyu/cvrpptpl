from random import random, shuffle

import numpy as np

from heuristic.reinsertion_operator import BestPositionReinsertionOperator
from heuristic.operator import Operator
from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

"""
   I think re-assignment is also similar to 
   re-insertion
   1. assign unassigned customers (self-pickup and flexible): in what order?
   2. re-assign them: which assignment? the best not always available (anymore because of previous re-assignment in the ongoing process)
   so -> re-assignment order, re-assignment decision
   
   but we need one other thing -> mrt usage
   I think Mrt usage can only be changed, if 
   1. the end station locker is not in the route yet, 
   2. and using it is possible 
    a. mrt line capacity
    b. start station is already in the route, is that vehicle + end station locker load <= vehicle capacity
   I think we can use a simple heuristic for this, I think..
   1. always the best usage?
   2. randomize? (CURRENT ATTEMPT)
   
   and try changing mrt usage via local search,, i think this is
   the best one. 
"""

class BestReassignmentOperator(BestPositionReinsertionOperator):
    def find_best_assignment(self, problem: Cvrpptpl, solution: Solution, cust_idx: int):
        alternative_dests_idx = problem.destination_alternatives[cust_idx]
        best_assignment_d_cost = 99999
        best_dest_idx = None
        demand = problem.demands[cust_idx]
        for dest_idx in alternative_dests_idx:
            d_cost = 0
            # if locker, then check if locker load suffice?
            # if locker is in route (in route or mrt line is used for this locker)
            # 1. check vehicle capacity
            # 2. check mrt capacity if used
            if dest_idx != cust_idx:
                if solution.locker_loads[dest_idx] + demand > problem.locker_capacities[dest_idx]:
                    continue
                v_idx = solution.destination_vehicle_assignmests[dest_idx]
                incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[dest_idx]
                if incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]:
                    # mrt line capacity check
                    if solution.mrt_loads[incoming_mrt_line_idx] + demand > problem.mrt_line_capacities[incoming_mrt_line_idx]:
                        continue
                    # get route idx
                    start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
                    v_idx = solution.destination_vehicle_assignmests[start_station_idx]
                    assert v_idx > -1 # ensure if mrt is used then start station is of course in routes
                if v_idx != -1 and solution.vehicle_loads[v_idx] + demand > problem.vehicle_capacities[v_idx]:
                    continue
            
            # if home delivery then just compute the distance_cost
            # or if locker, and locker not in route yet
            # then compute best potential distance cost
            # TODO: actually if locker and unserved by any vehicle, 
            # also consider the distance cost if MRT is used,, should we average this? 
            if dest_idx == cust_idx or solution.destination_vehicle_assignmests[dest_idx]==-1:
                best_distance_cost, _, _ = self.find_best_insertion_pos(
                    solution.routes,
                    dest_idx,
                    demand,
                    solution.vehicle_loads,
                    problem.vehicle_capacities,
                    problem.vehicle_costs,
                    problem.distance_matrix                            
                )
                d_cost += best_distance_cost
            
            if dest_idx != cust_idx:
                locker_cost = problem.locker_costs[dest_idx]*demand
                d_cost += locker_cost
                # if locker is already served via mrt,, also add mrt line cost
                incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[dest_idx]
                if incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]:
                    mrt_cost = solution.mrt_line_costs[incoming_mrt_line_idx]*demand
                    d_cost += mrt_cost
            
            if d_cost < best_assignment_d_cost:
                best_assignment_d_cost = d_cost
                best_dest_idx = dest_idx
        return best_assignment_d_cost, best_dest_idx
        
    def best_reassignment(self, problem: Cvrpptpl, solution: Solution, cust_idx: int):
        best_assignment_d_cost, best_dest_idx = self.find_best_assignment(problem, solution, cust_idx)
        assert best_dest_idx is not None
        demand = problem.demands[cust_idx]
        solution.package_destinations[cust_idx] = best_dest_idx
        # if locker then, add locker demand, add locker cost
        if best_dest_idx != cust_idx:
            solution.locker_loads[best_dest_idx] += demand
            solution.total_locker_charge += problem.locker_costs[best_dest_idx]*demand
        v_idx = solution.destination_vehicle_assignmests[best_dest_idx]
        if v_idx != -1:
            solution.vehicle_loads[v_idx] += demand
            solution.destination_total_demands[best_dest_idx] += demand
            return # done
        #if locker, and its not in the route, check
        # if it is actually served by mrt line, and start station must have been in the route     
        incoming_mrt_location_idx = solution.incoming_mrt_lines_idx[best_dest_idx]
        if incoming_mrt_location_idx is None:
            return
        
        if not solution.mrt_usage_masks[incoming_mrt_location_idx]:
            return
        start_station_idx = problem.mrt_lines[incoming_mrt_location_idx].start_station.idx
        v_idx = solution.destination_vehicle_assignmests[start_station_idx]
        assert v_idx != -1
        solution.vehicle_loads[v_idx] += demand
        solution.destination_total_demands[start_station_idx] += demand
        solution.mrt_loads[incoming_mrt_location_idx] += demand
        solution.total_mrt_charge += demand*problem.mrt_line_costs[incoming_mrt_location_idx]    

    def set_unserved_lockers_mrt_usage(self, problem:Cvrpptpl, solution: Solution):
        for locker in problem.lockers:
            locker_idx = locker.idx
            locker_demand = solution.locker_loads[locker_idx]
            if locker_demand == 0:
                continue    
            dest_idx = locker_idx
            try:
                incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[locker_idx]
                if incoming_mrt_line_idx is None:
                    continue
                if solution.mrt_usage_masks[incoming_mrt_line_idx]:
                    continue
                # now this locker is gonna be used,,
                # it can be served by MRT, but currently not considered yet
                # so let's just randomly determine whether to use mrt or not
                if random()>0.5:
                    continue
                # now use mrt, can we? check mrt cap and vehicle cap if start station already in route
                if solution.mrt_loads[incoming_mrt_line_idx] + locker_demand > solution.mrt_line_capacities[incoming_mrt_line_idx]:
                    continue
                start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
                v_idx = solution.destination_vehicle_assignmests[start_station_idx]
                if v_idx!=-1 and solution.vehicle_loads[v_idx] + locker_demand > solution.vehicle_capacities[v_idx]:
                    continue
                solution.mrt_usage_masks[incoming_mrt_line_idx] = True
                solution.mrt_loads[incoming_mrt_line_idx] += locker_demand
                solution.total_mrt_charge += problem.mrt_line_costs[incoming_mrt_line_idx]*locker_demand
                # if using mrt, then the destination is now the start station idx
                dest_idx = start_station_idx
            finally:
                solution.destination_total_demands[dest_idx] += locker_demand
        
            
# random
class RandomOrderBestReassignment(BestReassignmentOperator):
    def apply(self, problem, solution):
        custs_to_reassign_idx = [customer.idx for customer in problem.customers if (customer.is_flexible or customer.is_self_pickup) and solution.package_destinations[customer.idx]==-1]
        shuffle(custs_to_reassign_idx)
        for cust_idx in custs_to_reassign_idx:
            self.best_reassignment(problem, solution, cust_idx)
        self.set_unserved_lockers_mrt_usage(problem, solution)
# worst customer first
# partial worst customer first