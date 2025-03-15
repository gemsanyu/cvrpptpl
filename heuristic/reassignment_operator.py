from dataclasses import dataclass
from random import random, shuffle
from typing import List, Set

import numpy as np

from heuristic.reinsertion_operator import BestPositionReinsertionOperator
from heuristic.operator import Operator
from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl


@dataclass(frozen=True)
class ReassignmentTask:
    cust_idx: int
    dest_idx: int
    d_cost: float

# def generate_reassignment_tasks()
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

# class BestReassignmentOperator(BestPositionReinsertionOperator):
#     def find_best_assignment(self, problem: Cvrpptpl, solution: Solution, cust_idx: int):
#         alternative_dests_idx = problem.destination_alternatives[cust_idx]
#         best_assignment_d_cost = 99999
#         best_dest_idx = None
#         demand = problem.demands[cust_idx]
#         for dest_idx in alternative_dests_idx:
#             d_cost = 0
#             # if locker, then check if locker load suffice?
#             # if locker is in route (in route or mrt line is used for this locker)
#             # 1. check vehicle capacity
#             # 2. check mrt capacity if used
#             if dest_idx != cust_idx:
#                 if solution.locker_loads[dest_idx] + demand > problem.locker_capacities[dest_idx]:
#                     continue
#                 v_idx = solution.destination_vehicle_assignmests[dest_idx]
#                 incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[dest_idx]
#                 if incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]:
#                     # mrt line capacity check
#                     if solution.mrt_loads[incoming_mrt_line_idx] + demand > problem.mrt_line_capacities[incoming_mrt_line_idx]:
#                         continue
#                     # get route idx
#                     start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
#                     v_idx = solution.destination_vehicle_assignmests[start_station_idx]
#                     assert v_idx > -1 # ensure if mrt is used then start station is of course in routes
#                 if v_idx != -1 and solution.vehicle_loads[v_idx] + demand > problem.vehicle_capacities[v_idx]:
#                     continue
            
#             # if home delivery then just compute the distance_cost
#             # or if locker, and locker not in route yet
#             # then compute best potential distance cost
#             # TODO: actually if locker and unserved by any vehicle, 
#             # also consider the distance cost if MRT is used,, should we average this? 
#             if dest_idx == cust_idx or solution.destination_vehicle_assignmests[dest_idx]==-1:
#                 best_d_cost, best_pos, best_v_idx = self.find_best_insertion_pos(
#                     solution.routes,
#                     dest_idx,
#                     demand,
#                     solution.vehicle_loads,
#                     problem.vehicle_capacities,
#                     problem.vehicle_costs,
#                     problem.distance_matrix                            
#                 )
#                 print(best_d_cost, best_pos, best_v_idx)
#                 d_cost += best_d_cost
#             # print(d_cost,"1", best_pos, best_d_cost)
            
#             if dest_idx != cust_idx:
#                 locker_cost = problem.locker_costs[dest_idx]*demand
#                 d_cost += locker_cost
#                 # print(d_cost,"2")
            
#                 # if locker is already served via mrt,, also add mrt line cost
#                 incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[dest_idx]
#                 if incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]:
#                     mrt_cost = solution.mrt_line_costs[incoming_mrt_line_idx]*demand
#                     d_cost += mrt_cost
#                     # print(d_cost,"3")
            
#             if d_cost < best_assignment_d_cost:
#                 best_assignment_d_cost = d_cost
#                 best_dest_idx = dest_idx
#         return best_assignment_d_cost, best_dest_idx
        
#     def best_reassignment(self, problem: Cvrpptpl, solution: Solution, cust_idx: int):
#         best_assignment_d_cost, best_dest_idx = self.find_best_assignment(problem, solution, cust_idx)
#         assert best_dest_idx is not None
#         demand = problem.demands[cust_idx]
#         solution.package_destinations[cust_idx] = best_dest_idx
#         if best_dest_idx == cust_idx:
#             solution.destination_total_demands[cust_idx] = demand
#             return
#         # if locker then, add locker demand, add locker cost
#         if best_dest_idx != cust_idx:
#             solution.locker_loads[best_dest_idx] += demand
#             solution.total_locker_charge += problem.locker_costs[best_dest_idx]*demand
#         v_idx = solution.destination_vehicle_assignmests[best_dest_idx]
#         if v_idx != -1:
#             solution.vehicle_loads[v_idx] += demand
#             solution.destination_total_demands[best_dest_idx] += demand
#             return # done
#         #if locker, and its not in the route, check
#         # if it is actually served by mrt line, and start station must have been in the route     
#         incoming_mrt_location_idx = solution.incoming_mrt_lines_idx[best_dest_idx]
#         if incoming_mrt_location_idx is None:
#             return
        
#         if not solution.mrt_usage_masks[incoming_mrt_location_idx]:
#             return
#         start_station_idx = problem.mrt_lines[incoming_mrt_location_idx].start_station.idx
#         v_idx = solution.destination_vehicle_assignmests[start_station_idx]
#         assert v_idx != -1
#         solution.vehicle_loads[v_idx] += demand
#         solution.destination_total_demands[start_station_idx] += demand
#         solution.mrt_loads[incoming_mrt_location_idx] += demand
#         solution.total_mrt_charge += demand*problem.mrt_line_costs[incoming_mrt_location_idx]    

#     def set_unserved_lockers_mrt_usage(self, problem:Cvrpptpl, solution: Solution):
#         for locker in problem.lockers:
#             locker_idx = locker.idx
#             locker_demand = solution.locker_loads[locker_idx]
#             if locker_demand == 0:
#                 continue
#             dest_idx = locker_idx
#             try:
#                 incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[locker_idx]
#                 if incoming_mrt_line_idx is None:
#                     continue
#                 if solution.mrt_usage_masks[incoming_mrt_line_idx]:
#                     continue
#                 # now this locker is gonna be used,,
#                 # it can be served by MRT, but currently not considered yet
#                 # so let's just randomly determine whether to use mrt or not
#                 if random()>0.5:
#                     continue
#                 # now use mrt, can we? check mrt cap and vehicle cap if start station already in route
#                 if solution.mrt_loads[incoming_mrt_line_idx] + locker_demand > solution.mrt_line_capacities[incoming_mrt_line_idx]:
#                     continue
#                 start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
#                 v_idx = solution.destination_vehicle_assignmests[start_station_idx]
#                 if v_idx!=-1 and solution.vehicle_loads[v_idx] + locker_demand > solution.vehicle_capacities[v_idx]:
#                     continue
#                 solution.mrt_usage_masks[incoming_mrt_line_idx] = True
#                 solution.mrt_loads[incoming_mrt_line_idx] += locker_demand
#                 solution.total_mrt_charge += problem.mrt_line_costs[incoming_mrt_line_idx]*locker_demand
#                 # if using mrt, then the destination is now the start station idx
#                 dest_idx = start_station_idx
#             finally:
#                 solution.check_validity()
#                 solution.destination_total_demands[dest_idx] += locker_demand
#                 v_idx = solution.destination_vehicle_assignmests[dest_idx]
#                 if v_idx != -1:
#                     solution.vehicle_loads[v_idx] += locker_demand
#                 solution.check_validity()
                
                
class BestFirstFitReassignmentOperator(BestPositionReinsertionOperator):
    def __init__(self, problem):
        super().__init__(problem)
        self.reassigned_custs_idx: Set[int] = set()
        self.custs_to_reassign_idx: Set[int] = set()
    
    def compute_assignment_d_costs(self, problem: Cvrpptpl, solution: Solution, cust_idx: int):
        alternative_dests_idx = problem.destination_alternatives[cust_idx]
        assignment_d_costs = np.full([len(alternative_dests_idx)], np.inf, dtype=float)
        demand = problem.demands[cust_idx]
        for i, dest_idx in enumerate(alternative_dests_idx):
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
                best_d_cost, best_pos, best_v_idx = self.find_best_insertion_pos(
                    solution.routes,
                    dest_idx,
                    demand,
                    solution.vehicle_loads,
                    problem.vehicle_capacities,
                    problem.vehicle_costs,
                    problem.distance_matrix                            
                )
                d_cost += best_d_cost
            
            if dest_idx != cust_idx:
                locker_cost = problem.locker_costs[dest_idx]*demand
                d_cost += locker_cost
            
                # if locker is already served via mrt,, also add mrt line cost
                incoming_mrt_line_idx = solution.incoming_mrt_lines_idx[dest_idx]
                if incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]:
                    mrt_cost = solution.mrt_line_costs[incoming_mrt_line_idx]*demand
                    d_cost += mrt_cost
            assignment_d_costs[i] = d_cost
        return assignment_d_costs
    
    def best_first_fit_reassignment(self, problem: Cvrpptpl, solution: Solution, reassignment_tasks: List[ReassignmentTask]):
        self.reassigned_custs_idx.clear()
        return self.ffr(problem, solution, reassignment_tasks, 0)
    
    def is_r_task_applicable(self, problem: Cvrpptpl, solution: Solution, r_task: ReassignmentTask)->bool:
        cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        if cust_idx == dest_idx: # home delivery,, of course applicable
            return True
        demand = problem.demands[cust_idx]
        # then this must be a locker
        if solution.locker_loads[dest_idx] + demand > problem.locker_capacities[dest_idx]:
            return False
        # okay, load is fine, what if this locker is in a route? is the vehicle load ok?
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        if v_idx != -1:
            if solution.vehicle_loads[v_idx] + demand > problem.vehicle_capacities[v_idx]:
                return False
            return True
        
        # okay this locker is not in the route, what if this locker is served by an mrt?
        # and thats why its not served in a route, but its start station is
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
        if incoming_mrt_line_idx is None:
            return True
        using_mrt = solution.mrt_usage_masks[incoming_mrt_line_idx]
        if not using_mrt:
            return True
        if solution.mrt_loads[incoming_mrt_line_idx] + demand > problem.mrt_line_capacities[incoming_mrt_line_idx]:
            return False
        start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
        v_idx = solution.destination_vehicle_assignmests[start_station_idx]
        # it's using mrt, but not in the route yet (which is actually shouldnt happen)
        assert v_idx != -1 #TODO please comment it out or delete after we are sure the algo is valid, assertion is expensive
        if solution.vehicle_loads[v_idx]+demand>problem.vehicle_capacities[v_idx]:
            return False
        return True
    
    def apply_r_task(self, problem: Cvrpptpl, solution: Solution, r_task: ReassignmentTask):
        print(r_task)
        cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        if cust_idx == dest_idx:
            return
        demand = problem.demands[cust_idx]
        solution.locker_loads[dest_idx] += demand
        solution.package_destinations[cust_idx] = dest_idx
        solution.total_locker_charge += problem.locker_costs[dest_idx]*demand
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        if v_idx!=-1:
            solution.vehicle_loads[v_idx]+=demand
            solution.destination_total_demands[dest_idx]+=demand
            return
        
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
        
        if incoming_mrt_line_idx is None:
            return
        using_mrt = solution.mrt_usage_masks[incoming_mrt_line_idx]
        if not using_mrt:
            return 
        print("HALO", solution.mrt_loads[incoming_mrt_line_idx])
        solution.mrt_loads[incoming_mrt_line_idx] += demand
        solution.total_mrt_charge += problem.mrt_line_costs[incoming_mrt_line_idx]*demand
        start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
        v_idx = solution.destination_vehicle_assignmests[start_station_idx]
        solution.vehicle_loads[v_idx] += demand
        solution.destination_total_demands[start_station_idx] += demand   
    
    def revert_r_task(self, problem: Cvrpptpl, solution: Solution, r_task: ReassignmentTask):
        cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        if cust_idx == dest_idx:
            return
        demand = problem.demands[cust_idx]
        solution.locker_loads[dest_idx] -= demand
        solution.package_destinations[cust_idx] = -1
        solution.total_locker_charge -= problem.locker_costs[dest_idx]*demand
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        if v_idx!=-1:
            solution.vehicle_loads[v_idx]-=demand
            solution.destination_total_demands[dest_idx]-=demand
            return
        
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
        if incoming_mrt_line_idx is None:
            return
        using_mrt = solution.mrt_usage_masks[incoming_mrt_line_idx]
        if not using_mrt:
            return    
        solution.mrt_loads[incoming_mrt_line_idx] -= demand
        solution.total_mrt_charge -= problem.mrt_line_costs[incoming_mrt_line_idx]*demand
        start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
        v_idx = solution.destination_vehicle_assignmests[start_station_idx]
        solution.vehicle_loads[v_idx] -= demand
        solution.destination_total_demands[start_station_idx] -= demand
        
    def ffr(self, problem: Cvrpptpl, solution: Solution, reassignment_tasks: List[ReassignmentTask], r_idx):
        """complete search (might take long), we can actually use dynamic programming
        but will be too messy to code (in such a short time).
        steps:
            1. check for terminal case
                a. all customers have been reassigned -> Feasible
                b. no more reassignment task to consider -> Infeasible
            2. current customer has been reassigned in previous iteration?
                    -> skip directly to next task.
            3. Try assigning curremt customer to current destination, proceed
                a. get feasible solution? -> commit this cutomer changes -> return True
                b. infeasible? -> revert changes
            4. Try skipping this customer in this task, proceed
                a. get feasible? return True 
                b. get infeasible? return False -> no hope left.
                
                see? its actually like TSP bitmasking dynamic programming.

        Args:
            problem (Cvrpptpl): _description_
            solution (Solution): _description_
            reassignment_tasks (List[ReassignmentTask]): _description_
            r_idx (_type_): _description_
        """
        if len(self.reassigned_custs_idx) == len(self.custs_to_reassign_idx):
            return True
        if r_idx >= len(reassignment_tasks):
            return False
        r_task = reassignment_tasks[r_idx]
        cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        if cust_idx in self.reassigned_custs_idx:
            return self.ffr(problem, solution, reassignment_tasks, r_idx+1)

        feasible_reassignment_found = False
        is_applicable = self.is_r_task_applicable(problem, solution, r_task)
        if is_applicable:
            self.apply_r_task(problem, solution, r_task)
            print(r_task)
            solution.check_validity()
            self.reassigned_custs_idx.add(cust_idx)
            feasible_reassignment_found = self.ffr(problem, solution, reassignment_tasks, r_idx+1)
            if feasible_reassignment_found:
                return feasible_reassignment_found
            # :cry not found yet
            self.revert_r_task(problem, solution, r_task)
            self.reassigned_custs_idx.remove(cust_idx)
        # try skipping this task
        feasible_reassignment_found = self.ffr(problem, solution, reassignment_tasks, r_idx+1)
        return feasible_reassignment_found
    
    
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
                solution.check_validity()
                solution.destination_total_demands[dest_idx] += locker_demand
                v_idx = solution.destination_vehicle_assignmests[dest_idx]
                if v_idx != -1:
                    solution.vehicle_loads[v_idx] += locker_demand
                solution.check_validity()
# random
# class RandomOrderBestReassignment(BestFirstFitReassignmentOperator):
#     def apply(self, problem, solution):
#         custs_to_reassign_idx = [customer.idx for customer in problem.customers if (customer.is_flexible or customer.is_self_pickup) and solution.package_destinations[customer.idx]==-1]
#         shuffle(custs_to_reassign_idx)
#         for cust_idx in custs_to_reassign_idx:
#             self.ffr(problem, solution, cust_idx)
#         self.set_unserved_lockers_mrt_usage(problem, solution)

# worst customer first
class BestCustomerBestFirstFitReassignment(BestFirstFitReassignmentOperator):
    
    def apply(self, problem, solution):
        custs_to_reassign_idx = [customer.idx for customer in problem.customers if (customer.is_flexible or customer.is_self_pickup) and solution.package_destinations[customer.idx]==-1]
        self.custs_to_reassign_idx = set(custs_to_reassign_idx)
        custs_to_reassign_idx = np.asanyarray(custs_to_reassign_idx, dtype=int)
        reassignment_tasks: List[ReassignmentTask] = []
        for cust_idx in custs_to_reassign_idx:
            cust_assignment_d_costs = self.compute_assignment_d_costs(problem, solution, cust_idx)
            cust_alternative_dests_idx = problem.destination_alternatives[cust_idx]
            assert not np.all(np.isinf(cust_assignment_d_costs))
            for i, dest_idx in enumerate(cust_alternative_dests_idx):
                d_cost = cust_assignment_d_costs[i]
                if np.isinf(d_cost):
                    continue
                reassignment_tasks += [ReassignmentTask(cust_idx, dest_idx, d_cost)]
        sorted(reassignment_tasks, key= lambda r_task: r_task.d_cost)
        is_feasible_reassignment_found = self.ffr(problem, solution, reassignment_tasks, 0)
        # exit()
        # self.set_unserved_lockers_mrt_usage(problem, solution)
        # assert is_feasible_reassignment_found == True