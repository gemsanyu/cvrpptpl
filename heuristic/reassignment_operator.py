from dataclasses import dataclass
from enum import Enum
from random import random, shuffle
from typing import List, Set

import numpy as np

from heuristic.reinsertion_operator import BestPositionReinsertionOperator
from heuristic.operator import OperationStatus
from heuristic.solution import Solution, NO_VEHICLE, NO_DESTINATION
from problem.cvrpptpl import Cvrpptpl


@dataclass(frozen=True)
class ReassignmentTask:
    cust_idx: int
    dest_idx: int
    d_cost: float
    
class FirstFitReassignmentOperator(BestPositionReinsertionOperator):
    def __init__(self, problem):
        super().__init__(problem)
        self.reassigned_custs_idx: Set[int] = set()
        self.custs_to_reassign_idx: Set[int] = set()
    
    def compute_assignment_d_costs(self, problem: Cvrpptpl, solution: Solution, cust_idx: int):
        alternative_dests_idx = problem.destination_alternatives[cust_idx]
        assignment_d_costs = np.full([len(alternative_dests_idx)], np.inf, dtype=float)
        demand = problem.demands[cust_idx]
        # print(cust_idx, alternative_dests_idx)
        for i, dest_idx in enumerate(alternative_dests_idx):
            d_cost = 0
            if not self.is_r_task_applicable(problem, solution, cust_idx=cust_idx, dest_idx=dest_idx):
                # print("HM?")
                continue
            
            # if home delivery then just compute the distance_cost
            # or if locker, and locker not in route yet
            # then compute best potential distance cost
            # TODO: actually if locker and unserved by any vehicle, 
            # also consider the distance cost if MRT is used,, should we average this? 
            incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
            using_mrt = incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]
            if dest_idx == cust_idx or (not using_mrt and solution.destination_vehicle_assignmests[dest_idx]==NO_VEHICLE):
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
            
            # if using MRT, but the start station is not in route yet?
            if using_mrt:
                start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
                v_idx = solution.destination_vehicle_assignmests[start_station_idx]
                if v_idx == NO_VEHICLE:
                    best_d_cost, best_pos, best_v_idx = self.find_best_insertion_pos(
                        solution.routes,
                        start_station_idx,
                        demand,
                        solution.vehicle_loads,
                        problem.vehicle_capacities,
                        problem.vehicle_costs,
                        problem.distance_matrix                            
                    )
                    d_cost += best_d_cost
            
            if dest_idx != cust_idx:
                locker_cost = problem.locker_costs[dest_idx]
                d_cost += locker_cost
                if using_mrt:
                    mrt_cost = solution.mrt_line_costs[incoming_mrt_line_idx]*demand
                    d_cost += mrt_cost
            assignment_d_costs[i] = d_cost
        return assignment_d_costs
    
    def best_first_fit_reassignment(self, problem: Cvrpptpl, solution: Solution, reassignment_tasks: List[ReassignmentTask]):
        self.reassigned_custs_idx.clear()
        return self.ffr(problem, solution, reassignment_tasks, 0)
    
    def is_r_task_applicable(self, problem: Cvrpptpl, solution: Solution, r_task: ReassignmentTask=None, cust_idx:int = None, dest_idx:int = None)->bool:
        if r_task is not None:
            cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        if cust_idx == dest_idx: # home delivery,, of course applicable
            return True
        demand = problem.demands[cust_idx]
        # print(cust_idx, dest_idx, demand)
        # then this must be a locker
        if solution.locker_loads[dest_idx] + demand > problem.locker_capacities[dest_idx]:
            return False
        # if using mrt we need to check if the vehicle's capacity serving the start station
        # if its not using mrt, then check if the locker visited in a route
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
        using_mrt = incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]
        # print("HELO", using_mrt)
        if using_mrt: 
            # print(solution.mrt_loads[incoming_mrt_line_idx], problem.mrt_line_capacities[incoming_mrt_line_idx])
            if solution.mrt_loads[incoming_mrt_line_idx] + demand > problem.mrt_line_capacities[incoming_mrt_line_idx]:
                return False
            start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
            v_idx = solution.destination_vehicle_assignmests[start_station_idx]
            if v_idx!= NO_VEHICLE and solution.vehicle_loads[v_idx]+demand>problem.vehicle_capacities[v_idx]:
                # print(solution.vehicle_loads[v_idx], problem.vehicle_capacities[v_idx])
                return False
            return True
        # print("HOLA")
        # what if this locker is in a route? is the vehicle load ok?
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        if v_idx != NO_VEHICLE:
            if solution.vehicle_loads[v_idx] + demand > problem.vehicle_capacities[v_idx]:
                # print(solution.vehicle_loads[v_idx], problem.vehicle_capacities[v_idx])
                return False
            return True
        
        return True
    
    def apply_r_task(self, problem: Cvrpptpl, solution: Solution, r_task: ReassignmentTask):
        cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        demand = problem.demands[cust_idx]
        if cust_idx == dest_idx:
            solution.package_destinations[cust_idx] = dest_idx
            solution.destination_total_demands[cust_idx] = demand
            v_idx = solution.destination_vehicle_assignmests[dest_idx]
            # if v_idx!=NO_VEHICLE:
            #     print("A", solution.routes[v_idx], cust_idx)
            assert v_idx==NO_VEHICLE
                # solution.vehicle_loads[v_idx]-=demand
            return
        solution.locker_loads[dest_idx] += demand
        solution.package_destinations[cust_idx] = dest_idx
        solution.total_locker_charge += problem.locker_costs[dest_idx]
        
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
        using_mrt = incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]
        if using_mrt:
            solution.mrt_loads[incoming_mrt_line_idx] += demand
            solution.total_mrt_charge += problem.mrt_line_costs[incoming_mrt_line_idx]*demand
            start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
            v_idx = solution.destination_vehicle_assignmests[start_station_idx]
            solution.destination_total_demands[start_station_idx] += demand
            if v_idx!=NO_VEHICLE:
                solution.vehicle_loads[v_idx] += demand
            return
        
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        solution.destination_total_demands[dest_idx]+=demand
        if v_idx!=NO_VEHICLE:
            solution.vehicle_loads[v_idx]+=demand
            return
    
    def revert_r_task(self, problem: Cvrpptpl, solution: Solution, r_task: ReassignmentTask):
        cust_idx, dest_idx = r_task.cust_idx, r_task.dest_idx
        demand = problem.demands[cust_idx]
        if cust_idx == dest_idx:
            solution.package_destinations[cust_idx] = NO_DESTINATION
            solution.destination_total_demands[cust_idx] = 0
            return
        
        solution.locker_loads[dest_idx] -= demand
        solution.package_destinations[cust_idx] = NO_DESTINATION
        solution.total_locker_charge -= problem.locker_costs[dest_idx]
        
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[dest_idx]
        using_mrt = incoming_mrt_line_idx is not None and solution.mrt_usage_masks[incoming_mrt_line_idx]
        
        if using_mrt:
            solution.mrt_loads[incoming_mrt_line_idx] -= demand
            solution.total_mrt_charge -= problem.mrt_line_costs[incoming_mrt_line_idx]*demand
            start_station_idx = problem.mrt_lines[incoming_mrt_line_idx].start_station.idx
            v_idx = solution.destination_vehicle_assignmests[start_station_idx]
            solution.destination_total_demands[start_station_idx] -= demand
            if v_idx != NO_VEHICLE:
                solution.vehicle_loads[v_idx] -= demand
            return
        
        v_idx = solution.destination_vehicle_assignmests[dest_idx]
        solution.destination_total_demands[dest_idx]-=demand
        if v_idx!=NO_VEHICLE:
            solution.vehicle_loads[v_idx]-=demand
            return
        
        
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
    
    def randomize_lockers_mrt_usage(self, problem:Cvrpptpl, solution: Solution):
        # clean up, some mrt is active, but 0 mrt load.
        for mrt_line_idx, mrt_line in enumerate(problem.mrt_lines):
            if solution.mrt_loads[mrt_line_idx] == 0:
                solution.mrt_usage_masks[mrt_line_idx] = False
        
        # randomize usage
        for mrt_line_idx, mrt_line in enumerate(problem.mrt_lines):
            if solution.mrt_usage_masks[mrt_line_idx]:
                continue
            # mrt line is not used, but the end station is alread visited
            # then we alread decided to not use it
            start_station_idx, end_station_idx = mrt_line.start_station.idx, mrt_line.end_station.idx
            if solution.destination_vehicle_assignmests[end_station_idx] != NO_VEHICLE:
                continue
            
            # then, we can either use or not use the mrt
            if random()>0.5:
                continue
            locker_load = solution.locker_loads[end_station_idx]
            if solution.mrt_loads[mrt_line_idx] + locker_load > problem.mrt_line_capacities[mrt_line_idx]:
                continue
            # is the start station visited?
            v_idx = solution.destination_vehicle_assignmests[start_station_idx]
            if v_idx != NO_VEHICLE:
                if solution.vehicle_loads[v_idx] + locker_load > problem.vehicle_capacities[v_idx]:
                    continue
                solution.vehicle_loads[v_idx] += locker_load
            
                
            solution.mrt_usage_masks[mrt_line_idx] = True
            solution.mrt_loads[mrt_line_idx] += locker_load
            # transfer all demands of end_station's locker to the start_station node (not locker, node/destination)
            # (this need to be careful, 
            # end station might have >0 demands, because it is also a start station of another mrt line)
            solution.destination_total_demands[end_station_idx] -= locker_load
            solution.destination_total_demands[start_station_idx] += locker_load
            solution.total_mrt_charge += mrt_line.cost * locker_load
            
        # any unnecessary mrt line we need to switch off
        for mrt_line_idx, mrt_line in enumerate(problem.mrt_lines):
            if not solution.mrt_usage_masks[mrt_line_idx]:
                continue
            if solution.locker_loads[mrt_line.end_station.idx] == 0:
                solution.mrt_usage_masks[mrt_line_idx] = False
        
            
# random
class RandomFirstFitReassignment(FirstFitReassignmentOperator):
    def apply(self, problem, solution):
        custs_to_reassign_idx = [customer.idx for customer in problem.customers if (customer.is_flexible or customer.is_self_pickup) and solution.package_destinations[customer.idx]==NO_DESTINATION]
        self.reassigned_custs_idx.clear()
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
        shuffle(reassignment_tasks)
        is_feasible_reassignment_found = self.ffr(problem, solution, reassignment_tasks, 0)
        assert is_feasible_reassignment_found
        # self.randomize_lockers_mrt_usage(problem, solution)
        return OperationStatus.SUCCESS

    
    def __repr__(self):
        return "random-first-fit-reassignment"
        
class BestFirstFitReassignment(FirstFitReassignmentOperator):
    def apply(self, problem, solution):
        custs_to_reassign_idx = [customer.idx for customer in problem.customers if (customer.is_flexible or customer.is_self_pickup) and solution.package_destinations[customer.idx]==NO_DESTINATION]
        self.reassigned_custs_idx.clear()
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
        assert is_feasible_reassignment_found
        self.randomize_lockers_mrt_usage(problem, solution)
        return OperationStatus.SUCCESS
        # assert is_feasible_reassignment_found == True
    
    def __repr__(self):
        return "best-first-fit-reassignment"