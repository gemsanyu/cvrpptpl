from copy import deepcopy
from typing import Self, List
import math

import numpy as np

from problem.cvrpptpl import Cvrpptpl
from problem.node import Node
from problem.customer import Customer
from problem.locker import Locker
from problem.mrt_line import MrtLine

NO_VEHICLE = 99999
NO_DESTINATION = 99999

class Solution:
    """_summary_
    """
    def __init__(self, 
                 problem: Cvrpptpl,
                 package_destinations: np.ndarray = None,
                 mrt_usage_masks: np.ndarray = None,
                 mrt_loads: np.ndarray = None,
                 destination_vehicle_assignmests: np.ndarray = None,
                 destination_total_demands: np.ndarray = None,
                 routes: List[List[int]] = None,
                 vehicle_loads: np.ndarray = None,
                 locker_loads: np.ndarray = None,
                 total_locker_charge: float = None,
                 total_vehicle_charge: float = None,
                 total_mrt_charge: float = None,
                 ):
        self.problem = problem
        num_nodes = problem.num_nodes
        num_mrt_lines = len(problem.mrt_lines)
        num_vehicles = problem.num_vehicles
        
        # some necessary informations (similar across solutions)
        self.destination_alternatives: List[List[int]] = problem.destination_alternatives
        self.locker_capacities: np.ndarray = problem.locker_capacities
        self.locker_costs: np.ndarray = problem.locker_costs
        self.demands: np.ndarray = problem.demands
        self.vehicle_capacities: np.ndarray = problem.vehicle_capacities
        self.vehicle_costs: np.ndarray = problem.vehicle_costs
        self.mrt_line_costs: np.ndarray = problem.mrt_line_costs
        self.mrt_line_capacities: np.ndarray = problem.mrt_line_capacities
        self.mrt_line_stations_idx: np.ndarray = problem.mrt_line_stations_idx
        self.incoming_mrt_lines_idx: List[int] = problem.incoming_mrt_lines_idx
        self.distance_matrix: np.ndarray = problem.distance_matrix
        
        # self.service_times: np.ndarray = problem.service_times
        
        # initialize decision variables representations
        # simply copy if given
        if package_destinations is not None:
            self.package_destinations = np.copy(package_destinations)
        else:
            self.package_destinations: np.ndarray = np.full([num_nodes,], NO_DESTINATION, dtype=int)
        if mrt_usage_masks is not None:
            self.mrt_usage_masks = np.copy(mrt_usage_masks)
        else:
            self.mrt_usage_masks: np.ndarray = np.zeros([num_mrt_lines,], dtype=bool)
        if destination_vehicle_assignmests is not None:
            self.destination_vehicle_assignmests = np.copy(destination_vehicle_assignmests)
        else:
            self.destination_vehicle_assignmests: np.ndarray = np.full([num_nodes,], NO_VEHICLE, dtype=int)
        if routes is not None:
            self.routes: List[List[int]] = deepcopy(routes)
        else:
            self.routes: List[List[int]] = [[0] for _ in range(num_vehicles)]
    
        # memory of costs and loads
        if destination_total_demands is not None:
            self.destination_total_demands = np.copy(destination_total_demands)
        else:
            self.destination_total_demands: np.ndarray = np.zeros([num_nodes,], dtype=int)
        if vehicle_loads is not None:
            self.vehicle_loads = np.copy(vehicle_loads)
        else:
            self.vehicle_loads: np.ndarray = np.zeros([num_vehicles,], dtype=int)
        if locker_loads is not None:
            self.locker_loads = np.copy(locker_loads)
        else:
            self.locker_loads = np.zeros([num_nodes,], dtype=int)
        if mrt_loads is not None:
            self.mrt_loads = np.copy(mrt_loads)
        else:
            self.mrt_loads: np.ndarray = np.zeros([num_mrt_lines,], dtype=int)
       
        self.total_locker_charge: float = total_locker_charge or 0.
        self.total_vehicle_charge: float = total_vehicle_charge or 0.
        self.total_mrt_charge: float = total_mrt_charge or 0
        
    @property
    def total_cost(self):
        return self.total_locker_charge + self.total_vehicle_charge + self.total_mrt_charge
    
    @property
    def total_vehicle_charge_recalculated(self):
        total_vehicle_charge = 0
        for v_idx, route in enumerate(self.routes):
            route_ = route[1:] + [0]
            total_vehicle_charge += np.sum(self.problem.distance_matrix[route, route_]) * self.vehicle_costs[v_idx]
        return total_vehicle_charge
    
    def copy(self)->Self:
        new_copy: Self = self.__class__(self.problem, 
                                        self.package_destinations,
                                        self.mrt_usage_masks,
                                        self.mrt_loads,
                                        self.destination_vehicle_assignmests,
                                        self.destination_total_demands,
                                        self.routes,
                                        self.vehicle_loads,
                                        self.locker_loads,
                                        self.total_locker_charge,
                                        self.total_vehicle_charge,
                                        self.total_mrt_charge)
        return new_copy
    
    # this is not feasibility -> not all customers need to be visited,
    # but validity -> which ultimately leads to if the computed cost really reflect solution
    def check_validity(self):
        problem = self.problem
        
        # check if a customer is assigned to a locker
        # then this customer location' total demand should be 0
        # because all its demands are served somewhere else (in a locker)
        for customer in problem.customers:
            c_idx = customer.idx
            dest_idx = self.package_destinations[c_idx]
            if c_idx != dest_idx:
                assert self.destination_total_demands[c_idx]==0
        
        # check if a customer is assigned to locker
        # this customer should not be visited
        for v_idx in range(problem.num_vehicles):
            for node_idx in self.routes[v_idx]:
                if node_idx <=problem.num_customers and node_idx>0:
                    dest_idx = self.package_destinations[node_idx]
                    # print(node_idx, self.destination_total_demands[node_idx])
                    assert dest_idx == node_idx
        
        # check if a customer is not assigned to any destination,
        # then it should not exists in any route
        for v_idx in range(problem.num_vehicles):
            for node_idx in self.routes[v_idx]:
                if node_idx <=problem.num_customers and node_idx>0: #customer
                    dest_idx = self.package_destinations[node_idx]
                    # print(v_idx, node_idx, dest_idx)
                    assert dest_idx != NO_DESTINATION
        
        # check for locker load
        for locker in problem.lockers:
            locker_load = self.locker_loads[locker.idx]
            actual_load = 0
            for customer in problem.customers:
                if self.package_destinations[customer.idx] == locker.idx:
                    actual_load += customer.demand
            # print(locker.idx, locker_load, actual_load)
            assert locker_load == actual_load
            
        # let's check mrt loads
        for i, mrt_line in enumerate(problem.mrt_lines):
            mrt_line_load = self.mrt_loads[i]
            if not self.mrt_usage_masks[i]:
                assert mrt_line_load == 0
                continue
            actual_load = 0
            for locker in problem.lockers:
                locker_idx = locker.idx
                incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[locker_idx]
                if incoming_mrt_line_idx is None or incoming_mrt_line_idx!=i:
                    continue
                # print("---", locker_idx, self.locker_loads[locker_idx], problem.incoming_mrt_lines_idx[locker_idx], i)
                actual_load += self.locker_loads[locker_idx]
            assert mrt_line_load == actual_load

        # check for vehicle load is the same as its actual load
        # and if its not exceeding
        for v_idx in range(problem.num_vehicles):
            vehicle_load = self.vehicle_loads[v_idx]
            actual_load = 0
            for node_idx in self.routes[v_idx]:
                actual_load += self.destination_total_demands[node_idx]
            
            # print(self.routes[v_idx])
            # print(vehicle_load, actual_load)
            assert vehicle_load == actual_load
            assert vehicle_load <= problem.vehicle_capacities[v_idx]
            
        for node_idx in range(self.problem.num_nodes):
            assert self.destination_total_demands[node_idx] >= 0

        
        
            
    def check_feasibility(self):
        problem = self.problem
        self.check_validity()
        for customer in problem.customers:
            assert self.package_destinations[customer.idx] != NO_DESTINATION
            if customer.idx == self.package_destinations[customer.idx]:
                assert self.destination_vehicle_assignmests[customer.idx] != NO_VEHICLE

        total_locker_charge = 0
        for customer in problem.customers:
            dest_idx = self.package_destinations[customer.idx]
            if dest_idx != customer.idx:
                total_locker_charge += self.locker_costs[dest_idx]
        assert total_locker_charge == self.total_locker_charge
        
        # print(self.total_vehicle_charge, self.total_vehicle_charge_recalculated)
        assert math.isclose(self.total_vehicle_charge_recalculated, self.total_vehicle_charge, abs_tol=1e-9)
        