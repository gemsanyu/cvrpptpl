from copy import deepcopy
from typing import List, Self

import numpy as np

from problem.cvrpptpl import Cvrpptpl
from problem.node import Node
from problem.customer import Customer
from problem.locker import Locker
from problem.mrt_line import MrtLine


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
        
        # self.service_times: np.ndarray = problem.service_times
        
        # initialize decision variables representations
        # simply copy if given
        if package_destinations is not None:
            self.package_destinations = np.copy(package_destinations)
        else:
            self.package_destinations: np.ndarray = np.full([num_nodes,], -1, dtype=int)
        if mrt_usage_masks is not None:
            self.mrt_usage_masks = np.copy(mrt_usage_masks)
        else:
            self.mrt_usage_masks: np.ndarray = np.zeros([num_mrt_lines,], dtype=bool)
        if destination_vehicle_assignmests is not None:
            self.destination_vehicle_assignmests = np.copy(destination_vehicle_assignmests)
        else:        
            self.destination_vehicle_assignmests: np.ndarray = np.full([num_nodes,], -1, dtype=int)
        if routes is not None:
            self.routes = deepcopy(routes)
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
    
    def copy(self)->Self:
        new_copy: Self = self.__class__(self.problem, 
                                        self.package_destinations,
                                        self.mrt_usage_masks,
                                        self.mrt_loads,
                                        self.destination_vehicle_assignmests,
                                        self.destination_total_demands,
                                        self.routes,
                                        self.vehicle_loads,
                                        self.total_locker_charge,
                                        self.total_vehicle_charge,
                                        self.total_mrt_charge)
        return new_copy