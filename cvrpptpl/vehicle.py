
import math
from typing import List

import numpy as np

class Vehicle:
    def __init__(self,
                 idx: int,
                 capacity: int,
                 cost: float):
        self.idx = idx
        self.capacity = capacity
        self.cost = cost
    
    def __str__(self) -> str:
        return str(self.idx)+","+str(self.capacity)+","+str(self.cost)+"\n"
    
def generate_vehicles(num_vehicles: int,
                      num_customers: int,
                      total_customer_demand: int,
                      cost_reference: float,
                      desired_route_length=10)->List[Vehicle]:
    num_vehicle_reference = 5
    capacity_reference = math.ceil(desired_route_length*total_customer_demand/num_customers)
    capacities_ref = (0.4 + 0.2*np.arange(num_vehicle_reference, dtype=float))*capacity_reference
    costs_ref = (0.7+0.1*np.arange(num_vehicle_reference, dtype=float))*cost_reference
    vehicle_idxs = np.arange(num_vehicles, dtype=int) % num_vehicle_reference
    capacities = capacities_ref[vehicle_idxs]
    costs = costs_ref[vehicle_idxs]
    
    vehicles = [Vehicle(i, capacities[i], costs[i]) for i in range(num_vehicles)]
    return vehicles