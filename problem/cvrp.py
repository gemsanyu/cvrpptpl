import os
import pathlib
import re
from typing import Dict, List, Optional

import numpy as np
from problem.customer import Customer
from problem.node import Node
from problem.vehicle import Vehicle
from scipy.spatial import distance_matrix as dm_func


class Cvrp:
    def __init__(self,
                 depot: Node,
                 customers: List[Customer],
                 vehicles: List[Vehicle],
                 distance_matrix: Optional[np.ndarray] = None
                 ) -> None:
        self.depot = depot
        self.depot_coord = depot.coord
        self.customers = customers
        self.nodes: List[Node] = [depot] + customers
        self.vehicles = vehicles
        self.num_customers = len(customers)
        self.num_vehicles = len(vehicles)
        self.num_nodes = 1 + self.num_customers
        coords = [node.coord for node in self.nodes]
        self.coords = np.stack(coords, axis=0)
        self.demands = np.asanyarray([customer.demand for customer in customers])
        self.distance_matrix = distance_matrix
        if distance_matrix is None:
            self.distance_matrix = dm_func(self.coords, self.coords)
            self.distance_matrix = np.around(self.distance_matrix, decimals=2)
        
def read_from_file(filename:str)->Cvrp:
    """_summary_

    Args:
        filename (str): filename

    Returns:
        Cvrp: an instance of cvrp
    """
    directory = pathlib.Path("")/"instances"/"cvrp"
    filepath = directory/filename
    vehicle_capacity: int
    num_vehicles: int
    coords: List[List[int]] = []
    demands: List[int] = []
    with open(filepath.absolute(), "r") as file:
        reading_coords = False
        reading_demands = False
        for line in file:
            match = re.search(r"No of trucks: (\d+)", line)
            if match:
                num_vehicles = int(match.group(1))
            if line.startswith("CAPACITY"):
                parts = line.split()
                vehicle_capacity = int(parts[2])
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            if line.startswith("DEMAND_SECTION"):
                reading_coords = False
                reading_demands = True
                continue
            if line.startswith("DEPOT_SECTION"):
                break  # No need to read depot section
            if reading_coords:
                parts = line.split()
                coords.append([int(parts[1]), int(parts[2])])
            elif reading_demands:
                parts = line.split()
                demands.append(int(parts[1]))
    
    depot_coord = coords[0]
    depot = Node(0, depot_coord)
    coords = coords[1:]
    demands = demands[1:]
    customers: List[Customer] = [Customer(i, np.asanyarray(coord),15, demands[i], False, False, []) for i, coord in enumerate(coords)]
    vehicles: List[Vehicle] = [Vehicle(i, vehicle_capacity, 1) for i in range(num_vehicles)]
    
        # return lines
    problem = Cvrp(depot, customers, vehicles)
    return problem
    
    
    