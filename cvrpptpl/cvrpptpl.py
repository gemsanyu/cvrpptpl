import os
import pathlib
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import distance_matrix as dm_func

from cvrpptpl.customer import Customer
from cvrpptpl.locker import Locker
from cvrpptpl.mrt_line import MrtLine
from cvrpptpl.vehicle import Vehicle

class Cvrpptpl:
    def __init__(self,
                 depot_coord: np.ndarray,
                 customers: List[Customer],
                 lockers: List[Locker],
                 mrt_lines : List[MrtLine],
                 vehicles: List[Vehicle],
                 depot_location_mode: str,
                 locker_capacity_ratio: float,
                 locker_location_mode: str,
                 pickup_ratio: float,
                 flexible_ratio: float,
                 freight_capacity_mode: str,
                 distance_matrix: Optional[np.ndarray] = None
                 ) -> None:
        self.depot_coord = depot_coord
        self.customers = customers
        self.lockers = lockers
        self.mrt_lines = mrt_lines
        self.vehicles = vehicles
        self.depot_location_mode = depot_location_mode
        self.locker_capacity_ratio = locker_capacity_ratio
        self.locker_location_mode = locker_location_mode
        self.pickup_ratio = pickup_ratio
        self.flexible_ratio = flexible_ratio
        self.freight_capacity_mode = freight_capacity_mode
        self.num_customers = len(customers)
        self.num_lockers = len(lockers)
        self.num_vehicles = len(vehicles)
        self.num_nodes = 1 + self.num_customers + self.num_lockers        
        coords = [self.depot_coord]
        coords += [customer.coord for customer in customers]
        coords += [locker.coord for locker in lockers]
        self.coords = np.stack(coords, axis=0)
        self.distance_matrix = dm_func(self.coords, self.coords) if distance_matrix is None else distance_matrix
        
    def save_to_file(self):
        instance_dir = pathlib.Path(".")/"instances"
        filename = "nn_"+ str(self.num_nodes)
        filename += "_dlm_"+ str(self.depot_location_mode)
        filename += "_lcr_"+ str(self.locker_capacity_ratio)
        filename += "_llm_"+ str(self.locker_location_mode)
        filename += "_pr_"+ str(self.pickup_ratio)
        filename += "_fr_"+ str(self.flexible_ratio)
        filename += "_fcm_"+ str(self.freight_capacity_mode)
        filename += "_nc_"+ str(self.num_customers)
        filename += "_nl_"+ str(self.num_lockers)
        filename += "_nv_"+ str(self.num_vehicles)
        filepath = None
        for save_idx in range(100000):
            filepath = instance_dir/(filename+"_idx_"+str(save_idx)+".txt")
            if not os.path.exists(filepath.absolute()):
                break
        instance_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        lines += ["vehicles\n"]
        lines += ["vehicle_idx, capacity, cost_per_unit_distance\n"]
        for vehicle in self.vehicles:
            lines += [str(vehicle)]
        lines += ["depot\n"]
        lines += ["node_idx,x,y\n"]
        lines += ["0,"+str(self.depot_coord[0])+","+str(self.depot_coord[1])+"\n"]
        lines += ["home delivery customers\n"]
        lines += ["node_idx,x,y,demand\n"]
        for customer in self.customers:
            if not (customer.is_self_pickup or customer.is_flexible):
                lines += [str(customer)]
        lines += ["self pickup customers\n"]
        lines += ["node_idx,x,y,demand,locker_idxs\n"]
        for customer in self.customers:
            if customer.is_self_pickup:
                lines += [str(customer)]
        lines += ["flexible customers\n"]
        lines += ["node_idx,x,y,demand,locker_idxs\n"]
        for customer in self.customers:
            if customer.is_flexible:
                lines += [str(customer)]
        lines += ["lockers\n"]
        lines += ["node_idx,x,y,capacity,cost_per_unit_good\n"]
        for locker in self.lockers:
            lines += [str(locker)]
        lines += ["mrt lines\n"]
        lines += ["start_node_idx,end_node_idx,freight_capacity,cost_per_unit_good\n"]
        for mrt_line in self.mrt_lines:
            lines += [str(mrt_line)]
        lines += ["distance matrix\n"]
        node_idxs_str = ","+"".join([str(i)+"," for i in range(self.num_nodes)])+"\n"
        lines += node_idxs_str
        for i in range(self.num_nodes):
            line = str(i)+","
            for j in range(self.num_nodes):
                line += str(self.distance_matrix[i,j])
                if j < self.num_nodes-1:
                    line += ","
            lines += [line+"\n"]
        
        with open(filepath.absolute(), "w") as save_file:
            save_file.writelines(lines)

def read_from_file(filename:str)->Cvrpptpl:
    dir = pathlib.Path("")/"instances"
    filepath = dir/filename
    vehicles:List[Vehicle] = []
    customers:List[Customer] = []
    lockers:List[Locker] = []
    mrt_lines:List[MrtLine] = []
    locker_idx_dict: Dict[int, Locker] = {}
    with open(filepath.absolute(), "r") as f:
        lines = f.readlines()
        line_idx = 2
        # 1 vehicles
        while "depot" not in lines[line_idx]:
            line = lines[line_idx]
            line = line.split(",")
            idx, capacity, cost = int(line[0]), int(line[1]), int(line[2])
            vehicle = Vehicle(idx, capacity, cost)
            vehicles += [vehicle]
            line_idx += 1
        line_idx += 2
        line = lines[line_idx].split(",")
        # 2 depot coord
        depot_coord = np.asanyarray([int(line[1]), int(line[2])], dtype=int)
        line_idx += 3
        # 3 hd customers
        while "self pickup" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, demand = [int(v) for v in line]
           coord = np.asanyarray([x,y], dtype=int)
           hd_customer = Customer(idx, coord, demand)
           customers += [hd_customer]
           line_idx += 1
        line_idx += 2
        # 4 self pickup
        while "flexible" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, demand, locker_idxs_str = line
           idx, x, y, demand = int(idx), int(x), int(y), int(demand)
           locker_idxs_str = locker_idxs_str.split("-")
           locker_idxs = [int(locker_idx_str) for locker_idx_str in locker_idxs_str]
           coord = np.asanyarray([x,y], dtype=int)
           sp_customer = Customer(idx, coord, demand, is_self_pickup=True, preferred_locker_idxs=locker_idxs)
           customers += [sp_customer]
           line_idx += 1
        line_idx += 2
        # 5 flexible
        while "lockers" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, demand, locker_idxs_str = line
           idx, x, y, demand = int(idx), int(x), int(y), int(demand)
           locker_idxs_str = locker_idxs_str.split("-")
           locker_idxs = [int(locker_idx_str) for locker_idx_str in locker_idxs_str]
           coord = np.asanyarray([x,y], dtype=int)
           fx_customer = Customer(idx, coord, demand, is_flexible=True, preferred_locker_idxs=locker_idxs)
           customers += [fx_customer]
           line_idx += 1
        line_idx += 2
        # 6 lockers
        while "mrt" not in lines[line_idx]:
            line = lines[line_idx].split(",")
            idx,x,y,capacity,cost = [int(v) for v in line]
            coord = np.asanyarray([x,y], dtype=int)
            locker = Locker(idx, coord, capacity, cost)
            locker_idx_dict[idx] = locker
            lockers += [locker]
            line_idx += 1
        line_idx += 2
        # 7 mrt lines
        while "distance" not in lines[line_idx]:
            line = lines[line_idx].split(",")
            start_idx, end_idx, capacity, cost = [int(v) for v in line]
            start_station, end_station = locker_idx_dict[start_idx], locker_idx_dict[end_idx]
            mrt_line = MrtLine(start_station, end_station, cost, capacity)
            mrt_lines += [mrt_line]
            line_idx += 1
        line_idx += 2
        # 8 precomputed distance matrix
        num_nodes = len(customers) + len(lockers) + 1
        distance_matrix = np.zeros([num_nodes, num_nodes], dtype=float)
        for i in range(num_nodes):
            line = lines[line_idx].split(",")
            line = [float(v) for v in line]
            idx, distances = line[0], np.asanyarray(line[1:])
            distance_matrix[i, :] = distances
            line_idx += 1
    
    # let's parse the filename if parseable
    # depot_location_mode = "unknown"
    # locker_capacity_ratio = -1
    # locker_location_mode = "unknown"
    # pickup_ratio = -1
    # flexible_ratio = -1
    # freight_capacity_mode = "unknown"

    dlm_pos = filename.find("dlm_")
    lcr_pos = filename.find("lcr_")
    llm_pos = filename.find("llm_")
    pr_pos = filename.find("pr_")
    fr_pos = filename.find("fr_")
    fcm_pos = filename.find("fcm_")
    nc_pos = filename.find("nc_")
    
    depot_location_mode = filename[dlm_pos+4:lcr_pos-1]
    locker_capacity_ratio = float(filename[lcr_pos+4:llm_pos-1])
    locker_location_mode =  filename[llm_pos+4:pr_pos-1]
    pickup_ratio = float(filename[pr_pos+3:fr_pos-1])
    flexible_ratio = float(filename[fr_pos+3:fcm_pos-1])
    freight_capacity_mode = filename[fcm_pos+4:nc_pos-1]
    problem = Cvrpptpl(depot_coord,
                       customers,
                       lockers,
                       mrt_lines,
                       vehicles,
                       depot_location_mode,
                       locker_capacity_ratio,
                       locker_location_mode,
                       pickup_ratio,
                       flexible_ratio,
                       freight_capacity_mode,
                       distance_matrix)
    return problem
    
    
    