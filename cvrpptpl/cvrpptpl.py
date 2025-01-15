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
                 customer_location_mode: str,
                 distance_matrix: Optional[np.ndarray] = None
                 ) -> None:
        self.depot_coord = depot_coord
        self.customers = customers
        self.lockers = lockers
        self.mrt_lines = mrt_lines
        mrt_lockers_idx = [mrt_line.start_station.idx for mrt_line in mrt_lines] + [mrt_line.end_station.idx for mrt_line in mrt_lines]
        self.non_mrt_lockers = [locker for locker in lockers if not locker.idx in mrt_lockers_idx]
        self.vehicles = vehicles
        self.depot_location_mode = depot_location_mode
        self.locker_capacity_ratio = locker_capacity_ratio
        self.locker_location_mode = locker_location_mode
        self.pickup_ratio = pickup_ratio
        self.flexible_ratio = flexible_ratio
        self.freight_capacity_mode = freight_capacity_mode
        self.customer_location_mode = customer_location_mode
        self.num_customers = len(customers)
        self.num_lockers = len(lockers)
        self.num_vehicles = len(vehicles)
        self.num_nodes = 1 + self.num_customers + self.num_lockers        
        coords = [self.depot_coord]
        coords += [customer.coord for customer in customers]
        coords += [locker.coord for locker in lockers]
        self.coords = np.stack(coords, axis=0)
        self.distance_matrix = distance_matrix
        if distance_matrix is None:
            self.distance_matrix = dm_func(self.coords, self.coords)
            self.distance_matrix = np.around(self.distance_matrix, decimals=2)
        self.filename = self.init_filename()
        
    def init_filename(self):
        instance_dir = pathlib.Path(".")/"instances"
        filename = "nn_"+ str(self.num_nodes)
        filename += "_clm_"+ str(self.customer_location_mode)
        filename += "_dlm_"+ str(self.depot_location_mode)
        filename += "_lcr_"+ str(self.locker_capacity_ratio)
        filename += "_llm_"+ str(self.locker_location_mode)
        filename += "_pr_"+ str(self.pickup_ratio)
        filename += "_fr_"+ str(self.flexible_ratio)
        filename += "_fcm_"+ str(self.freight_capacity_mode)
        filename += "_nc_"+ str(self.num_customers)
        filename += "_nl_"+ str(self.num_lockers)
        filename += "_nv_"+ str(self.num_vehicles)
        final_filename = None
        txt_filepath = None
        ampl_filepath = None
        for save_idx in range(100000):
            final_filename = filename+"_idx_"+str(save_idx)
            txt_filepath = instance_dir/(final_filename+".txt")
            ampl_filepath = instance_dir/(final_filename+"_ampl.txt")
            if not os.path.exists(txt_filepath.absolute()) and not os.path.exists(ampl_filepath.absolute()):
                break
        return filename
        
    def save_to_file(self):
        instance_dir = pathlib.Path(".")/"instances"
        filepath = instance_dir/(self.filename+".txt")
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
        lines += ["node_idx,x,y,service_time,demand\n"]
        for customer in self.customers:
            if not (customer.is_self_pickup or customer.is_flexible):
                lines += [str(customer)]
        lines += ["self pickup customers\n"]
        lines += ["node_idx,x,y,service_time,demand,locker_idxs\n"]
        for customer in self.customers:
            if customer.is_self_pickup:
                lines += [str(customer)]
        lines += ["flexible customers\n"]
        lines += ["node_idx,x,y,service_time,demand,locker_idxs\n"]
        for customer in self.customers:
            if customer.is_flexible:
                lines += [str(customer)]
        lines += ["lockers\n"]
        lines += ["node_idx,x,y,service_time,capacity,cost_per_unit_good\n"]
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

    def save_to_ampl_file(self):
        instance_dir = pathlib.Path(".")/"instances"
        filepath = instance_dir/(self.filename+"ampl_.txt")
        instance_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        vehicles_idx_str = "\t".join([str(vehicle.idx) for vehicle in self.vehicles])
        lines += ["set K:= "+vehicles_idx_str+";\n"]
        hd_custs_idx = [customer.idx for customer in self.customers if not (customer.is_flexible or customer.is_self_pickup)]
        hd_custs_idx_str = "\t".join([str(c_idx) for c_idx in hd_custs_idx])
        lines += ["set C_H:= "+hd_custs_idx_str+";\n"]
        sp_custs_idx = [customer.idx for customer in self.customers if customer.is_self_pickup]
        sp_custs_idx_str = "\t".join([str(c_idx) for c_idx in sp_custs_idx])
        lines += ["set C_S:= "+sp_custs_idx_str+";\n"]
        f_custs_idx = [customer.idx for customer in self.customers if customer.is_flexible]
        f_custs_idx_str = "\t".join([str(c_idx) for c_idx in f_custs_idx])
        lines += ["set C_F:= "+f_custs_idx_str+";\n"]
        mrts_idx = [mrt_line.start_station.idx for mrt_line in self.mrt_lines] + [mrt_line.end_station.idx for mrt_line in self.mrt_lines]
        mrts_idx.sort()
        mrts_idx_str = "\t".join([str(mrt_idx) for mrt_idx in mrts_idx])
        lines += ["set M:= "+mrts_idx_str+";\n"]
        mrt_ts_idx = [mrt_line.start_station.idx for mrt_line in self.mrt_lines]
        mrt_ts_idx_str = "\t".join([str(ts_idx) for ts_idx in mrt_ts_idx])
        lines += ["set M_t:= "+mrt_ts_idx_str+";\n"]
        non_mrt_lockers_idx = [locker.idx for locker in self.non_mrt_lockers]
        non_mrt_lockers_idx_str = "\t".join([str(l_idx) for l_idx in non_mrt_lockers_idx])
        lines += ["set L_B:= "+non_mrt_lockers_idx_str+";\n"]
        lines += ["set A1:=\n"]
        
        reachable_nodes_idx = [0] + hd_custs_idx + f_custs_idx + mrt_ts_idx + non_mrt_lockers_idx
        for i in reachable_nodes_idx:
            reachable_nodes_idx_str = [str(idx) for idx in reachable_nodes_idx if idx!=i]    
            line = f"({i},*) "+" ".join(reachable_nodes_idx_str)+"\n"
            lines += [line]
        lines+=[";\n"]
        
        lines += ["set A2:=\n"]
        for mrt_line in self.mrt_lines:
            lines += [f"({mrt_line.start_station.idx},*) {mrt_line.end_station.idx}\n"]
        lines+=[";\n"]
        
        lines+= ["param BigM:=999;\n"] 
        lines+= ["param r:= 2;\n"]
        
        lines+= ["param d:=\n"]
        lines+= ["0\t0\n"]
        
        for customer in self.customers:
            lines+= [f"{customer.idx}\t{customer.demand}\n"]
        lines+= [";\n"]
        
        lines+= ["param Q:=\n"]
        for vehicle in self.vehicles:
            lines+= [f"{vehicle.idx}\t{vehicle.capacity}\n"]
        lines+= [";\n"]
        
        lines+= ["param w:=\n"]
        for mrt_line in self.mrt_lines:
            lines+= [f"[{mrt_line.start_station.idx},*] {mrt_line.end_station.idx} {mrt_line.cost}\n"]
        lines+= [";\n"]
        
        lines+= ["param V:=\n"]
        for mrt_line in self.mrt_lines:
            lines+= [f"[{mrt_line.start_station.idx},*] {mrt_line.end_station.idx} {mrt_line.freight_capacity}\n"]
        lines+= [";\n"]
        
        lines+= ["param f:=\n"]
        for locker in self.lockers:
            lines+= [f"{locker.idx}\t{locker.cost}\n"]
        lines+= [";\n"]
        
        lines+= ["param e:\n"]
        lockers_idx = [locker.idx for locker in self.lockers]
        lines+= ["\t"+"\t".join([str(l_idx) for l_idx in lockers_idx])+":=\n"]
        for customer in self.customers:
            if not (customer.is_self_pickup or customer.is_flexible):
                continue
            line = f"{customer.idx}\t"
            for locker in self.lockers:
                if locker.idx in customer.preferred_locker_idxs:
                    line+= "1\t"
                else:
                    line+= "0\t"
            line += "\n"
            lines+=[line]
        
        lines+= ["param p:=\n"]
        for vehicle in self.vehicles:
            lines+= [f"{vehicle.idx}\t{vehicle.cost}\n"]
        lines+= [";\n"]
        
        lines+= ["param G:=\n"]
        for locker in self.lockers:
            lines+= [f"{locker.idx}\t{locker.capacity}\n"]
        lines+= [";\n"]
        
        lines+= ["param s:=\n"]
        lines+= ["0\t0\n"]
        for customer in self.customers:
            lines+= [f"{customer.idx}\t{customer.service_time}"]
        for locker in self.lockers:
            lines+= [f"{locker.idx}\t{locker.service_time}"]
        lines+= [";\n"]
        
        lines+= ["param t:=\n"]
        line = "\t"+"\t".join([str(i) for i in range(self.num_nodes)])
        lines+= [line]
        for i in range(self.num_nodes):
            lines+= [f"{str(i)}\t"+"\t".join(str(self.distance_matrix[i,j]) for j in range(self.num_nodes)  )]
        lines+= [";\n"]
         
        with open(filepath.absolute(), "w") as save_file:
            save_file.writelines(lines)
        

def read_from_file(filename:str)->Cvrpptpl:
    directory = pathlib.Path("")/"instances"
    filepath = directory/filename
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
           idx, x, y, service_time, demand = [int(v) for v in line]
           coord = np.asanyarray([x,y], dtype=int)
           hd_customer = Customer(idx, coord, service_time, demand)
           customers += [hd_customer]
           line_idx += 1
        line_idx += 2
        # 4 self pickup
        while "flexible" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, service_time, demand, locker_idxs_str = line
           idx, x, y, service_time, demand = int(idx), int(x), int(y), int(service_time), int(demand)
           locker_idxs_str = locker_idxs_str.split("-")
           locker_idxs = [int(locker_idx_str) for locker_idx_str in locker_idxs_str]
           coord = np.asanyarray([x,y], dtype=int)
           sp_customer = Customer(idx, coord, service_time, demand, is_self_pickup=True, preferred_locker_idxs=locker_idxs)
           customers += [sp_customer]
           line_idx += 1
        line_idx += 2
        # 5 flexible
        while "lockers" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, service_time, demand, locker_idxs_str = line
           idx, x, y, service_time, demand = int(idx), int(x), int(y), int(service_time), int(demand)
           locker_idxs_str = locker_idxs_str.split("-")
           locker_idxs = [int(locker_idx_str) for locker_idx_str in locker_idxs_str]
           coord = np.asanyarray([x,y], dtype=int)
           fx_customer = Customer(idx, coord, service_time, demand, is_flexible=True, preferred_locker_idxs=locker_idxs)
           customers += [fx_customer]
           line_idx += 1
        line_idx += 2
        # 6 lockers
        while "mrt" not in lines[line_idx]:
            line = lines[line_idx].split(",")
            idx,x,y,service_time,capacity,cost = [int(v) for v in line]
            coord = np.asanyarray([x,y], dtype=int)
            locker = Locker(idx, coord, service_time, capacity, cost)
            locker_idx_dict[idx] = locker
            lockers += [locker]
            line_idx += 1
        line_idx += 2
        # 7 mrt lines
        while "distance" not in lines[line_idx]:
            line = lines[line_idx].split(",")
            start_idx, end_idx, capacity, cost = [int(v) for v in line]
            start_station, end_station = locker_idx_dict[start_idx], locker_idx_dict[end_idx]
            mrt_line = MrtLine(start_station, end_station, start_station.service_time, cost, capacity)
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

    clm_pos = filename.find("clm_")
    dlm_pos = filename.find("dlm_")
    lcr_pos = filename.find("lcr_")
    llm_pos = filename.find("llm_")
    pr_pos = filename.find("pr_")
    fr_pos = filename.find("fr_")
    fcm_pos = filename.find("fcm_")
    nc_pos = filename.find("nc_")
    
    
    customer_location_mode = filename[clm_pos+4:dlm_pos-1]
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
                       customer_location_mode,
                       distance_matrix)
    return problem
    
    
    