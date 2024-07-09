import os
import pathlib
from typing import List

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
        self.distance_matrix = dm_func(self.coords, self.coords)
        
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

 
# def visualize_instance(depot_coord, customer_coords, locker_coords, mrt_idxs, mrt_adj_list, hd_cust_idx, sp_cust_idx, sp_cust_locker_pairs):
#     plt.scatter(depot_coord[:, 0], depot_coord[:, 1], marker="s", s=80, label="Depot")
#     plt.scatter(customer_coords[:,0], customer_coords[:,1], label="Customers")
#     plt.scatter(locker_coords[:,0], locker_coords[:,1], label="Lockers", marker="h", s=70)

#     start_mrt_stations = [mrt_pair[0] for mrt_pair in mrt_adj_list]
#     end_mrt_stations = [mrt_pair[1] for mrt_pair in mrt_adj_list]
#     plt.scatter(locker_coords[start_mrt_stations,0],locker_coords[start_mrt_stations,1], s=100, marker="^", label="Start MRT")
#     plt.scatter(locker_coords[end_mrt_stations,0],locker_coords[end_mrt_stations,1], s=100, marker="v", label="End MRT")
#     for i, mrt_pair in enumerate(mrt_adj_list):
#         mrt_pair_coords = locker_coords[mrt_pair,:]
#         if i==0:
#             plt.plot(mrt_pair_coords[:, 0], mrt_pair_coords[:, 1], "k--", label="MRT Line")
#         else:
#             plt.plot(mrt_pair_coords[:, 0], mrt_pair_coords[:, 1], "k--")
    
#     # plotting lines of self pickup and their lockers
#     for i, sp_locker_pair in enumerate(sp_cust_locker_pairs):
#         sp_idx, locker_idx = sp_locker_pair
#         sp_coord = customer_coords[sp_idx, :]
#         locker_coord = locker_coords[locker_idx, :]
#         coords_to_plot = np.stack([sp_coord, locker_coord], axis=0)
#         if i == 0:
#             plt.plot(coords_to_plot[:, 0], coords_to_plot[:, 1], "r--", label="Customers' pickup route")
#         else:
#             plt.plot(coords_to_plot[:, 0], coords_to_plot[:, 1], "r--")
#         plt.arrow(sp_coord[0], sp_coord[1], (locker_coord[0]-sp_coord[0])*0.9, (locker_coord[1]-sp_coord[1])*0.9, color="red")
#     plt.legend()
#     plt.show()
    
    