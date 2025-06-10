from typing import List

import numpy as np
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial import distance_matrix as dm_func
from scipy.special import softmax

from problem.node import Node
from problem.utils import get_coords_in_grid


class Locker(Node):
    def __init__(self,
                 idx: int,
                 coord: np.ndarray,
                 service_time: int,
                 capacity: int) -> None:
        super().__init__(idx, coord)
        self.service_time = service_time
        self.capacity = capacity
    
    def __str__(self) -> str:
        return str(self.idx)+","+str(self.coord[0])+","+str(self.coord[1])+","+str(self.service_time)+","+str(self.capacity)+"\n"


def generate_locker_capacities(num_lockers: int,
                                total_customer_demand: int,
                                capacity_ratio: float) -> np.ndarray:
    
    locker_total_capacity = int(capacity_ratio*total_customer_demand)
    capacity = int(locker_total_capacity/num_lockers)
    locker_capacities = np.full([num_lockers,], capacity, dtype=int)
    return locker_capacities

def generate_locker_coords(num_lockers: int,
                           customer_coords: np.ndarray,
                           mode: str):
    """generate lockers, one for each cluster,
    cluster are first ordered by the number of nodes in it

    Args:
        num_lockers (int): _description_
        customer_coords (np.ndarray): _description_
        mode (str): _description_

    Returns:
        _type_: _description_
    """
    grid_size = 1001
    all_coords = get_coords_in_grid(grid_size)
    chosen_idxs = customer_coords[:,0]//grid_size + customer_coords[:, 1]%grid_size
    locker_coords = np.empty([0,2], dtype=int)
    if mode in ["c", "rc"]:
        z = ward(dm_func(customer_coords,customer_coords))
        cluster_idxs = fcluster(z, t=900, criterion="distance")
        cluster_idx_unique = np.unique(cluster_idxs)
        num_nodes_in_cluster = np.asanyarray([np.sum(cluster_idxs==cluster_idx) for cluster_idx in cluster_idx_unique])
        sort_idx = np.argsort(-num_nodes_in_cluster)
        cluster_idx_unique = cluster_idx_unique[sort_idx]
        num_locker_to_generate = num_lockers
        if mode == "rc":
            num_locker_to_generate = min(len(cluster_idx_unique), num_lockers)
        
        i = 0
        while len(locker_coords)<num_locker_to_generate:
            cluster_idx = cluster_idx_unique[i%len(cluster_idx_unique)]
            i += 1
            coords = customer_coords[cluster_idxs==cluster_idx]
            min_coord, max_coord = np.min(coords, axis=0, keepdims=True), np.max(coords, axis=0, keepdims=True)
            centroid = (min_coord+max_coord)/2
            dist_to_centroid = dm_func(all_coords, centroid)
            dist_to_centroid = np.sum(dist_to_centroid, axis=-1)
            # also reduce the probs based on the closest distance to other locker
            if len(locker_coords)>0:
                dist_to_lockers = dm_func(all_coords, locker_coords)
                dist_to_lockers = np.min(dist_to_lockers, axis=-1)
                dist_to_centroid = dist_to_centroid - dist_to_lockers*0.7

            dist_to_centroid[chosen_idxs] = np.inf
            probs = softmax(-dist_to_centroid/3)

            
            chosen_idx = np.random.choice(len(all_coords), size=1, p=probs)
            chosen_idxs = np.concatenate([chosen_idxs, chosen_idx])
            locker_coord = all_coords[chosen_idx, :]
            locker_coords = np.concatenate([locker_coords, locker_coord], axis=0)
    num_remaining_lockers = num_lockers-len(locker_coords)
    if num_remaining_lockers>0:
        logits = np.ones([len(all_coords),])
        logits[chosen_idxs] = -np.inf
        probs = softmax(logits)
        new_locker_idxs = np.random.choice(len(all_coords), size=[num_remaining_lockers,], p=probs)
        new_locker_coords = all_coords[new_locker_idxs, :]
        locker_coords = np.concatenate([locker_coords, new_locker_coords], axis=0)
    return locker_coords
        
def generate_lockers(num_lockers:int,
                     customer_coords: np.ndarray,
                     total_customer_demand: int,
                     capacity_ratio: int,
                     locker_cost: float,
                     locker_location_mode: str,
                     service_time:int=15):
    locker_coords = generate_locker_coords(num_lockers, customer_coords, locker_location_mode)
    locker_capacities = generate_locker_capacities(num_lockers,total_customer_demand,capacity_ratio)
    num_customers = len(customer_coords)
    lockers = [Locker(i+num_customers+1, locker_coords[i,:], service_time, locker_capacities[i]) for i in range(num_lockers)]
    return lockers

def generate_lockers_v2(num_lockers:int,
                     customer_coords: np.ndarray,
                     min_locker_capacity:int,
                     max_locker_capacity:int,
                     locker_location_mode: str,
                     service_time:int=10):
    locker_coords = generate_locker_coords(num_lockers, customer_coords, locker_location_mode)
    locker_capacities = np.random.randint(min_locker_capacity, max_locker_capacity+1, size=(num_lockers,))
    num_customers = len(customer_coords)
    lockers = [Locker(i+num_customers+1, locker_coords[i,:], service_time, locker_capacities[i]) for i in range(num_lockers)]
    return lockers