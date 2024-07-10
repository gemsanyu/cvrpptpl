from typing import List

import numpy as np
from scipy.spatial import distance_matrix as dm_func
from scipy.cluster.hierarchy import fcluster, ward
from scipy.special import softmax

from cvrpptpl.utils import get_coords_in_grid

class Locker:
    def __init__(self,
                 idx: int,
                 coord: np.ndarray,
                 capacity: int,
                 cost: float) -> None:
        self.idx = idx
        self.coord = coord
        self.capacity = capacity
        self.cost = cost
    
    def __str__(self) -> str:
        return str(self.idx)+","+str(self.coord[0])+","+str(self.coord[1])+","+str(self.capacity)+","+str(self.cost)+"\n"


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
    grid_size = 1001
    all_coords = get_coords_in_grid(grid_size)
    chosen_idxs = customer_coords[:,0]//grid_size + customer_coords[:, 1]%grid_size
    locker_coords = np.empty([0,2], dtype=int)
    if mode in ["c", "rc"]:
        z = ward(dm_func(customer_coords,customer_coords))
        cluster_idxs = fcluster(z, t=500, criterion="distance")
        cluster_idx_unique = np.unique(cluster_idxs)
        cluster_idx = 1
        num_locker_to_generate = num_lockers
        if mode == "rc":
            num_locker_to_generate = min(len(cluster_idx_unique), num_lockers)
        
        while len(locker_coords)<num_locker_to_generate:
            coords = customer_coords[cluster_idxs==cluster_idx]
            min_coord, max_coord = np.min(coords, axis=0, keepdims=True), np.max(coords, axis=0, keepdims=True)
            centroid = (min_coord+max_coord)/2
            dist_to_centroid = dm_func(all_coords, centroid)
            dist_to_centroid[chosen_idxs] = np.inf
            probs = softmax(-np.sum(dist_to_centroid, axis=-1)/10)
            
            chosen_idx = np.random.choice(len(all_coords), size=1, p=probs)
            chosen_idxs = np.concatenate([chosen_idxs, chosen_idx])
            locker_coord = all_coords[chosen_idx, :]
            locker_coords = np.concatenate([locker_coords, locker_coord], axis=0)
            cluster_idx = (cluster_idx + 1)%len(cluster_idx_unique)
            if cluster_idx == 0:
                cluster_idx += 1
    num_remaining_lockers = num_lockers-len(locker_coords)
    # if mode == "r":
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
                     locker_location_mode: str):
    locker_coords = generate_locker_coords(num_lockers, customer_coords, locker_location_mode)
    locker_capacities = generate_locker_capacities(num_lockers,total_customer_demand,capacity_ratio)
    num_customers = len(customer_coords)
    lockers = [Locker(i+num_customers+1, locker_coords[i,:], locker_capacities[i], locker_cost) for i in range(num_lockers)]
    return lockers