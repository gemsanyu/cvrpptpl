import random
from typing import List, Optional

import numpy as np
from scipy.spatial import distance_matrix as dm_func
from scipy.special import softmax

from cvrpptpl.utils import get_coords_in_grid

GRID_SIZE = 1001

class Customer:
    def __init__(self,
                 idx: int,
                 coord: np.ndarray,
                 service_time: int,
                 demand: int,
                 is_self_pickup: bool=False,
                 is_flexible: bool= False,
                 preferred_locker_idxs: Optional[List[int]] = None) -> None:
        self.idx = idx
        self.coord = coord
        self.service_time = service_time
        self.demand = demand
        self.is_self_pickup = is_self_pickup    
        self.is_flexible = is_flexible
        self.preferred_locker_idxs = preferred_locker_idxs
        
    def __str__(self) -> str:
        customer_str = str(self.idx)+","+str(self.coord[0])+","+str(self.coord[1])+","+str(self.service_time)+","+str(self.demand)
        if self.is_self_pickup or self.is_flexible:
            locker_idxs_str = [str(locker_idx)+"-" for locker_idx in self.preferred_locker_idxs]
            locker_idxs_str = "".join(locker_idxs_str)
            locker_idxs_str = locker_idxs_str[:-1]
            customer_str += ","+locker_idxs_str
        return customer_str+"\n"
        
def generate_clustered_coords(num_customers: int,
                      num_clusters: int,
                      cluster_dt: float) -> np.ndarray:
    num_clusters = min(num_clusters, num_customers)
    seeds = np.random.randint(0,GRID_SIZE,size=[num_clusters, 2])
    num_customers -= num_clusters
    all_coords = get_coords_in_grid(grid_size=GRID_SIZE)
    all_to_seeds_distance = dm_func(all_coords, seeds)
    seed_coord_idxs = seeds[:,0]*GRID_SIZE + seeds[:, 1]
    all_to_seeds_distance[seed_coord_idxs, :] = np.inf
    chosen_coords = np.empty([0,2], dtype=int)
    dist_to_ccs = np.empty([0,num_clusters], dtype=float)
    num_cust_left = num_customers
    cluster_size = int(num_customers/num_clusters)
    for cl_idx in range(num_clusters):
        pick_size = int(cluster_size * (random.random()*0.4 + 0.8))
        if cl_idx == num_clusters-1:        
            pick_size = num_cust_left
        logits = all_to_seeds_distance[:, cl_idx]
        probs = softmax(-logits/cluster_dt)
        pick_size = min(pick_size, num_cust_left)
        chosen_coord_idx = np.random.choice(len(probs), size=pick_size, replace=False, p=probs)
        all_to_seeds_distance[chosen_coord_idx, :] = np.inf
        num_cust_left -= pick_size
        chosen_coord = all_coords[chosen_coord_idx, :]
        dist_to_cc = dm_func(chosen_coord, seeds)
        dist_to_ccs = np.concatenate([dist_to_ccs, dist_to_cc], axis=0)
        chosen_coords = np.concatenate([chosen_coords, chosen_coord], axis=0)
    all_chosen_coords = np.concatenate([seeds, chosen_coords], axis=0)
    return all_chosen_coords
    
    

def generate_customer_coords(num_customers: int,
                             mode: str = "r",
                             num_clusters: int = 5,
                             cluster_dt: float = 40) -> np.ndarray:
    """_summary_

    Args:
        num_customers (int): _description_
        mode (str, optional): _description_. Defaults to "r".
        num_clusters (int, optional): _description_. Defaults to 5.
        cluster_dt (float, optional): _description_. Defaults to 40.

    Returns:
        np.ndarray: coordinates
    """
    if mode == "r":
        coords = np.random.randint(0,10001,size=[num_customers, 2])
    if mode == "c":
        coords = generate_clustered_coords(num_customers, num_clusters, cluster_dt)
    if mode == "rc":
        num_clustered_customers = int(num_customers/2)
        clustered_coords = generate_clustered_coords(num_clustered_customers, num_clusters, cluster_dt)
        num_random_customers = num_customers-num_clustered_customers
        random_coords = np.random.randint(0,10001,size=[num_random_customers, 2])
        coords = np.concatenate([clustered_coords, random_coords], axis=0)
    return coords

def generate_customer_demands(
                     num_customers: int, 
                     customer_coords:np.ndarray, 
                     mode: str = "u"):
    """_summary_

    Args:
        num_customers (int): _description_
        customer_coords (np.ndarray): _description_
        mode (str, optional):   u for uniform (all demands equal 1)
                                q means dividing the coords into 4 quartiles,
                                then even quartiles get small demands,
                                odd quartiles get large demands. Defaults to "u".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if mode == "u":
        cust_demands = np.ones([num_customers,], dtype=int)
    elif mode == "q":
        min_coord = np.min(customer_coords, axis=0, keepdims=True)
        max_coord = np.max(customer_coords, axis=0, keepdims=True)
        mid_coord = (min_coord+max_coord)/2
        is_first_quadrant = np.all(customer_coords>=mid_coord, axis=-1)
        is_third_quadrant = np.all(customer_coords<=mid_coord, axis=-1)
        is_odd_quadrant = np.logical_or(is_first_quadrant, is_third_quadrant)
        is_even_quadrant = np.logical_not(is_odd_quadrant)
        cust_demands = np.zeros([num_customers,], dtype=int)
        cust_demands[is_even_quadrant] = np.random.randint(1, 51, size=[np.sum(is_even_quadrant),])
        cust_demands[is_odd_quadrant] = np.random.randint(51, 101, size=[np.sum(is_odd_quadrant),])
    elif "-" in mode:
        bounds = mode.split(sep="-")
        lb, ub = int(bounds[0]), int(bounds[1])
        cust_demands = np.random.randint(lb, ub+1, size=[num_customers,])
    elif mode == "sl":
        r = random.random()*0.25 + 0.7
        num_small_demands = int(r*num_customers)
        num_large_demands = num_customers-num_small_demands
        small_demands = np.random.randint(1, 11, size=[num_small_demands])
        large_demands = np.random.randint(50,101, size=[num_large_demands,])
        cust_demands = np.concatenate([small_demands, large_demands])
        np.random.shuffle(cust_demands)  
    else:
        raise ValueError("mode unrecognized")
    return cust_demands

def generate_customers(num_customers:int,
                       location_mode:str,
                       num_clusters:int,
                       cluster_dt:float,
                       demand_generation_mode:str,
                       service_time:int=15)->List[Customer]:
    cust_coords = generate_customer_coords(num_customers,
                                               location_mode,
                                               num_clusters,
                                               cluster_dt)
    cust_demands = generate_customer_demands(num_customers,
                                             cust_coords,
                                             demand_generation_mode)
    customers = [Customer(i+1, cust_coords[i], service_time, cust_demands[i]) for i in range(num_customers)]
    return customers
