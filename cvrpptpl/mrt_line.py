import random
from typing import List

import numpy as np
from scipy.spatial import distance_matrix as dm_func
from scipy.cluster.hierarchy import fcluster, ward
from scipy.special import softmax
import matplotlib.pyplot as plt

from cvrpptpl.customer import Customer
from cvrpptpl.locker import Locker


class MrtLine:
    def __init__(self,
                 start_station: Locker,
                 end_station: Locker,
                 start_station_service_time:int,
                 cost: int,
                 freight_capacity:int) -> None:
        self.start_station = start_station
        self.end_station = end_station
        self.start_station.service_time = start_station_service_time
        self.cost = cost
        self.freight_capacity = freight_capacity
        
    def __str__(self) -> str:
        return str(self.start_station.idx)+","+str(self.end_station.idx)+","+str(self.freight_capacity)+","+str(self.cost)+"\n"

        
def generate_mrt_lines(num_mrt_stations: int,
                        lockers: List[Locker],
                        customers: List[Customer],
                        mrt_line_cost: float,
                        freight_capacity_mode: str="a",
                        service_time: int=30)->List[MrtLine]:
    locker_coords = np.stack([locker.coord for locker in lockers])
    num_mrt_lines = num_mrt_stations//2
    num_lockers = len(lockers)
    # locker_capacities = np.asanyarray([locker.capacity for locker in lockers])
    # z = ward(dm_func(locker_coords,locker_coords))
    # cluster_idxs = fcluster(z, t=500, criterion="distance")
    # cluster_idx_unique = np.unique(cluster_idxs)
    locker_idxs = np.asanyarray([locker.idx for locker in lockers])
    locker_demands = np.zeros_like(locker_idxs, dtype=float)
    for customer in customers:
        if not customer.is_self_pickup:
            continue
        for locker_idx in customer.preferred_locker_idxs:
            locker_demands[locker_idxs==locker_idx] += customer.demand
    chosen_idxs = []
    mrt_lines: List[MrtLine] = []
    locker_dm = dm_func(locker_coords, locker_coords)
    # we will prefer stations with high demands as end stations
    # and stations far from the chosen end stations as the starting stations
    
    end_station_coords = np.empty([0,2], dtype=float)
    for i in range(num_mrt_lines):
        logits = locker_demands.copy()
        if len(end_station_coords)>0:
            locker_to_end_station_dm = dm_func(locker_coords, end_station_coords)
            locker_to_end_station_dm = np.min(locker_to_end_station_dm, axis=-1)
            logits += locker_to_end_station_dm
        logits[chosen_idxs] = -np.inf
        probs = softmax(logits)
        chosen_end_idx = np.random.choice(num_lockers, size=1, p=probs)[0]
        chosen_idxs += [chosen_end_idx]
        end_station = lockers[chosen_end_idx]
        end_station_coords = np.concatenate([end_station_coords, end_station.coord[np.newaxis,:]], axis=0)
        
        logits = locker_dm[:, chosen_end_idx].copy()
        logits[chosen_idxs] = -np.inf
        probs = softmax(logits)
        chosen_start_idx = np.random.choice(num_lockers, size=1, p=probs)[0]
        chosen_idxs += [chosen_start_idx]
        start_station = lockers[chosen_start_idx]
        
        if freight_capacity_mode == "a":
            freight_capacity = 10000
        elif freight_capacity_mode == "e":
            r = random.random()*0.2 + 0.8
            freight_capacity = int(locker_demands[chosen_end_idx]*r)
        freight_capacity = max(freight_capacity, end_station.capacity)
        mrt_line = MrtLine(start_station, end_station, service_time, mrt_line_cost,  freight_capacity)
        mrt_lines += [mrt_line]
    return mrt_lines