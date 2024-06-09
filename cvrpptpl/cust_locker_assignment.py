from typing import List

import numpy as np
from scipy.spatial import distance_matrix as dm_func


from cvrpptpl.customer import Customer
from cvrpptpl.locker import Locker

def assign_customers_to_lockers(customers:List[Customer],
                                lockers:List[Locker],
                                self_pickup_ratio: float)->List[Customer]:
    num_customers = len(customers)
    num_sp_customers = int(num_customers*self_pickup_ratio)
    customer_coords = np.stack([customer.coord for customer in customers])
    demands = np.asanyarray([customer.demand for customer in customers])
    locker_coords = np.stack([locker.coord for locker in lockers])
    locker_capacities = np.asanyarray([locker.capacity for locker in lockers])

    potential_sp_cust_idxs = np.random.choice(num_customers, size=num_sp_customers, replace=False)
    potential_sp_cust_idxs = np.sort(potential_sp_cust_idxs)
    cust_to_locker_distance = dm_func(customer_coords, locker_coords)
    locker_loads = np.zeros_like(locker_capacities)
    locker_sorted_idxs = []
    for p_sp_idx in potential_sp_cust_idxs:
        dists = cust_to_locker_distance[p_sp_idx, :]
        sorted_idxs = np.argsort(-dists)
        locker_sorted_idxs += [sorted_idxs]
    sp_locker_sorted_idxs = [[potential_sp_cust_idxs[i], locker_sorted_idxs[i]] for i in range(num_sp_customers)]
    
    
    for p_sp_locker_sorted_idxs in sp_locker_sorted_idxs:
        p_sp_idx, sorted_idxs = p_sp_locker_sorted_idxs
        for p_locker_idx in sorted_idxs:
            if locker_loads[p_locker_idx] + demands[p_sp_idx] > locker_capacities[p_locker_idx]:
                continue
            locker_loads[p_locker_idx] += demands[p_sp_idx]
            customers[p_sp_idx].is_self_pickup = True
            locker_idx = lockers[p_locker_idx].idx
            customers[p_sp_idx].preferred_locker_idxs = [locker_idx]
            customers[p_sp_idx].is_self_pickup = True
            break
    return customers
    