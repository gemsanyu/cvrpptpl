import random
from typing import List

import numpy as np
from scipy.spatial import distance_matrix as dm_func

from cvrpptpl.customer import Customer
from cvrpptpl.locker import Locker


def generate_customer_locker_preferences(customers:List[Customer],
                                lockers:List[Locker],
                                self_pickup_ratio: float,
                                flexible_ratio)->List[Customer]:
    num_customers = len(customers)
    num_sp_customers = int(num_customers*(self_pickup_ratio+flexible_ratio))
    customer_coords = np.stack([customer.coord for customer in customers])
    locker_coords = np.stack([locker.coord for locker in lockers])

    potential_sp_cust_idxs = np.random.choice(num_customers, size=num_sp_customers, replace=False)
    potential_sp_cust_idxs = np.sort(potential_sp_cust_idxs)
    cust_to_locker_distance = dm_func(customer_coords, locker_coords)
    for p_sp_idx in potential_sp_cust_idxs:
        dists = cust_to_locker_distance[p_sp_idx, :]
        sorted_idxs = np.argsort(-dists)
        locker_idxs = [lockers[p_locker_idx].idx for p_locker_idx in sorted_idxs]
        customers[p_sp_idx].is_self_pickup = len(locker_idxs)>0
        customers[p_sp_idx].preferred_locker_idxs = locker_idxs
        
    
    # divide self pickup customers and flexible customers
    sp_customer_idxs = [c_idx for c_idx, customer in enumerate(customers) if customer.is_self_pickup]
    num_flexible_customer = int(flexible_ratio*num_customers)
    fx_customer_idxs = random.sample(sp_customer_idxs, k=num_flexible_customer)
    for f_idx in fx_customer_idxs:
        customers[f_idx].is_self_pickup = False
        customers[f_idx].is_flexible = True
    
    # trim the preferences randomly
    for c_idx, customer in enumerate(customers):
        if not customer.is_self_pickup:
            continue
        min_num_locker = min(3, len(customer.preferred_locker_idxs))
        num_locker = random.randint(min_num_locker, len(customer.preferred_locker_idxs))
        customers[c_idx].preferred_locker_idxs = customers[c_idx].preferred_locker_idxs[:num_locker]
    return customers
    