from copy import copy
import random
from typing import List

import numpy as np
from scipy.spatial import distance_matrix as dm_func

from problem.customer import Customer
from problem.locker import Locker


def generate_customer_locker_preferences(customers:List[Customer],
                                lockers:List[Locker])->List[Customer]:
    all_coords = [np.zeros([2])]
    all_coords += [customer.coord for customer in customers]
    all_coords += [locker.coord for locker in lockers]
    all_coords = np.stack(all_coords)
    all_coords[0,:] = all_coords[1:,:].min(axis=0)
    sp_customers = [customer for customer in customers if (customer.is_self_pickup or customer.is_flexible)]
    customer_coords = np.stack([customer.coord for customer in customers])
    locker_coords = np.stack([locker.coord for locker in lockers])

    cust_to_locker_distance = dm_func(customer_coords, locker_coords)
    for i, customer in enumerate(sp_customers):
        dists = cust_to_locker_distance[i, :]
        sorted_idxs = np.argsort(-dists)
        locker_idxs = [lockers[p_locker_idx].idx for p_locker_idx in sorted_idxs]
        sp_customers[i].preferred_locker_idxs = locker_idxs
        
    
    # trim again with reasonable radius, locker must be in a reasonable radius from each customer
    # but ensure at list, one locker is assigned, even though it's far away
    min_coord, max_coord = all_coords.min(axis=0), all_coords.max(axis=0)
    diag_range = np.linalg.norm(min_coord-max_coord)
    for i, customer in enumerate(sp_customers):
        c_coord = customer.coord
        final_list = copy(customer.preferred_locker_idxs)
        for l_idx in customer.preferred_locker_idxs:
            if len(final_list)==1:
                break
            l_coord = all_coords[l_idx, :]
            dist = np.linalg.norm(c_coord-l_coord)
            ratio = dist/diag_range
            if ratio > 0.2 and len(final_list)>1:
                final_list.remove(l_idx)
        customer.preferred_locker_idxs = final_list
    
    # trim the preferences randomly
    for c_idx, customer in enumerate(sp_customers):
        min_num_locker = min(1, len(customer.preferred_locker_idxs))
        max_num_locker = min(5, len(customer.preferred_locker_idxs))
        num_locker = random.randint(min_num_locker, max_num_locker)
        customers[c_idx].preferred_locker_idxs = customers[c_idx].preferred_locker_idxs[:num_locker]    
    
    return customers


# def generate_customer_locker_preferences(customers:List[Customer],
#                                 lockers:List[Locker],
#                                 self_pickup_ratio: float,
#                                 flexible_ratio)->List[Customer]:
#     num_customers = len(customers)
#     num_sp_customers = int(num_customers*(self_pickup_ratio+flexible_ratio))
#     customer_coords = np.stack([customer.coord for customer in customers])
#     locker_coords = np.stack([locker.coord for locker in lockers])

#     potential_sp_cust_idxs = np.random.choice(num_customers, size=num_sp_customers, replace=False)
#     potential_sp_cust_idxs = np.sort(potential_sp_cust_idxs)
#     cust_to_locker_distance = dm_func(customer_coords, locker_coords)
#     for p_sp_idx in potential_sp_cust_idxs:
#         dists = cust_to_locker_distance[p_sp_idx, :]
#         sorted_idxs = np.argsort(-dists)
#         locker_idxs = [lockers[p_locker_idx].idx for p_locker_idx in sorted_idxs]
#         customers[p_sp_idx].is_self_pickup = len(locker_idxs)>0
#         customers[p_sp_idx].preferred_locker_idxs = locker_idxs
        
    
#     # divide self pickup customers and flexible customers
#     sp_customer_idxs = [c_idx for c_idx, customer in enumerate(customers) if customer.is_self_pickup]
#     num_flexible_customer = int(flexible_ratio*num_customers)
#     fx_customer_idxs = random.sample(sp_customer_idxs, k=num_flexible_customer)
#     for f_idx in fx_customer_idxs:
#         customers[f_idx].is_self_pickup = False
#         customers[f_idx].is_flexible = True
    
#     # trim the preferences randomly
#     for c_idx, customer in enumerate(customers):
#         if not customer.is_self_pickup:
#             continue
#         min_num_locker = min(3, len(customer.preferred_locker_idxs))
#         num_locker = random.randint(min_num_locker, len(customer.preferred_locker_idxs))
#         customers[c_idx].preferred_locker_idxs = customers[c_idx].preferred_locker_idxs[:num_locker]
#     return customers
    