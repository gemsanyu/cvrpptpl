import argparse
import sys
from random import randint
from typing import List

import numpy as np
from scipy.spatial import distance_matrix as dm_func

from problem.customer import Customer
from problem.locker import Locker


def prepare_args():
    parser = argparse.ArgumentParser(description='CVRP-PT-PL instance generation')
    
    # args for generating instance based on CVRP problem instances
    parser.add_argument('--cvrp-instance-name',
                        type=str,
                        default="A-n32-k5",
                        help="the cvrp instance name")
    
    parser.add_argument('--num-customers',
                        type=int,
                        default=0,
                        help="the number of customers, must be between 1 and number of customers in the original problem instance, \
                            or if set to 0 means it follows the original problem instance")
    
    
    
    # customers
    parser.add_argument('--pickup-ratio',
                        type=float,
                        default=1/3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=1/3,
                        help='ratio of flexible customers/number of customers, used to determine number of flexible customers')
    
    
    # locker
    parser.add_argument('--num-external-lockers',
                        type=int,
                        default=4,
                        help='number of lockers outside of mrt stations')
    parser.add_argument('--min-locker-capacity',
                        type=int,
                        default=70,
                        help='min range of locker capacity to random')
    parser.add_argument('--max-locker-capacity',
                        type=int,
                        default=100,
                        help='max range of locker capacity to random')
    
    parser.add_argument('--locker-location-mode',
                        type=str,
                        default="c",
                        choices=["c","r","rc"],
                        help='lockers\' location distribution mode. \
                            r: randomly scattered \
                            c: each cluster of customers gets a locker if possible \
                            rc: half clustered half random')
    
    # mrt
    parser.add_argument('--mrt-line-cost',
                        type=float,
                        default=2,
                        help='mrt line cost per unit goods')
    
    
    # vehicles
    parser.add_argument('--num-vehicles',
                        type=int,
                        default=0,
                        help='0 means use same num vehicles as original, >0 means use this num instead')
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=1,
                        help='vehicle cost per unit travelled distance')
    parser.add_argument('--vehicle-capacity',
                        type=int,
                        default=-1,
                        help='vehicle capacity')
    
    args = parser.parse_args(sys.argv[1:])
    return args




def generate_customers(original_customers: List[Customer],
                       lockers: List[Locker],
                       depot_coord: np.ndarray,
                       num_hd_customers: int,
                       num_sp_customers: int,
                       num_fx_customers: int)->List[Customer]:

    cust_coords: np.ndarray = np.stack([customer.coord for customer in original_customers])
    customers: list[Customer] = []
    is_coords_chosen = np.zeros((len(cust_coords),), dtype=bool)
    num_sp_fx_customers = num_sp_customers + num_fx_customers

    mrt_coords_list = [locker.coord for locker in lockers if locker.is_mrt_station]
    mrt_coords = np.stack(mrt_coords_list)
    distance_to_mrt_coords = dm_func(cust_coords, mrt_coords)
    distance_to_depot = dm_func(cust_coords, depot_coord)
    print(distance_to_mrt_coords)
    # # Filter coordinates within 10 km radius of Taipei center
    # is_not_too_far = distance_to_center < 5
    # # is_far_enough_from_mrt_stations = distances_to_mrt_stations > 8
    # # potential_hd_coords = coords[np.logical_and(is_not_too_far, is_far_enough_from_mrt_stations)]
    # potential_hd_coords = coords[is_not_too_far]
    # n_candidates = len(potential_hd_coords)
    # if n_candidates < num_hd_customers:
    #     raise ValueError("Not enough potential HD coordinates within 10 km.")
    # chosen_indices:List[int] = []
    # shuffled_indices = np.random.permutation(n_candidates)
    # min_dist_km = 0.2
    # for idx in shuffled_indices:
    #     coord = potential_hd_coords[idx]

    #     # Check distance from all previously chosen ones
    #     if not chosen_indices:
    #         chosen_indices.append(idx)
    #         continue

    #     chosen_coords = potential_hd_coords[chosen_indices]
    #     dists = haversine_distances(
    #         np.radians([coord]), np.radians(chosen_coords)
    #     )[0] * 6371.088  # km

    #     if np.all(dists >= min_dist_km):
    #         chosen_indices.append(idx)

    #     if len(chosen_indices) >= num_hd_customers:
    #         break

    # if len(chosen_indices) < num_hd_customers:
    #     print(f"⚠️ Only selected {len(chosen_indices)} customers due to spacing constraint.")

    # hd_cust_coords = potential_hd_coords[chosen_indices]

    # for i, c_idx in enumerate(chosen_indices):
    #     demand = randint(min_demand, max_demand)
    #     is_coords_chosen[c_idx] = True
    #     customer = Customer(i, hd_cust_coords[i], 10, demand)
    #     customers.append(customer)
    
    # # distribute customer counts evenly among lockers
    # num_customers_for_lockers = np.zeros((len(lockers),), dtype=int)
    # for i in range(num_sp_fx_customers):
    #     num_customers_for_lockers[i % len(lockers)] += 1

    # locker_coords = np.asanyarray([locker.coord for locker in lockers])
    # sp_fx_cust_coords = np.empty((0, 2), dtype=float)

    # max_dist_to_locker_km = 2
    # for li, locker in enumerate(lockers):
    #     locker_coord = np.asanyarray([locker.coord])
    #     n_to_sample = num_customers_for_lockers[li]
    #     if (n_to_sample==0):
    #         continue

    #     # Distance from all candidates to this locker
    #     distance_to_locker = (
    #         haversine_distances(np.radians(coords), np.radians(locker_coord)).flatten() * 6371.088
    #     )
    #     is_close_enough = distance_to_locker < max_dist_to_locker_km
    #     potential_coords = coords[is_close_enough]
    #     potential_dists = distance_to_locker[is_close_enough]

    #     if len(potential_coords) < n_to_sample:
    #         print(f"⚠️ Locker {li}: Not enough candidates within {max_dist_to_locker_km} km.")
    #         continue

    #     # Prefer nearer customers with probability ∝ 1/dist
    #     weights = np.maximum(max_dist_to_locker_km - potential_dists, 0)
    #     probs = weights / np.sum(weights)

    #     chosen_coords_for_locker = []
    #     attempts = 0

    #     while len(chosen_coords_for_locker) < n_to_sample and attempts < len(potential_coords) * 3:
    #         attempts += 1

    #         # Sample one candidate based on distance weighting
    #         idx = np.random.choice(len(potential_coords), p=probs)
    #         candidate = potential_coords[idx]

    #         # Skip if too close to any already chosen customer (global)
    #         if len(sp_fx_cust_coords) > 0:
    #             dists_to_existing = (
    #                 haversine_distances(
    #                     np.radians([candidate]),
    #                     np.radians(sp_fx_cust_coords)
    #                 )[0] * 6371.088
    #             )
    #             if np.any(dists_to_existing < min_dist_km):
    #                 continue

    #         # Skip if too close to previously chosen in this locker
    #         if len(chosen_coords_for_locker) > 0:
    #             dists_to_local = (
    #                 haversine_distances(
    #                     np.radians([candidate]),
    #                     np.radians(chosen_coords_for_locker)
    #                 )[0] * 6371.088
    #             )
    #             if np.any(dists_to_local < min_dist_km):
    #                 continue

    #         chosen_coords_for_locker.append(candidate)

    #     if len(chosen_coords_for_locker) < n_to_sample:
    #         print(f"⚠️ Locker {li}: Only got {len(chosen_coords_for_locker)} out of {n_to_sample} due to spacing.")
    #     print(sp_fx_cust_coords.shape, np.array(chosen_coords_for_locker).shape)
    #     sp_fx_cust_coords = np.concatenate(
    #         [sp_fx_cust_coords, np.array(chosen_coords_for_locker)], axis=0
    #     )
    
    # np.random.shuffle(sp_fx_cust_coords)
    # for i in range(num_sp_fx_customers):
    #     demand = randint(min_demand, max_demand)
    #     preferred_lockers_idx = []
    #     distance_to_lockers = haversine_distances(np.radians(sp_fx_cust_coords[[i]]), np.radians(locker_coords)).flatten()*6371.088
    #     sorted_idxs = np.argsort(distance_to_lockers)
    #     num_preferred_lockers = randint(2, 3)
    #     for l in range(num_preferred_lockers):
    #         locker = lockers[sorted_idxs[l]]
    #         distance = distance_to_lockers[sorted_idxs[l]]
    #         # print(distance, locker.idx)
    #         if distance < 3:
    #             preferred_lockers_idx.append(locker.idx)
    #     # exit()
    #     is_self_pickup = i<num_sp_customers
    #     is_flexible = i>=num_sp_customers
    #     customer = Customer(i, 
    #                         sp_fx_cust_coords[i],
    #                         10,
    #                         demand,
    #                         is_self_pickup,
    #                         is_flexible,
    #                         preferred_lockers_idx)
    #     customers.append(customer)
    # for ci, customer in enumerate(customers):
    #     customer.idx = ci+1
        
    return customers
