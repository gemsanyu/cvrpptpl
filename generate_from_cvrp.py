import math
from copy import deepcopy
from random import randint, random, shuffle
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix as dm_func
from sklearn.cluster import KMeans

from cpsat import best_fit_spfx_assignment
from cvrp_instance_utils import prepare_args
from problem.customer import Customer
from problem.cvrp import read_from_file
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_line import generate_mrt_network_soumen


def add_mrt_lockers_to_preference(new_customers: List[Customer], mrt_lockers: List[Locker]) -> List[Customer]:
    if len(mrt_lockers)==0:
        return new_customers
    cust_coords = np.asanyarray([cust.coord for cust in new_customers])
    locker_coords = np.asanyarray([locker.coord for locker in mrt_lockers])
    dist_custs_to_lockers = dm_func(cust_coords, locker_coords)
    all_coords = [cust.coord for cust in new_customers] + [locker.coord for locker in mrt_lockers]
    all_coords = np.stack(all_coords)
    min_coord, max_coord = all_coords.min(axis=0), all_coords.max(axis=0)
    diag_range = np.linalg.norm(min_coord-max_coord)
    for ci, customer in enumerate(new_customers):
        if not(customer.is_self_pickup or customer.is_flexible):
            continue
        dist_to_lockers = dist_custs_to_lockers[ci,:].flatten()
        sorted_idxs = np.argsort(dist_to_lockers)
        sorted_dist_to_lockers = dist_to_lockers[sorted_idxs]
        # has a 5% chance to include the closest mrt lockers even if outside 
        # reasonable radius
        customer.preferred_locker_idxs.append(mrt_lockers[sorted_idxs[0]].idx)
        closest_dist_2 = sorted_dist_to_lockers[1]
        if closest_dist_2/diag_range <= 0.2:
            customer.preferred_locker_idxs.append(mrt_lockers[sorted_idxs[1]].idx)
        elif random()<=0.05:
            customer.preferred_locker_idxs.append(mrt_lockers[sorted_idxs[1]].idx)
    return new_customers

def readjust_lockers_capacities(customers:List[Customer], lockers:List[Locker], vehicle_capacity:int)->List[Locker]:
    cust_dest_assignments = np.full([len(customers)+1,], -1, dtype=int)
    num_nodes = max([locker.idx for locker in lockers]) + 1
    locker_loads = np.zeros([num_nodes,], dtype=int)
    cust_dest_assignments = best_fit_spfx_assignment(customers, lockers, vehicle_capacity)
    locker_loads = np.zeros([num_nodes,], dtype=int)
    for customer in customers:
        if cust_dest_assignments[customer.idx] > 0:
            locker_idx = cust_dest_assignments[customer.idx]
            locker_loads[locker_idx] += customer.demand
    for locker in lockers:
        locker.capacity = max(locker.capacity, locker_loads[locker.idx].item())
    return lockers


def cluster_customers_for_lockers(customers, num_ext_lockers: int):
    """
    Cluster leftover customers into groups for external lockers.
    Returns: 
        cluster_centers (np.ndarray): shape (num_ext_lockers, 2)
        customer_labels (np.ndarray): shape (n_customers,)
    """
    cust_coords = np.asarray([cust.coord for cust in customers])
    
    kmeans = KMeans(n_clusters=num_ext_lockers, n_init=20, random_state=42)
    labels = kmeans.fit_predict(cust_coords)
    centers = kmeans.cluster_centers_
    
    return centers, labels

def select_external_lockers_from_clusters(
    customers: List[Customer],
    num_external_lockers: int,
    mrt_lockers: List[Locker],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cluster all customers into (num_external_lockers + len(mrt_lockers)) clusters,
    then choose the cluster centroids farthest from any MRT locker as external lockers.

    Returns:
      chosen_centroids: (num_external_lockers, 2)
      all_centroids: (K_total, 2)
      labels: (n_customers,)
    """
    K_total = num_external_lockers + len(mrt_lockers)

    # cluster all customers
    centroids, labels = cluster_customers_for_lockers(customers, K_total)

    # compute distance from each centroid to nearest MRT locker
    if len(mrt_lockers) > 0:
        mrt_coords = np.asarray([l.coord for l in mrt_lockers])
        dist_to_mrt = dm_func(centroids, mrt_coords).min(axis=1)
    else:
        dist_to_mrt = np.full(K_total, np.inf)

    # select farthest centroids
    chosen_idxs = np.argsort(dist_to_mrt)[-num_external_lockers:]  # largest distances
    chosen_centroids = centroids[chosen_idxs]

    return chosen_centroids, centroids, labels

def generate_instance():
    args = prepare_args()
    cvrp_instance_name = args.cvrp_instance_name
    filename = f"{cvrp_instance_name}.vrp"
    cvrp_problem = read_from_file(filename)
    customers = cvrp_problem.customers
    num_customers = len(customers)
    if args.num_customers>len(customers):
        raise ValueError(f"num-customers must be less than actual number of \
                         customers in original cvrp instance, got {args.num_customers}, expected < {len(customers)}")
    if args.num_customers>0:
        num_customers = args.num_customers
    # randomizing customer types
    # [0,1,2] -> [hd, sp, fx]
    num_sp = math.ceil(args.pickup_ratio*num_customers)
    num_fx = math.ceil(args.flexible_ratio*num_customers)
    num_sp_fx = num_sp + num_fx
    num_hd = num_customers-num_sp_fx
    all_mrt_lockers, all_mrt_lines = generate_mrt_network_soumen(args.min_locker_capacity,args.max_locker_capacity,args.mrt_line_cost)
    cust_coords = np.stack([customer.coord for customer in customers])
    mrt_coords = np.stack([locker.coord for locker in all_mrt_lockers])
    is_cust_chosen = np.zeros([len(customers),], dtype=bool)
    depot_coord = np.asanyarray(cvrp_problem.depot_coord)[None, :]
    distance_to_depot = dm_func(cust_coords, depot_coord)
    epsilon = 1e-6  # avoid div-by-zero
    inv_dist = 1.0 / (distance_to_depot + epsilon)
    hd_probs = inv_dist.ravel()**10
    hd_probs /= hd_probs.sum()
    selected_hd_idxs = np.random.choice(len(customers), size=num_hd, p=hd_probs, replace=False)
    is_cust_chosen[selected_hd_idxs] = True
    unchosen_customers = [customers[i] for i in range(len(customers))]
    distance_from_mrt_to_depot = dm_func(mrt_coords, depot_coord).ravel()
    far_mrt_idxs = np.argpartition(distance_from_mrt_to_depot, -3)[-3:]
    # print(far_mrt_idxs)
    external_lockers: List[Locker] = []
    if args.num_external_lockers>0:
        chosen_centroids, _, _ = select_external_lockers_from_clusters(
            customers=unchosen_customers,
            num_external_lockers=args.num_external_lockers,
            mrt_lockers=[all_mrt_lockers[i] for i in far_mrt_idxs]
        )
        
        locker_capacities = np.random.randint(args.min_locker_capacity, args.max_locker_capacity+1, size=(args.num_external_lockers,))
        for li, coord in enumerate(chosen_centroids):
            new_locker = Locker(0, coord, 10, locker_capacities[li])
            external_lockers.append(new_locker)    
        plt.scatter(chosen_centroids[:, 0], chosen_centroids[:, 1], label="External lockers")
    lockers = all_mrt_lockers + external_lockers
    for li, locker in enumerate(lockers):
        locker.idx = num_customers + li + 1
    locker_coords = np.stack([locker.coord for locker in lockers])
    dist_to_lockers = dm_func(cust_coords, locker_coords)
    dist_to_nearest_locker = dist_to_lockers.min(axis=1)
    masked_dist = dist_to_nearest_locker.copy()
    masked_dist[is_cust_chosen] = np.inf
    inv_dist = 1.0 / (masked_dist + epsilon)
    sp_fx_probs = inv_dist.ravel()**3
    sp_fx_probs /= sp_fx_probs.sum()
    selected_sp_fx_idxs = np.random.choice(len(customers), size=num_sp_fx, p=sp_fx_probs, replace=False)
    shuffle(selected_sp_fx_idxs)
    fx_idxs = selected_sp_fx_idxs[:num_fx]
    sp_idxs = selected_sp_fx_idxs[num_fx:]
    hd_customers: List[Customer] = [customers[i] for i in selected_hd_idxs]
    fx_customers: List[Customer] = [customers[i] for i in fx_idxs]
    sp_customers: List[Customer] = [customers[i] for i in sp_idxs]
    reasonable_radius = 60
    for sp_customer in sp_customers:
        sp_customer.is_self_pickup=True
        distance_to_lockers = dm_func(sp_customer.coord[None, :], locker_coords).ravel()
        num_preferred_lockers = randint(2,4)
        closest_idxs = np.argsort(distance_to_lockers)[:num_preferred_lockers]
        li=1
        for li in range(1, len(closest_idxs)):
            if distance_to_lockers[closest_idxs[li]]>reasonable_radius:
                break
        closest_idxs = closest_idxs[:li]
        # print(len(closest_idxs))
        sp_customer.preferred_locker_idxs = [lockers[i].idx for i in closest_idxs]

    for fx_customer in fx_customers:
        fx_customer.is_flexible = True
        distance_to_lockers = dm_func(fx_customer.coord[None, :], locker_coords).ravel()
        num_preferred_lockers = randint(2,4)
        closest_idxs = np.argsort(distance_to_lockers)[:num_preferred_lockers]
        li=1
        for li in range(1, len(closest_idxs)):
            if distance_to_lockers[closest_idxs[li]]>reasonable_radius:
                break
        closest_idxs = closest_idxs[:li]
        # print(len(closest_idxs))
        fx_customer.preferred_locker_idxs = [lockers[i].idx for i in closest_idxs]
    
    customers = hd_customers + sp_customers + fx_customers
    for ci, customer in enumerate(customers):
        customer.idx = ci+1
    
    vehicles = cvrp_problem.vehicles
    if args.num_vehicles > 0:
        vehicles = []
        for vi in range(args.num_vehicles):
            vehicle = deepcopy(cvrp_problem.vehicles[0])
            vehicle.idx = vi
            vehicles += [vehicle]
    
    lockers = readjust_lockers_capacities(customers, lockers, vehicles[0].capacity)
    plain_name = f"An{len(customers)+1}-k{len(vehicles)}-b{len(external_lockers)}"
    prob = Cvrpptpl(cvrp_problem.depot, 
                    customers,
                    lockers,
                    all_mrt_lines,
                    vehicles,
                    instance_name=plain_name)
    return prob

if __name__ == "__main__":
    problem = generate_instance()
    problem.visualize_graph(savefig=True)

    for num_mrt_lines in range(1,4):
        instance_name = f"A-n{len(problem.customers)+1}-k{len(problem.vehicles)}-m{num_mrt_lines}-b{len(problem.non_mrt_lockers)}"
        problem_copy = deepcopy(problem)
        problem_copy.filename = instance_name
        problem_copy.mrt_lines = problem_copy.mrt_lines[:2*num_mrt_lines]

        # new_problem.visualize_graph()
        problem_copy.save_to_ampl_file(is_v2=True)
        # new_problem.save_to_ampl_file(is_v2=False)
        problem_copy.save_to_file()
        
        if num_mrt_lines == 1:
            instance_name = f"A-n{len(problem.customers)+1}-k{len(problem.vehicles)}-m0-b{len(problem.non_mrt_lockers)}"
            problem.filename = instance_name
            problem.save_to_ampl_file(set_without_mrt=True, is_v2=True)
            # new_problem.save_to_ampl_file(set_without_mrt=True, is_v2=False)
            problem.save_to_file(set_without_mrt=True)
            # new_problem.visualize_graph()    
    