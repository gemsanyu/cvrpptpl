import random

from scipy.special import softmax
from scipy.spatial import distance_matrix as dm_func
import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def get_all_coords(n):
    n = 1001
    a = np.arange(n*n).reshape([n,n])
    b = a // n
    a = a % n
    all_coords = np.stack([b,a], axis=-1)
    all_coords = all_coords.reshape([n*n,2])
    return all_coords


def generate_clusters(num_customers: int,
                      num_clusters: int,
                      cluster_dt: float) -> np.ndarray:
    max_val = 1001
    if num_clusters > num_customers:
        raise ValueError("Number of cluster cannot be larger than number of customers")
    seeds = np.random.randint(0,max_val,size=[num_clusters, 2])
    num_customers -= num_clusters
    all_coords = get_all_coords(n=max_val)
    all_to_seeds_distance = dm_func(all_coords, seeds)
    seed_coord_idxs = seeds[:,0]*max_val + seeds[:, 1]
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
        coords = generate_clusters(num_customers, num_clusters, cluster_dt)
    if mode == "rc":
        num_clustered_customers = int(num_customers/2)
        clustered_coords = generate_clusters(num_clustered_customers, num_clusters, cluster_dt)
        num_random_customers = num_customers-num_clustered_customers
        random_coords = np.random.randint(0,10001,size=[num_random_customers, 2])
        coords = np.concatenate([clustered_coords, random_coords], axis=0)
    return coords

def generate_demands(num_customers: int, customer_coords:np.ndarray, mode: str = "u"):
    if mode == "u":
        demands = np.ones([num_customers,], dtype=int)
    elif mode == "q":
        min_coord = np.min(customer_coords, axis=0, keepdims=True)
        max_coord = np.max(customer_coords, axis=0, keepdims=True)
        mid_coord = (min_coord+max_coord)/2
        is_first_quadrant = np.all(customer_coords>=mid_coord, axis=-1)
        is_third_quadrant = np.all(customer_coords<=mid_coord, axis=-1)
        is_odd_quadrant = np.logical_or(is_first_quadrant, is_third_quadrant)
        is_even_quadrant = np.logical_not(is_odd_quadrant)
        demands = np.zeros([num_customers,], dtype=int)
        demands[is_even_quadrant] = np.random.randint(1, 51, size=[np.sum(is_even_quadrant),])
        demands[is_odd_quadrant] = np.random.randint(51, 101, size=[np.sum(is_odd_quadrant),])
    elif "-" in mode:
        bounds = mode.split(sep="-")
        lb, ub = int(bounds[0]), int(bounds[1])
        demands = np.random.randint(lb, ub+1, size=[num_customers,])
    elif mode == "sl":
        r = random.random()*0.25 + 0.7
        num_small_demands = int(r*num_customers)
        num_large_demands = num_customers-num_small_demands
        small_demands = np.random.randint(1, 11, size=[num_small_demands])
        large_demands = np.random.randint(50,101, size=[num_large_demands,])
        demands = np.concatenate([small_demands, large_demands])
        np.random.shuffle(demands)        
    return demands

def generate_lockers(num_lockers, customer_coords, customer_demands, mode="c", capacity_ratio=0.4):
    """generate lockers coords and their capacities

    Args:
        num_lockers (_type_): _description_
        customer_coords (_type_): _description_
        customer_demands (_type_): _description_
        mode (str, optional):   "c" means generate lockers so that lockers located in clusters
                                "r" means generate lockers randomly on the graph
                                "rc" means generate 1 locker for each cluster, and generate the rest randomly.
                                Defaults to "c".
        capacity_ratio (float, optional): ratio of total demands that become the total capacity of all lockers. Defaults to 0.4.
    """
    max_val = 1001
    all_coords = get_all_coords(n=max_val)
    chosen_idxs = customer_coords[:,0]//max_val + customer_coords[:, 1]%max_val
    locker_coords = np.empty([0,2], dtype=int)
    if mode=="c":
        z = ward(pdist(customer_coords))
        cluster_idxs = fcluster(z, t=500, criterion="distance")
        cluster_idx_unique = np.unique(cluster_idxs)
        cluster_idx = 1
        while len(locker_coords)<num_lockers:
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
    elif mode=="rc":
        z = ward(pdist(customer_coords))
        cluster_idxs = fcluster(z, t=500, criterion="distance")
        cluster_idx_unique = np.unique(cluster_idxs)
        for cluster_idx in cluster_idx_unique:
            if len(locker_coords)>=num_lockers:
                break
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
    num_remaining_lockers = num_lockers- len(locker_coords)
    # if mode == "r":
    if num_remaining_lockers>0:
        logits = np.ones([len(all_coords),])
        logits[chosen_idxs] = -np.inf
        probs = softmax(logits)
        new_locker_idxs = np.random.choice(len(all_coords), size=[num_remaining_lockers,], p=probs)
        new_locker_coords = all_coords[new_locker_idxs, :]
        locker_coords = np.concatenate([locker_coords, new_locker_coords], axis=0)
    
    locker_total_capacity = int(capacity_ratio*np.sum(customer_demands))
    locker_capacity = int(locker_total_capacity/num_lockers)
    locker_capacities = np.full([num_lockers,], locker_capacity, dtype=int)
    return locker_coords, locker_capacities


def generate_mrt_stations(num_mrt_stations, locker_coords, mode="c"):
    if mode == "c":
        mrt_idxs = [i for i in range(num_mrt_stations)]
    if mode == "r":
        mrt_idxs = np.random.choice(len(locker_coords), size=num_mrt_stations, replace=False)
    mrt_adj_list = [[mrt_idxs[i], mrt_idxs[i+1]] for i in range(0, len(mrt_idxs), 2)]
    return mrt_idxs, mrt_adj_list

def generate_depot(customer_coords, mode="c"):
    if mode == "c":
        min_coord = np.min(customer_coords, axis=0, keepdims=True)
        max_coord = np.max(customer_coords, axis=0, keepdims=True)
        depot_coord = np.floor((min_coord+max_coord)/2)
    elif mode == "r":
        depot_coord = np.random.randint(low=0, high=1001, size=[1,2], dtype=int)
    return depot_coord

def split_and_map_customers(customer_coords, demands, self_pickup_ratio, locker_coords, locker_capacities):
    num_customers = len(customer_coords)
    num_sp_customers = int(num_customers*self_pickup_ratio)
    potential_sp_cust_idxs = np.random.choice(num_customers, size=num_sp_customers, replace=False)
    potential_sp_cust_idxs = np.sort(potential_sp_cust_idxs)
    cust_to_locker_distance = dm_func(customer_coords, locker_coords)
    locker_loads = np.zeros_like(locker_capacities)
    num_lockers = len(locker_coords)
    locker_sorted_idxs = []
    for p_sp_idx in potential_sp_cust_idxs:
        dists = cust_to_locker_distance[p_sp_idx, :]
        sorted_idxs = np.argsort(-dists)
        locker_sorted_idxs += [sorted_idxs]
    sp_locker_sorted_idxs = [[i, locker_sorted_idxs[i]] for i in range(num_sp_customers)]
    
    sp_locker_pairs = []
    sp_cust_idxs = []
    for p_sp_locker_sorted_idxs in sp_locker_sorted_idxs:
        p_sp_idx, sorted_idxs = p_sp_locker_sorted_idxs
        for locker_idx in sorted_idxs:
            if locker_loads[locker_idx] + demands[p_sp_idx] > locker_capacities[locker_idx]:
                continue
            locker_loads[locker_idx] += demands[p_sp_idx]
            sp_cust_idxs += [p_sp_idx]
            sp_locker_pairs += [[p_sp_idx, locker_idx]]
            break
        
    num_sp_customers = len(sp_cust_idxs)
    hd_cust_idxs = [i for i in range(num_customers) if i not in sp_cust_idxs]
    hd_cust_idxs = np.asanyarray(hd_cust_idxs)
    sp_cust_idxs = np.asanyarray(sp_cust_idxs)
    sp_locker_pairs = np.asanyarray(sp_locker_pairs)
    return hd_cust_idxs, sp_cust_idxs, sp_locker_pairs
    
def visualize_instance(depot_coord, customer_coords, locker_coords, mrt_idxs, mrt_adj_list, hd_cust_idx, sp_cust_idx, sp_cust_locker_pairs):
    plt.scatter(depot_coord[:, 0], depot_coord[:, 1], marker="s", s=80, label="Depot")
    plt.scatter(customer_coords[:,0], customer_coords[:,1], label="Customers")
    plt.scatter(locker_coords[:,0], locker_coords[:,1], label="Lockers", marker="h", s=70)

    start_mrt_stations = [mrt_pair[0] for mrt_pair in mrt_adj_list]
    end_mrt_stations = [mrt_pair[1] for mrt_pair in mrt_adj_list]
    plt.scatter(locker_coords[start_mrt_stations,0],locker_coords[start_mrt_stations,1], s=100, marker="^", label="Start MRT")
    plt.scatter(locker_coords[end_mrt_stations,0],locker_coords[end_mrt_stations,1], s=100, marker="v", label="End MRT")
    for i, mrt_pair in enumerate(mrt_adj_list):
        mrt_pair_coords = locker_coords[mrt_pair,:]
        if i==0:
            plt.plot(mrt_pair_coords[:, 0], mrt_pair_coords[:, 1], "k--", label="MRT Line")
        else:
            plt.plot(mrt_pair_coords[:, 0], mrt_pair_coords[:, 1], "k--")
    
    # plotting lines of self pickup and their lockers
    for i, sp_locker_pair in enumerate(sp_cust_locker_pairs):
        sp_idx, locker_idx = sp_locker_pair
        sp_coord = customer_coords[sp_idx, :]
        locker_coord = locker_coords[locker_idx, :]
        coords_to_plot = np.stack([sp_coord, locker_coord], axis=0)
        if i == 0:
            plt.plot(coords_to_plot[:, 0], coords_to_plot[:, 1], "r--", label="Customers' pickup route")
        else:
            plt.plot(coords_to_plot[:, 0], coords_to_plot[:, 1], "r--")
        plt.arrow(sp_coord[0], sp_coord[1], (locker_coord[0]-sp_coord[0])*0.9, (locker_coord[1]-sp_coord[1])*0.9, color="red")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    num_customers = 100
    num_lockers = 10
    num_mrt_stations = 6
    self_pickup_ratio = 0.4
    capacity_ratio = 0.6
    customer_coords = generate_customer_coords(num_customers, mode="c", cluster_dt=30)
    demands = generate_demands(num_customers, customer_coords, "sl")
    locker_coords, locker_capacities = generate_lockers(num_lockers, customer_coords, demands, mode="c", capacity_ratio=capacity_ratio)
    mrt_idxs, mrt_adj_list = generate_mrt_stations(num_mrt_stations, locker_coords, mode="c")
    depot_coord = generate_depot(customer_coords, mode="r")
    hd_cust_idx, sp_cust_idx, sp_cust_locker_pairs = split_and_map_customers(customer_coords, demands, self_pickup_ratio, locker_coords, locker_capacities)
    vehicle_costs, vehicle_capacities = generate_vehicles(num_vehicles, demands, cost_reference)
    visualize_instance(depot_coord, customer_coords, locker_coords, mrt_idxs, mrt_adj_list, hd_cust_idx, sp_cust_idx, sp_cust_locker_pairs)
    