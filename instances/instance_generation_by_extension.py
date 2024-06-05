from scipy.special import softmax
from scipy.spatial import distance_matrix as dm_func
import numpy as np

from vrptw import read_instance

def extend_vrptw_instance(vrptw_instance_name:str="solomon_100/rc208.txt",
                            num_customers:int=50,
                            home_delivery_ratio:float=0.4,
                            num_parcel_lockers:int=12,
                            num_mrt_stations:int=6,
                            num_vehicles:int=3):
    vrptw_instance = read_instance(vrptw_instance_name)
    num_nodes = num_customers + num_parcel_lockers + 1
    if num_nodes > vrptw_instance.num_nodes:
        raise ValueError(f"total num nodes cannot be larger than num nodes in instance source\n need {num_nodes} nodes, but only have {vrptw_instance.num_nodes} nodes")
    chosen_idxs = np.random.choice(vrptw_instance.num_nodes, num_nodes)
    chosen_coords = vrptw_instance.coords[chosen_idxs, :]
    distance_matrix = dm_func(chosen_coords, chosen_coords)
    all_idxs = np.arange(num_nodes)
    depot_idx = 0
    cust_idxs = all_idxs[1:num_customers+1]
    cust_demands = vrptw_instance.demands[cust_idxs]
    cust_demands[cust_demands==0] = np.average(cust_demands)
    num_home_delivery_cust = int(home_delivery_ratio*num_customers)
    num_self_pickup_cust = num_customers-num_home_delivery_cust
    home_delivery_cust_idxs = cust_idxs[:num_home_delivery_cust+1]
    self_pickup_cust_idxs = cust_idxs[num_home_delivery_cust+1:]
    locker_idxs = all_idxs[num_customers+2:]
    mrt_idxs = locker_idxs[:num_mrt_stations+1]
    cust_locker_mapping = generate_cl_mapping(distance_matrix, num_customers, self_pickup_cust_idxs, locker_idxs, mode="distance-based")


def generate_cl_mapping(distance_matrix, num_customers, self_pickup_cust_idxs, locker_idxs, mode="distance-based"):
    """generate self pick-up to locker preference/selection mapping

    Args:
        distance_matrix (_type_): _description_
        self_pickup_cust_idxs (_type_): _description_
        locker_idxs (_type_): _description_
        mode (str, optional): options: "closest", "distance-based", "random"
            "closest" means closest locker will automatically be selected
            "distance-based" means sampling from lockers with closer lockers having bigger probability
            "random" means locker chosen randomly (uniform probs)
            . Defaults to "distance-based".
    """
    num_lockers = len(locker_idxs)
    cust_locker_distances = distance_matrix[self_pickup_cust_idxs][locker_idxs]    
    num_self_pickup_cust = len(self_pickup_cust_idxs)
    if mode == "random":
        chosen_locker_idx = np.random.randint(0, num_lockers, size=num_self_pickup_cust)
    if mode == "closest":
        chosen_locker_idx = 
    if mode == "distance-based":
        probability = softmax
        
    cl_mapping = np.zeros([num_customers, num_lockers], dtype=bool)
    cl_mapping[self_pickup_cust_idxs, chosen_locker_idx] = 1
    

if __name__ == "__main__":
    vrpptpl = extend_vrptw_instance()
    