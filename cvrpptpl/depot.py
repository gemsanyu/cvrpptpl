from typing import List

import numpy as np



def generate_depot_coord(customer_coords: np.ndarray, mode="c"):
    if mode == "c":
        min_coord = np.min(customer_coords, axis=0, keepdims=False)
        max_coord = np.max(customer_coords, axis=0, keepdims=False)
        depot_coord = np.floor((min_coord+max_coord)/2)
        depot_coord = depot_coord.astype(int)
    elif mode == "r":
        depot_coord = np.random.randint(low=0, high=1001, size=[2,], dtype=int)
    return depot_coord