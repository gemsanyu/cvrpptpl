import numpy as np


def get_coords_in_grid(grid_size: int)-> np.ndarray:
    n = grid_size
    a = np.arange(n*n).reshape([n,n])
    b = a // n
    c = a % n
    coords = np.stack([b,c], axis=-1)
    coords = coords.reshape([n*n,2])
    return coords
