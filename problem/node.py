import numpy as np

class Node:
    def __init__(self,
                 idx: int,
                 coord: np.ndarray):
        self.idx = idx
        self.coord = coord