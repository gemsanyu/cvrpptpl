import pathlib

import numpy as np

class VRPTW:
    def __init__(self, 
                 coords: np.ndarray,
                 demands: np.ndarray,
                 time_windows: np.ndarray,
                 service_times: np.ndarray,
                 num_vehicles:int,
                 capacity:int) -> None:
        self.coords = coords
        self.demands = demands
        self.time_windows = time_windows
        self.service_times = service_times
        self.num_nodes = len(coords)
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        
        
def read_instance(instance_name: str)->VRPTW:
    data_all = []
    num_vehicles = 0
    capacity = 0
    instance_dir = pathlib.Path(".")/"instance_sources"
    instance_path = instance_dir/instance_name
    with open(instance_path.absolute(), "r") as instance_file:
        lines = instance_file.readlines()
        for i, line in enumerate(lines):
            if i==4:
                line = line.split()
                num_vehicles = int(line[0])
                capacity = int(line[1])
            if i<9:
                continue
            line = line.split()
            data = [float(v) for v in line]
            data_all.append(data)
    data_all = np.asanyarray(data_all)
    coords = data_all[:, [1,2]]
    demands = data_all[:, 3]
    time_windows = data_all[:, [4,5]]
    service_times = data_all[:, -1]
    return VRPTW(coords, demands, time_windows, service_times, num_vehicles, capacity)
    