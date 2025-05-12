from typing import List, Tuple

import numpy as np
from problem.locker import Locker


class MrtLine:
    def __init__(self,
                 start_station: Locker,
                 end_station: Locker,
                 start_station_service_time:int,
                 cost: int,
                 freight_capacity:int) -> None:
        self.start_station = start_station
        self.end_station = end_station
        self.start_station.service_time = start_station_service_time
        self.cost = cost
        self.freight_capacity = freight_capacity
        
    def __str__(self) -> str:
        return str(self.start_station.idx)+","+str(self.end_station.idx)+","+str(self.freight_capacity)+","+str(self.cost)+"\n"


def generate_mrt_network(args_dicts,
                                total_customer_demand, 
                                min_coord, 
                                max_coord, 
                                locker_cost: float = 1,
                                locker_service_time: int = 15,
                                mrt_service_time: int = 30,
                                mrt_cost:int = 10)->Tuple[List[Locker],List[MrtLine]]:
    lockers: List[Locker] = []
    mrt_lines: List[MrtLine] = []
    for d in args_dicts:
        new_lockers, new_mrt_lines = generate_mrt_lines(d["num_mrt_lines"],
                                           total_customer_demand,
                                           d["coordinate_mode"],
                                           min_coord,
                                           max_coord,
                                           locker_cost,
                                           locker_service_time,
                                           mrt_service_time,
                                           mrt_cost)
        mrt_lines += new_mrt_lines
        lockers += new_lockers
    return lockers, mrt_lines
    

def generate_mrt_lines(num_mrt_lines: int,
                       total_customer_demand: int,
                       coordinate_mode: str,
                       min_coord: np.ndarray,
                       max_coord: np.ndarray,
                       locker_cost: float = 1,
                       locker_service_time: int = 15,
                       mrt_service_time: int = 30,
                       mrt_cost:int = 10):
    """generating mrt lines based on predefined templates
    there are several templates based on coordinate mode

    Args:
        num_mrt_lines (int): _description_
        coordinate_mode (str): it is in the format shape-size
                               shape : vertical_line, horizontal_line, diagonal_line, cross
                               size  : large, small
                               cross must be 2 lines
                               line can be any
        min_coord (np.ndarray): so that the lines are always inside this range
        max_coord (np.ndarray): so that the lines are always inside this range
    """
    locker_capacity = int(0.6 * total_customer_demand)
    freight_capacity = locker_capacity
    coord_1,coord_2 = generate_mrt_coords(num_mrt_lines, coordinate_mode, min_coord, max_coord)
    lockers: List[Locker] = []
    mrt_lines: List[MrtLine] = []
    for i in range(num_mrt_lines):
        locker_1 = Locker(2*i, coord_1[i,:], locker_service_time, locker_capacity, locker_cost)
        locker_2 = Locker(2*i + 1, coord_2[i,:], locker_service_time, locker_capacity, locker_cost)
        mrt_line_1 = MrtLine(locker_1, locker_2, mrt_service_time, mrt_cost, freight_capacity)
        mrt_line_2 = MrtLine(locker_2, locker_1, mrt_service_time, mrt_cost, freight_capacity)
        lockers += [locker_1, locker_2]
        mrt_lines += [mrt_line_1, mrt_line_2]
    return lockers, mrt_lines

def generate_mrt_coords(num_mrt_lines:int,
                        mode:str,
                        min_coord: int,
                        max_coord: int):
    """generating the coords 
        there are several templates based on coordinate mode
    Args:
        num_mrt_lines (int): _description_
        mode (str): it is in the format shape-size
                               shape : vertical_line, horizontal_line, diagonal_line, cross
                               size  : large, small
                               cross is either 1 or 2 lines
                               line can be any
        min_coord (np.ndarray): so that the lines are always inside this range
        max_coord (np.ndarray): so that the lines are always inside this range
    """
    shape, size = mode.split("-")
    xs_1, ys_1, xs_2, ys_2 = None, None, None, None
    if shape == "vertical_line":
        ys_1 = np.zeros([num_mrt_lines, 1], dtype=float)
        xs_1 = np.arange(num_mrt_lines)/num_mrt_lines
        if num_mrt_lines == 1:
            xs_1 = np.asanyarray([[0.5]], dtype=float)
        elif num_mrt_lines == 2:
            xs_1 = np.asanyarray([[0], [1]], dtype=float)
        else:
            r = 1/(num_mrt_lines-1)
            xs_1 = np.arange(num_mrt_lines, dtype=float)*r
            xs_1 = xs_1[:, np.newaxis]
        xs_2 = xs_1.copy()
        ys_2 = np.ones([num_mrt_lines, 1], dtype=float)
    if shape == "horizontal_line":
        xs_1 = np.zeros([num_mrt_lines, 1], dtype=float)
        ys_1 = np.arange(num_mrt_lines)/num_mrt_lines
        if num_mrt_lines == 1:
            ys_1 = np.asanyarray([[0.5]], dtype=float)
        elif num_mrt_lines == 2:
            ys_1 = np.asanyarray([[0], [1]], dtype=float)
        else:
            r = 1/(num_mrt_lines-1)
            ys_1 = np.arange(num_mrt_lines, dtype=float)*r
            ys_1 = ys_1[:, np.newaxis]
        ys_2 = ys_1.copy()
        xs_2 = np.ones([num_mrt_lines, 1], dtype=float)
    if shape == "cross":
        assert num_mrt_lines == 2
        xs_1 = np.asanyarray([[0], [0.9]], dtype=float)
        ys_1 = np.asanyarray([[0], [0]], dtype=float)
        xs_2 = np.asanyarray([[0.9], [0]], dtype=float)
        ys_2 = np.asanyarray([[1], [1]], dtype=float)
            
    scale = 0.85
    if size=="small":
        scale = 0.4
    shift = (1-scale)/2
    xs_1 *= scale
    ys_1 *= scale
    xs_2 *= scale
    ys_2 *= scale
    xs_1 += shift
    ys_1 += shift
    xs_2 += shift
    ys_2 += shift
    
    range_x = max_coord[0]-min_coord[0]
    range_y = max_coord[1]-min_coord[1]
    xs_1 = xs_1*range_x + min_coord[0]
    xs_2 = xs_2*range_x + min_coord[0]
    ys_1 = ys_1*range_y + min_coord[1]
    ys_2 = ys_2*range_y + min_coord[1]
    coord_1 = np.concatenate([xs_1, ys_1], axis=1)
    coord_2 = np.concatenate([xs_2, ys_2], axis=1)
    
    return coord_1, coord_2