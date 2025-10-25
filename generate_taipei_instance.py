import copy
import math
from random import randint
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import haversine_distances

from generate_from_cvrp import readjust_lockers_capacities
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_line import MrtLine
from problem.mrt_station import MrtStation
from problem.node import Node
from problem.vehicle import Vehicle
from taipei_instance_utils import (generate_customers, get_mrt_line_color,
                                   prepare_args, sample_locker_coords,
                                   visualize_taipei_instance)


def read_taipei_mrt_stations(included_colors=["red","green","blue"]):
    taipei_mrt_df = pd.read_csv("taipei_mrt.csv")
    mrt_stations: List[MrtStation] = [
        MrtStation(float(row["Latitude"]),
                   float(row["Longitude"]),
                   row["Line"],
                   row["Name"],
                   row["Position"],
                   row["Code"]
                   )
        for _, row in taipei_mrt_df.iterrows()
        if get_mrt_line_color(row["Line"]) in included_colors
    ]
    terminal_mrt_stations = [mrt_station for mrt_station in mrt_stations if mrt_station.position == "Terminal"]
    mrt_line_terminals = []
    for i in range(len(terminal_mrt_stations)-1):
        start_station = terminal_mrt_stations[i]
        for j in range(i+1, len(terminal_mrt_stations)):
            end_station = terminal_mrt_stations[j]
            if start_station.line != end_station.line:
                continue
            terminal_pair = (start_station, end_station)
            mrt_line_terminals.append(terminal_pair)
            break
    complete_mrt_lines = []
    line_dict = {}
    for mrt_station in mrt_stations:
        if mrt_station.line not in line_dict.keys():
            line_dict[mrt_station.line] = len(complete_mrt_lines)
            complete_mrt_lines.append([mrt_station])
        else:
            i = line_dict[mrt_station.line]
            complete_mrt_lines[i].append(mrt_station)
    return complete_mrt_lines, terminal_mrt_stations, mrt_line_terminals


def generate_problem_mrt_lines(mrt_line_terminals:List[Tuple[MrtStation]], 
                               locker_cap_dict: Dict[str, int],
                               mrt_cap_dict: Dict[str, int],
                               mrt_cost_dict: Dict[str, int]) -> Tuple[List[MrtLine], List[Locker]]:
    lockers: List[Locker] = []
    mrt_lines: List[MrtLine] = []
    li = 0
    for (ma, mb) in mrt_line_terminals:
        locker_a = Locker(li, 
                          np.asanyarray([ma.latitude, ma.longitude]),
                          10,
                          locker_cap_dict[ma.code])
        locker_b = Locker(li+1, 
                          np.asanyarray([mb.latitude, mb.longitude]),
                          10,
                          locker_cap_dict[mb.code])
        li += 2
        lockers += [locker_a, locker_b]
        mrt_color = ma.line
        mrt_lines.append(MrtLine(locker_a, locker_b, 10, mrt_cost_dict[mrt_color], mrt_cap_dict[mrt_color]))
        mrt_lines.append(MrtLine(locker_b, locker_a, 10, mrt_cost_dict[mrt_color], mrt_cap_dict[mrt_color]))
    return mrt_lines, lockers

def generate_depot(coords:np.ndarray)->Node:
    center_taipei_coord = np.asanyarray([[25.0476522,121.5163016]], dtype=float)
    distance_to_center = haversine_distances(np.radians(coords), np.radians(center_taipei_coord))*6371.088
    # probs = -distance_to_center/np.sum(-distance_to_center)
    # depot_coord_idx = np.random.choice(len(coords), p=probs.flatten())
    depot_coord_idx = np.argmin(distance_to_center)
    depot_coord = coords[depot_coord_idx]
    depot = Node(0, depot_coord)
    return depot

def generate_external_lockers(mrt_lockers: List[Locker], num_lockers)->List[Locker]:
    taipei_store_df = pd.read_csv("taipei_store.csv")
    store_coords = taipei_store_df[["Latitude","Longitude"]].to_numpy()
    mrt_locker_coords = np.asanyarray([locker.coord for locker in mrt_lockers])
    locker_coords = sample_locker_coords(mrt_locker_coords, store_coords, num_lockers=num_lockers)
    external_lockers: List[Locker] = []
    for lcoord in locker_coords:
        ext_locker = Locker(0, lcoord,
                            10, 
                            randint(20, 60))
        external_lockers.append(ext_locker)
    return external_lockers

def get_driving_distance_matrix(coords: np.ndarray, max_elements: int = 5000) -> np.ndarray:
    """
    Retrieve driving distance matrix from OSRM, chunking requests so that
    each has at most `max_elements` entries (n*n <= max_elements).

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with [lat, lon] or [y, x].
    max_elements : int
        Maximum number of elements (origins * destinations) per OSRM request.

    Returns
    -------
    np.ndarray
        N x N distance matrix in kilometers.
    """

    OSRM_URL = "http://router.project-osrm.org"
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=float)

    # max number of points per chunk
    chunk_size = int(math.floor(math.sqrt(max_elements)))
    if chunk_size < 2:
        raise ValueError("max_elements too small; must allow at least 2x2 matrix")

    for i_start in range(0, n, chunk_size):
        for j_start in range(0, n, chunk_size):
            i_end = min(i_start + chunk_size, n)
            j_end = min(j_start + chunk_size, n)

            origins = coords[i_start:i_end]
            destinations = coords[j_start:j_end]

            # OSRM expects lon,lat order
            coord_str_orig = ";".join(f"{lon},{lat}" for lat, lon in origins)
            coord_str_dest = ";".join(f"{lon},{lat}" for lat, lon in destinations)

            # OSRM /table API supports `sources` and `destinations` indices
            url = (
                f"{OSRM_URL}/table/v1/driving/{coord_str_orig};{coord_str_dest}"
                f"?sources={';'.join(map(str, range(len(origins))))}"
                f"&destinations={';'.join(map(str, range(len(origins), len(origins) + len(destinations))))}"
                f"&annotations=distance"
            )

            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "distances" not in data:
                raise RuntimeError(f"No 'distances' in OSRM response: {data}")

            block = np.asanyarray(data["distances"], dtype=float) / 1000.0  # km
            distance_matrix[i_start:i_end, j_start:j_end] = block

    return distance_matrix

def generate_problem(args)->Cvrpptpl:
    locker_cap_dict:Dict[str,int] = {
        "R01": 30,
        "R28": 120,
        "BL01": 30,
        "BL23": 70,
        "G01": 30,
        "G19":50,
    }
    mrt_cap_dict:Dict[str,int] = {
        "Blue Line":100,
        "Red Line":80,
        "Green Line":80
    }
    mrt_cost_dict:Dict[str,float] = {
        "Blue Line":65*args.mrt_line_cost_multiplier,
        "Red Line":65*args.mrt_line_cost_multiplier,
        "Green Line":55*args.mrt_line_cost_multiplier
    }
    complete_mrt_lines, terminal_mrt_stations, mrt_line_terminals = read_taipei_mrt_stations()
    mrt_lines, mrt_lockers = generate_problem_mrt_lines(mrt_line_terminals,
                                                        locker_cap_dict,
                                                        mrt_cap_dict,
                                                        mrt_cost_dict)
    num_external_lockers = 1 + int(math.ceil(args.num_customers/10.))
    external_lockers = generate_external_lockers(mrt_lockers, num_external_lockers)
    lockers = mrt_lockers + external_lockers
    for li, locker in enumerate(lockers):
        locker.idx = args.num_customers + 1 + li
    
    gdf = gpd.read_file("taipei.geojson")
    gdf_proj = gdf.to_crs(epsg=3826)
    gdf["centroid"] = gdf_proj.geometry.centroid
    gdf["latitude"] = gdf.centroid.y
    gdf["longitude"] = gdf.centroid.x
    coords = gdf[["latitude","longitude"]].to_numpy()
    
    num_customers = args.num_customers
    num_hd_customers = int(num_customers*args.hd_cust_ratio)
    num_sp_customers = int(num_customers*args.sp_cust_ratio)
    num_fx_customers = num_customers - num_hd_customers - num_sp_customers

    customers = generate_customers(coords, 
                                   lockers, 
                                   num_hd_customers,
                                   num_sp_customers,
                                   num_fx_customers)
    depot = generate_depot(coords)
    vehicles: List[Vehicle] = []
    
    vehicle_capacity = args.vehicle_capacity
    total_demand = sum([customer.demand for customer in customers])
    num_vehicles = int(math.ceil(total_demand/(vehicle_capacity))) + 1

    for vi in range(num_vehicles):
        vehicle = Vehicle(vi, 
                          vehicle_capacity,
                          args.vehicle_variable_cost)
        vehicles.append(vehicle)

    all_coords = []
    all_coords.append(depot.coord)
    all_coords.extend([customer.coord for customer in customers])
    all_coords.extend([locker.coord for locker in lockers])
    all_coords = np.asanyarray(all_coords)
    distance_matrix = get_driving_distance_matrix(all_coords)
    print("distance matrix computed")
    # distance_matrix = haversine_distances(np.radians(all_coords))*6371.088
    
    instance_name = f"taipei-n{len(customers)}-k{len(vehicles)}-m{int(len(mrt_lines)/2)}-b{len(external_lockers)}"
    lockers = readjust_lockers_capacities(customers, lockers, vehicle_capacity)
    problem = Cvrpptpl(depot,
                       customers,
                       lockers,
                       mrt_lines,
                       vehicles,
                       distance_matrix=distance_matrix,
                       instance_name=instance_name,
                       complete_mrt_lines=complete_mrt_lines)
    return problem
    

if __name__ == "__main__":
    args = prepare_args()
    problem = generate_problem(args)
    visualize_taipei_instance(problem, save=True)
    
    for num_mrt_lines in range(1,4):
        instance_name = f"taipei-n{len(problem.customers)+1}-k{len(problem.vehicles)}-m{num_mrt_lines}-b{len(problem.non_mrt_lockers)}"
        problem_copy = copy.deepcopy(problem)
        problem_copy.filename = instance_name
        problem_copy.mrt_lines = problem_copy.mrt_lines[:2*num_mrt_lines]

        problem_copy.save_to_ampl_file(is_v2=True)
        problem_copy.save_to_file()
        
        if num_mrt_lines == 1:
            instance_name = f"taipei-n{len(problem.customers)+1}-k{len(problem.vehicles)}-m0-b{len(problem.non_mrt_lockers)}"
            problem_copy.filename = instance_name
            problem_copy.save_to_ampl_file(set_without_mrt=True, is_v2=True)
            problem_copy.save_to_file(set_without_mrt=True)
        