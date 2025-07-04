from random import randint
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_line import MrtLine
from problem.mrt_station import MrtStation
from problem.node import Node
from problem.vehicle import Vehicle
from sklearn.metrics.pairwise import haversine_distances
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
                   row["Position"]
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


def generate_problem_mrt_lines(args, mrt_line_terminals):
    lockers: List[Locker] = []
    mrt_lines: List[MrtLine] = []
    li = 0
    for pair in mrt_line_terminals:
        ma, mb = pair
        locker_a = Locker(li, 
                          np.asanyarray([ma.latitude, ma.longitude]),
                          10,
                          randint(50, 100))
        locker_b = Locker(li+1, 
                          np.asanyarray([mb.latitude, mb.longitude]),
                          10,
                          randint(50, 100))
        li += 2
        lockers += [locker_a, locker_b]
        mrt_lines.append(MrtLine(locker_a, locker_b, 10, args.mrt_line_cost, locker_b.capacity + locker_a.capacity))
        mrt_lines.append(MrtLine(locker_b, locker_a, 10, args.mrt_line_cost, locker_b.capacity + locker_a.capacity))
    return mrt_lines, lockers

def generate_depot(coords)->Node:
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
                            randint(50, 100))
        external_lockers.append(ext_locker)
    return external_lockers

def get_driving_distance_matrix(coords: np.ndarray):
    import requests

    # Use the public OSRM demo server (for production, set up your own OSRM instance)
    OSRM_URL = "http://router.project-osrm.org"
    # Format coordinates for the OSRM API
    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])

    # Construct the API request URL
    url = f"{OSRM_URL}/table/v1/driving/{coord_str}?annotations=distance"

    # Send request
    response = requests.get(url)
    data = response.json()


    # Optional: distances (in meters) if OSRM is built with that option
    if "distances" not in data:
        raise RuntimeError("Distance data not retrieved")
    distance_matrix = np.asanyarray(data["distances"], dtype=float)/1000 #km
    return distance_matrix


if __name__ == "__main__":
    args = prepare_args()
    
    complete_mrt_lines, terminal_mrt_stations, mrt_line_terminals = read_taipei_mrt_stations()
    mrt_lines, mrt_lockers = generate_problem_mrt_lines(args, mrt_line_terminals)
    external_lockers = generate_external_lockers(mrt_lockers, args.num_external_lockers)
    lockers = mrt_lockers + external_lockers
    for li, locker in enumerate(lockers):
        locker.idx = args.num_customers + 1 + li
    
    gdf = gpd.read_file("taipei.geojson")
    gdf_proj = gdf.to_crs(epsg=3826)
    gdf["centroid"] = gdf_proj.geometry.centroid
    gdf["latitude"] = gdf.centroid.y
    gdf["longitude"] = gdf.centroid.x
    coords = gdf[["latitude","longitude"]].to_numpy()
    
    customers = generate_customers(coords, lockers, args.num_customers)
    depot = generate_depot(coords)
    vehicles: List[Vehicle] = []
    for vi in range(args.num_vehicles):
        vehicle = Vehicle(vi, 
                          100,
                          args.vehicle_variable_cost)
        vehicles.append(vehicle)

    all_coords = []
    all_coords.append(depot.coord)
    all_coords.extend([customer.coord for customer in customers])
    all_coords.extend([locker.coord for locker in lockers])
    all_coords = np.asanyarray(all_coords)
    distance_matrix = get_driving_distance_matrix(all_coords)
    # distance_matrix = haversine_distances(np.radians(all_coords))*6371.088
    
    instance_name = f"taipei-n{len(customers)}-k{len(vehicles)}-m{int(len(mrt_lines)/2)}-b{len(external_lockers)}"
    problem = Cvrpptpl(depot,
                       customers,
                       lockers,
                       mrt_lines,
                       vehicles,
                       distance_matrix=distance_matrix,
                       instance_name=instance_name,
                       complete_mrt_lines=complete_mrt_lines)
    # problem.save_to_ampl_file(is_v2=False)
    problem.save_to_ampl_file(is_v2=True)
    problem.save_to_file()
    visualize_taipei_instance(problem, save=True)
    # if num_mrt_lines == 1:
    instance_name = f"taipei-n{len(customers)}-k{len(vehicles)}-m0-b{len(external_lockers)}"
    problem.filename = instance_name
    problem.save_to_ampl_file(set_without_mrt=True, is_v2=True)
    # new_problem.save_to_ampl_file(set_without_mrt=True, is_v2=False)
    problem.save_to_file()
    # new_problem.visualize_graph()    

    # print(depot_coord) 
    # problem = Cvrpptpl(
        
    # )
    # visualize_taipei_instance(problem, complete_mrt_lines)
    # print(num_hd, num_sp, num_fx)
