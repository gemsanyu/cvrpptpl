import argparse
import sys
from random import randint, sample, shuffle
from typing import List

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import KMeans, DBSCAN
from shapely.geometry import LineString
import numpy as np

from problem.customer import Customer
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_line import MrtLine
from problem.mrt_station import MrtStation
from taipei_instance_utils import sample_coords, sample_locker_coords, visualize_taipei_instance, prepare_args

def read_taipei_mrt_stations():
    taipei_mrt_df = pd.read_csv("taipei_mrt.csv")
    mrt_stations: List[MrtStation] = [
        MrtStation(float(row["Latitude"]),
                   float(row["Longitude"]),
                   row["Line"],
                   row["Name"],
                   row["Position"]
                   )
        for _, row in taipei_mrt_df.iterrows()
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
                          randint(50, 100),
                          100)
        locker_b = Locker(li+1, 
                          np.asanyarray([mb.latitude, mb.longitude]),
                          10,
                          randint(50, 100),
                          100)
        li += 2
        lockers += [locker_a, locker_b]
        mrt_lines.append(MrtLine(locker_a, locker_b, 10, 1, locker_b.capacity + locker_a.capacity))
        mrt_lines.append(MrtLine(locker_b, locker_a, 10, 1, locker_b.capacity + locker_a.capacity))
    return mrt_lines, lockers

def generate_external_lockers(customer_coords, num_lockers, locker_cost)->List[Locker]:
    taipei_store_df = pd.read_csv("taipei_store.csv")
    store_coords = taipei_store_df[["Latitude","Longitude"]].to_numpy()
    locker_coords = sample_locker_coords(customer_coords, store_coords, num_lockers=num_lockers)
    external_lockers: List[Locker] = []
    for lcoord in locker_coords:
        ext_locker = Locker(len(lockers), lcoord,
                            15, 
                            randint(50, 100),
                            locker_cost)
        lockers.append(ext_locker)
        external_lockers.append(ext_locker)

if __name__ == "__main__":
    args = prepare_args()
    # np.random.seed(42)
    # args = prepare_args()
    
    complete_mrt_lines, terminal_mrt_stations, mrt_line_terminals = read_taipei_mrt_stations()
    mrt_lines, lockers = generate_problem_mrt_lines(args, mrt_line_terminals)
    
    gdf = gpd.read_file("taipei.geojson")
    gdf_proj = gdf.to_crs(epsg=3826)
    gdf["centroid"] = gdf_proj.geometry.centroid
    gdf["latitude"] = gdf.centroid.y
    gdf["longitude"] = gdf.centroid.x
    coords = gdf[["latitude","longitude"]].to_numpy()
    customer_coords = sample_coords(coords, args.num_customers, args.num_clusters)
    external_lockers = generate_external_lockers(args.num_lockers, args.locker_cost)
    
    num_customers_per_type = np.asanyarray([0, int(args.num_customers/3), int(args.num_customers/3)])
    num_customers_per_type[0] = args.num_customers - 2*int(args.num_customers/3)
    customers: List[Customer] = []
    locker_coords = np.asanyarray([locker.coord for locker in lockers])
    for i in range(args.num_customers):
        demand = np.random.randint(3, 30)
        need_customer_of_type = num_customers_per_type>0
        ncot_idxs = np.where(need_customer_of_type)[0]
        t = np.random.choice(3, p=need_customer_of_type/np.sum(need_customer_of_type))
        num_customers_per_type[t]-=1
        preferred_lockers = []
        if t>0:
            cust_coord = np.radians(customer_coords[[i]])
            distance_to_lockers = haversine_distances(cust_coord, np.radians(locker_coords))*6371.088
            distance_to_lockers  =distance_to_lockers.flatten()
            # print(distance_to_lockers)
            preferred_lockers = [locker for li, locker in enumerate(lockers) if distance_to_lockers[li] <= args.locker_preference_radius]
            if len(preferred_lockers)>5:
                shuffle(preferred_lockers)
                preferred_lockers = preferred_lockers[:5]
            # print(preferred_lockers)
        
        customer = Customer(i+1,
                            customer_coords[i],
                            15,
                            demand,
                            t==1,
                            t==2,
                            preferred_lockers)
        customers.append(customer)
        
    num_hd, num_sp, num_fx = 0,0,0
    for customer in customers:
        if customer.is_self_pickup:
            num_sp += 1
        elif customer.is_flexible:
            num_fx += 1
        else:
            num_hd += 1
    
    problem = Cvrpptpl()
    visualize_taipei_instance(problem, complete_mrt_lines)
    # print(num_hd, num_sp, num_fx)
