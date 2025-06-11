import argparse
import sys
from typing import List
from random import randint

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
from problem.locker import Locker
from problem.cvrpptpl import Cvrpptpl
from problem.mrt_station import MrtStation

def prepare_args():
    parser = argparse.ArgumentParser(description='Taipei instance generation')
    
    
    
    # customers
    parser.add_argument('--num-customers',
                        type=int,
                        help="the number of customers")
    parser.add_argument('--num-clusters',
                        type=int,
                        default=1,
                        help="number of clusters of customers, 1 means no cluster or random")
    
    parser.add_argument('--pickup-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of flexible customers/number of customers, used to determine number of flexible customers')
    
    
    # locker
    parser.add_argument('--locker-preference-radius',
                        type=float,
                        default=10,
                        help='its in kilometers')
    parser.add_argument('--num-external-lockers',
                        type=int,
                        default=2,
                        help='number of lockers outside of mrt stations')
    parser.add_argument('--min-locker-capacity',
                        type=int,
                        default=70,
                        help='min range of locker capacity to random')
    parser.add_argument('--max-locker-capacity',
                        type=int,
                        default=100,
                        help='max range of locker capacity to random')
    
    # parser.add_argument('--locker-location-mode',
    #                     type=str,
    #                     default="c",
    #                     choices=["c","r","rc"],
    #                     help='lockers\' location distribution mode. \
    #                         r: randomly scattered \
    #                         c: each cluster of customers gets a locker if possible \
    #                         rc: half clustered half random')

    # parser.add_argument('--mrt-line-cost',
    #                     type=float,
    #                     default=1,
    #                     help='mrt line cost per unit goods')
    
    
    # # vehicles
    parser.add_argument('--num-vehicles',
                        type=int,
                        default=4,
                        help='0 means use same num vehicles as original, >0 means use this num instead')
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=5,
                        help='vehicle cost per unit travelled distance')
    
    args = parser.parse_args(sys.argv[1:])
    return args

def generate_customers(coords: np.ndarray, lockers:List[Locker], num_customers:int)->List[Customer]:
    customers: List[Customer] = []
    num_hd_customers = int(num_customers/3)
    hd_cust_coord_idxs  = np.random.choice(len(coords), size=num_hd_customers, replace=False)
    hd_cust_coords = coords[hd_cust_coord_idxs]
    for i in range(num_hd_customers):
        demand = randint(3, 30)
        customer = Customer(i, hd_cust_coords[i], 10, demand)
        customers.append(customer)
    num_sp_fx_customers = num_customers-num_hd_customers
    num_sp_customers = int(num_sp_fx_customers/2)
    num_fx_customers = num_sp_fx_customers-num_sp_customers
    num_custers_for_lockers = np.zeros((len(lockers),), dtype=int)
    for i in range(num_sp_fx_customers):
        num_custers_for_lockers[i%len(lockers)]+=1
    
    locker_coords = np.asanyarray([locker.coord for locker in lockers])
    sp_fx_cust_coords = np.empty((0, 2), dtype=float)
    for li, locker in enumerate(lockers):
        locker_coord = locker_coords[[li]]
        distance_to_lockers = haversine_distances(coords, locker_coord).flatten()
        weights = -distance_to_lockers
        probs = np.exp(weights)/np.sum(np.exp(weights))
        chosen_idxs = np.random.choice(len(coords), size=num_custers_for_lockers[li], p=probs, replace=False)
        chosen_coords = coords[chosen_idxs]
        sp_fx_cust_coords = np.concatenate([sp_fx_cust_coords, chosen_coords], axis=0)
    
    np.random.shuffle(sp_fx_cust_coords)
    for i in range(num_sp_fx_customers):
        demand = randint(3, 30)
        preferred_lockers = []
        distance_to_lockers = haversine_distances(sp_fx_cust_coords[[i]], locker_coords).flatten()
        sorted_idxs = np.argsort(distance_to_lockers)
        num_preferred_lockers = randint(2, 3)
        for l in range(num_preferred_lockers):
            locker = lockers[sorted_idxs[l]]
        
        customer = Customer(i, 
                            sp_fx_cust_coords[i],
                            10,
                            demand,
                            i<num_sp_customers,
                            i>=num_sp_customers)

        
    return customer_coords


def sample_locker_coords(mrt_locker_coords: np.ndarray,
                         minimart_coords: np.ndarray,
                         num_lockers: int) -> np.ndarray:
    """Place lockers at minimarts that are inside customer clusters and not too close to each other."""
    all_coords = np.copy(mrt_locker_coords)
    locker_coords = np.empty((0,2), dtype=float)
    for _ in range(num_lockers):
        distance_to_existing_lockers = haversine_distances(minimart_coords, all_coords)
        distance_to_closest_ex_lockers = np.min(distance_to_existing_lockers, axis=1)
        probs = -distance_to_closest_ex_lockers/np.sum(-distance_to_closest_ex_lockers)
        chosen_coord_idx = np.random.choice(len(minimart_coords), p=probs)
        chosen_coord = minimart_coords[[chosen_coord_idx]]
        locker_coords = np.concatenate([locker_coords, chosen_coord])
        all_coords = np.concatenate([all_coords, chosen_coord])
    return locker_coords

def get_mrt_line_color(line_name:str):
    colors = ["blue","red","brown","green","orange"]
    for color in colors:
        if color in line_name.lower():
            return color
        

def visualize_taipei_instance(problem: Cvrpptpl, complete_mrt_lines: List[List[MrtStation]]):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot base Taipei map
    gdf = gpd.read_file("taipei.geojson")
    gdf.boundary.plot(ax=ax, color="gray", linewidth=0.5)

    # Plot customers
    customer_gdf = gpd.GeoDataFrame(geometry=[Point(customer.coord[1], customer.coord[0]) for customer in problem.customers], crs="EPSG:4326")
    customer_gdf.to_crs(epsg=3857).plot(ax=ax, color="blue", markersize=10, label="Customers")

    # Plot lockers
    locker_gdf = gpd.GeoDataFrame(geometry=[Point(locker.coord[1], locker.coord[0]) for locker in problem.non_mrt_lockers], crs="EPSG:4326")
    locker_gdf.to_crs(epsg=3857).plot(ax=ax, color="green", markersize=50, marker='s', label="Lockers")

    # Plot MRT terminals
    mrt_terminal_stations = [mrt_line.start_station for mrt_line in problem.mrt_lines] + [mrt_line.end_station for mrt_line in problem.mrt_lines]
    terminal_gdf = gpd.GeoDataFrame(geometry=[Point(station.coord[1], station.coord[0]) for station in mrt_terminal_stations], crs="EPSG:4326")
    terminal_gdf.to_crs(epsg=3857).plot(ax=ax, color="red", markersize=40, marker='^', label="MRT Terminals")

    # Optionally, plot MRT lines
    for mi, mrt_line in enumerate(complete_mrt_lines):
        coords = []
        for mrt_station in mrt_line:
            coords += [[mrt_station.longitude, mrt_station.latitude]]  # reverse to (lon, lat)
        line = LineString(coords)
        line_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
        line_gdf = line_gdf.to_crs(epsg=3857)
        mrt_line_color = get_mrt_line_color(mrt_line[0].line)
        line_gdf.plot(ax=ax, color=mrt_line_color, linewidth=5, alpha=0.6)

    # Add basemap
     # Combine all plotted points to compute bounding box
    all_gdfs = [customer_gdf, locker_gdf, terminal_gdf]
    all_combined = pd.concat([gdf.to_crs(epsg=3857) for gdf in all_gdfs])
    bounds = all_combined.total_bounds  # [minx, miny, maxx, maxy]

    # # Set plot limits with some padding
    pad = 2000  # in meters
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ctx.add_basemap(ax, crs=customer_gdf.to_crs(epsg=3857).crs)

    # # Style and legend
    ax.set_title("Taipei Instance Visualization", fontsize=16)
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.show()