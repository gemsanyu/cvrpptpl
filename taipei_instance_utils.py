import argparse
import pathlib
import sys
from random import randint
from typing import List

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from problem.customer import Customer
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_station import MrtStation
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import haversine_distances


def prepare_args():
    parser = argparse.ArgumentParser(description='Taipei instance generation')
    
    
    
    # customers
    parser.add_argument('--num-customers',
                        type=int,
                        help="the number of customers")
   
    
    # locker
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

    parser.add_argument('--mrt-line-cost',
                        type=float,
                        default=0.1,
                        help='mrt line cost per unit goods')
    
    
    # # vehicles
    parser.add_argument('--num-vehicles',
                        type=int,
                        default=5,
                        help='0 means use same num vehicles as original, >0 means use this num instead')
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=3,
                        help='vehicle cost per unit travelled distance')
    
    args = parser.parse_args(sys.argv[1:])
    return args

def generate_customers(coords: np.ndarray, lockers:List[Locker], num_customers:int)->List[Customer]:
    customers: List[Customer] = []
    num_hd_customers = int(num_customers/3)
    center_taipei_coord = np.asanyarray([[25.0476522,121.5163016]], dtype=float)
    distance_to_center = (haversine_distances(np.radians(center_taipei_coord), np.radians(coords))*6371.088).flatten()
    is_not_too_far = distance_to_center<10
    potential_hd_coords = coords[is_not_too_far]
    hd_cust_coord_idxs  = np.random.choice(len(potential_hd_coords), size=num_hd_customers, replace=False)
    hd_cust_coords = potential_hd_coords[hd_cust_coord_idxs]
    for i in range(num_hd_customers):
        demand = randint(3, 30)
        customer = Customer(i, hd_cust_coords[i], 10, demand)
        customers.append(customer)
    num_sp_fx_customers = num_customers-num_hd_customers
    num_sp_customers = int(num_sp_fx_customers/2)
    num_fx_customers = num_sp_fx_customers-num_sp_customers
    num_customerrs_for_lockers = np.zeros((len(lockers),), dtype=int)
    for i in range(num_sp_fx_customers):
        num_customerrs_for_lockers[i%len(lockers)]+=1
    locker_coords = np.asanyarray([locker.coord for locker in lockers])
    sp_fx_cust_coords = np.empty((0, 2), dtype=float)
    for li, locker in enumerate(lockers):
        locker_coord = locker_coords[[li]]
        distance_to_lockers = haversine_distances(np.radians(coords), np.radians(locker_coord)).flatten()*6371.088
        # print(distance_to_lockers)
        is_close_enough = distance_to_lockers<2
        potential_coords = coords[is_close_enough]
        distance_to_lockers = distance_to_lockers[is_close_enough]
        weights = -distance_to_lockers
        probs = weights/np.sum(weights)
        # probs = np.exp(weights)/np.sum(np.exp(weights))
        chosen_idxs = np.random.choice(len(potential_coords), size=num_customerrs_for_lockers[li], p=probs, replace=False)
        chosen_coords = potential_coords[chosen_idxs]
        sp_fx_cust_coords = np.concatenate([sp_fx_cust_coords, chosen_coords], axis=0)
    
    np.random.shuffle(sp_fx_cust_coords)
    for i in range(num_sp_fx_customers):
        demand = randint(3, 30)
        preferred_lockers_idx = []
        distance_to_lockers = haversine_distances(np.radians(sp_fx_cust_coords[[i]]), np.radians(locker_coords)).flatten()*6371.088
        sorted_idxs = np.argsort(distance_to_lockers)
        num_preferred_lockers = randint(2, 3)
        for l in range(num_preferred_lockers):
            locker = lockers[sorted_idxs[l]]
            distance = distance_to_lockers[sorted_idxs[l]]
            # print(distance, locker.idx)
            if distance < 3:
                preferred_lockers_idx.append(locker.idx)
        # exit()
        is_self_pickup = i<num_sp_customers
        is_flexible = i>=num_sp_customers
        customer = Customer(i, 
                            sp_fx_cust_coords[i],
                            10,
                            demand,
                            is_self_pickup,
                            is_flexible,
                            preferred_lockers_idx)
        customers.append(customer)
    for ci, customer in enumerate(customers):
        customer.idx = ci+1
        
    return customers


def sample_locker_coords(mrt_locker_coords: np.ndarray,
                         minimart_coords: np.ndarray,
                         num_lockers: int) -> np.ndarray:
    """Place lockers at minimarts that are inside customer clusters and not too close to each other."""
    all_coords = np.copy(mrt_locker_coords)
    locker_coords = np.empty((0,2), dtype=float)
    for _ in range(num_lockers):
        distance_to_existing_lockers = haversine_distances(np.radians(minimart_coords), np.radians(all_coords))
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
        

def visualize_taipei_instance(problem: Cvrpptpl, save=False):
    fig, ax = plt.subplots(figsize=(12, 12))
    complete_mrt_lines = problem.complete_mrt_lines

    # Plot base Taipei map
    gdf = gpd.read_file("taipei.geojson")
    gdf.boundary.plot(ax=ax, color="gray", linewidth=0.5)

    # Plot customers
    # home deliveries
    hd_customer_gdf = gpd.GeoDataFrame(geometry=[Point(customer.coord[1], customer.coord[0]) for customer in problem.customers if not (customer.is_flexible or customer.is_self_pickup)], crs="EPSG:4326")
    hd_customer_gdf.to_crs(epsg=3857).plot(ax=ax, color="blue", markersize=30, label="Home delivery customers")
    # self pickups
    customer_gdf = gpd.GeoDataFrame(geometry=[Point(customer.coord[1], customer.coord[0]) for customer in problem.customers if customer.is_self_pickup], crs="EPSG:4326")
    customer_gdf.to_crs(epsg=3857).plot(ax=ax, color="red", markersize=30, label="Self-pickup customers")
    # flexibles
    customer_gdf = gpd.GeoDataFrame(geometry=[Point(customer.coord[1], customer.coord[0]) for customer in problem.customers if customer.is_flexible], crs="EPSG:4326")
    customer_gdf.to_crs(epsg=3857).plot(ax=ax, color="yellow", markersize=30, label="Flexible customers")
    
    # Plot lockers
    locker_gdf = gpd.GeoDataFrame(geometry=[Point(locker.coord[1], locker.coord[0]) for locker in problem.non_mrt_lockers], crs="EPSG:4326")
    locker_gdf.to_crs(epsg=3857).plot(ax=ax, color="green", markersize=50, marker='s', label="Lockers")

    # Plot MRT terminals
    mrt_terminal_stations = [mrt_line.start_station for mrt_line in problem.mrt_lines] + [mrt_line.end_station for mrt_line in problem.mrt_lines]
    terminal_gdf = gpd.GeoDataFrame(geometry=[Point(station.coord[1], station.coord[0]) for station in mrt_terminal_stations], crs="EPSG:4326")
    terminal_gdf.to_crs(epsg=3857).plot(ax=ax, color="red", markersize=40, marker='^', label="MRT Terminals")

    # plot MRT lines
    for mi, mrt_line in enumerate(complete_mrt_lines):
        coords = []
        for mrt_station in mrt_line:
            coords += [[mrt_station.longitude, mrt_station.latitude]]  # reverse to (lon, lat)
        line = LineString(coords)
        line_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
        line_gdf = line_gdf.to_crs(epsg=3857)
        mrt_line_color = get_mrt_line_color(mrt_line[0].line)
        line_gdf.plot(ax=ax, color=mrt_line_color, linewidth=5, alpha=0.6, label="MRT lines")

    # plot
    locker_preference_lines = [] 
    for customer in problem.customers:
        cust_coord = customer.coord        
        for l_idx in customer.preferred_locker_idxs:
            locker= problem.lockers[l_idx-problem.num_customers-1]
            locker_coord = locker.coord
            # print(locker.idx, locker_coord)
            line = LineString([[cust_coord[1], cust_coord[0]], [locker_coord[1],locker_coord[0]]])
            locker_preference_lines.append(line)
    lines_gdf = gpd.GeoDataFrame(geometry=locker_preference_lines, crs="EPSG:4326")
    lines_gdf = lines_gdf.to_crs(epsg=3857)
    lines_gdf.plot(ax=ax, color="red", linestyle="--", linewidth=1, label="Locker preference")

    # plot the depot
    depot_gdf = gpd.GeoDataFrame(geometry=[Point(problem.depot.coord[1], problem.depot.coord[0])], crs="EPSG:4326")
    depot_gdf.to_crs(epsg=3857).plot(ax=ax, marker="^", color="black", markersize=100, label="Depot")

    # Add basemap
     # Combine all plotted points to compute bounding box
    all_gdfs = [customer_gdf, locker_gdf, terminal_gdf, depot_gdf]
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
    if save:
        figure_dir = pathlib.Path()/"instances"/"figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figure_dir/f"{problem.filename}.pdf"
        plt.savefig(figure_path.absolute(), dpi=300, bbox_inches="tight")  # high-res image
    # plt.show()
        