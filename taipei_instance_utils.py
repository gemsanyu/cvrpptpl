import argparse
import pathlib
import sys
from argparse import Namespace
from random import randint
from typing import List

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import haversine_distances

from problem.customer import Customer
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_station import MrtStation


def prepare_args()->Namespace:
    parser = argparse.ArgumentParser(description='Taipei instance generation')
    
    # customers
    parser.add_argument('--num-customers',
                        type=int,
                        help="the number of customers")
    parser.add_argument('--hd-cust-ratio',
                        type=float,
                        default=1/3,
                        help="ratio of hd customers")
    parser.add_argument('--sp-cust-ratio',
                        type=float,
                        default=1/3,
                        help="ratio of sp customers")   
    # locker
    parser.add_argument('--num-external-lockers',
                        type=int,
                        default=2,
                        help='number of lockers outside of mrt stations')
    
    parser.add_argument('--mrt-line-cost-multiplier',
                        type=float,
                        default=1,
                        help='we multiply this factor to the mrt cost like a discount')
    
    # # vehicles
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=18,
                        help='vehicle cost per unit travelled distance')
    parser.add_argument('--vehicle-capacity',
                        type=int,
                        default=160,
                        help='vehicle capacity')
    
    args = parser.parse_args(sys.argv[1:])
    return args

def generate_customers(coords: np.ndarray, 
                       lockers: List[Locker], 
                       num_hd_customers: int,
                       num_sp_customers: int,
                       num_fx_customers: int)->List[Customer]:
    min_demand = 1
    max_demand = 5
    customers: list[Customer] = []
    is_coords_chosen = np.zeros((len(coords),), dtype=bool)
    num_sp_fx_customers = num_sp_customers + num_fx_customers

    center_taipei_coord = np.asanyarray([[25.0476522, 121.5163016]], dtype=float)
    distance_to_center = (
        haversine_distances(np.radians(center_taipei_coord), np.radians(coords)) * 6371.088
    ).flatten()
    mrt_coords_list = [locker.coord for locker in lockers if locker.is_mrt_station]
    mrt_coords = np.stack(mrt_coords_list)
    distances_to_mrt_stations = (
        haversine_distances(np.radians(mrt_coords), np.radians(coords)) * 6371.088
    ).min(axis=0)

    # Filter coordinates within 10 km radius of Taipei center
    is_not_too_far = distance_to_center < 5
    # is_far_enough_from_mrt_stations = distances_to_mrt_stations > 8
    # potential_hd_coords = coords[np.logical_and(is_not_too_far, is_far_enough_from_mrt_stations)]
    potential_hd_coords = coords[is_not_too_far]
    n_candidates = len(potential_hd_coords)
    if n_candidates < num_hd_customers:
        raise ValueError("Not enough potential HD coordinates within 10 km.")
    chosen_indices:List[int] = []
    shuffled_indices = np.random.permutation(n_candidates)
    min_dist_km = 0.2
    for idx in shuffled_indices:
        coord = potential_hd_coords[idx]

        # Check distance from all previously chosen ones
        if not chosen_indices:
            chosen_indices.append(idx)
            continue

        chosen_coords = potential_hd_coords[chosen_indices]
        dists = haversine_distances(
            np.radians([coord]), np.radians(chosen_coords)
        )[0] * 6371.088  # km

        if np.all(dists >= min_dist_km):
            chosen_indices.append(idx)

        if len(chosen_indices) >= num_hd_customers:
            break

    if len(chosen_indices) < num_hd_customers:
        print(f"⚠️ Only selected {len(chosen_indices)} customers due to spacing constraint.")

    hd_cust_coords = potential_hd_coords[chosen_indices]

    for i, c_idx in enumerate(chosen_indices):
        demand = randint(min_demand, max_demand)
        is_coords_chosen[c_idx] = True
        customer = Customer(i, hd_cust_coords[i], 10, demand)
        customers.append(customer)
    
    # distribute customer counts evenly among lockers
    num_customers_for_lockers = np.zeros((len(lockers),), dtype=int)
    for i in range(num_sp_fx_customers):
        num_customers_for_lockers[i % len(lockers)] += 1

    locker_coords = np.asanyarray([locker.coord for locker in lockers])
    sp_fx_cust_coords = np.empty((0, 2), dtype=float)

    max_dist_to_locker_km = 2
    for li, locker in enumerate(lockers):
        locker_coord = np.asanyarray([locker.coord])
        n_to_sample = num_customers_for_lockers[li]
        if (n_to_sample==0):
            continue

        # Distance from all candidates to this locker
        distance_to_locker = (
            haversine_distances(np.radians(coords), np.radians(locker_coord)).flatten() * 6371.088
        )
        is_close_enough = distance_to_locker < max_dist_to_locker_km
        potential_coords = coords[is_close_enough]
        potential_dists = distance_to_locker[is_close_enough]

        if len(potential_coords) < n_to_sample:
            print(f"⚠️ Locker {li}: Not enough candidates within {max_dist_to_locker_km} km.")
            continue

        # Prefer nearer customers with probability ∝ 1/dist
        weights = np.maximum(max_dist_to_locker_km - potential_dists, 0)
        probs = weights / np.sum(weights)

        chosen_coords_for_locker = []
        attempts = 0

        while len(chosen_coords_for_locker) < n_to_sample and attempts < len(potential_coords) * 3:
            attempts += 1

            # Sample one candidate based on distance weighting
            idx = np.random.choice(len(potential_coords), p=probs)
            candidate = potential_coords[idx]

            # Skip if too close to any already chosen customer (global)
            if len(sp_fx_cust_coords) > 0:
                dists_to_existing = (
                    haversine_distances(
                        np.radians([candidate]),
                        np.radians(sp_fx_cust_coords)
                    )[0] * 6371.088
                )
                if np.any(dists_to_existing < min_dist_km):
                    continue

            # Skip if too close to previously chosen in this locker
            if len(chosen_coords_for_locker) > 0:
                dists_to_local = (
                    haversine_distances(
                        np.radians([candidate]),
                        np.radians(chosen_coords_for_locker)
                    )[0] * 6371.088
                )
                if np.any(dists_to_local < min_dist_km):
                    continue

            chosen_coords_for_locker.append(candidate)

        if len(chosen_coords_for_locker) < n_to_sample:
            print(f"⚠️ Locker {li}: Only got {len(chosen_coords_for_locker)} out of {n_to_sample} due to spacing.")
        print(sp_fx_cust_coords.shape, np.array(chosen_coords_for_locker).shape)
        sp_fx_cust_coords = np.concatenate(
            [sp_fx_cust_coords, np.array(chosen_coords_for_locker)], axis=0
        )
    
    np.random.shuffle(sp_fx_cust_coords)
    for i in range(num_sp_fx_customers):
        demand = randint(min_demand, max_demand)
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
    else:
        plt.show()
        