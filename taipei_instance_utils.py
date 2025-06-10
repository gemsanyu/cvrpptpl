import argparse
import sys
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
    # parser.add_argument('--num-external-lockers',
    #                     type=int,
    #                     default=4,
    #                     help='number of lockers outside of mrt stations')
    # parser.add_argument('--min-locker-capacity',
    #                     type=int,
    #                     default=70,
    #                     help='min range of locker capacity to random')
    # parser.add_argument('--max-locker-capacity',
    #                     type=int,
    #                     default=100,
    #                     help='max range of locker capacity to random')
    
    # parser.add_argument('--locker-location-mode',
    #                     type=str,
    #                     default="c",
    #                     choices=["c","r","rc"],
    #                     help='lockers\' location distribution mode. \
    #                         r: randomly scattered \
    #                         c: each cluster of customers gets a locker if possible \
    #                         rc: half clustered half random')
    parser.add_argument('--locker-cost',
                        type=float,
                        default=0,
                        help='locker cost')

    # parser.add_argument('--mrt-line-cost',
    #                     type=float,
    #                     default=1,
    #                     help='mrt line cost per unit goods')
    
    
    # # vehicles
    # parser.add_argument('--num-vehicles',
    #                     type=int,
    #                     default=0,
    #                     help='0 means use same num vehicles as original, >0 means use this num instead')
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=5,
                        help='vehicle cost per unit travelled distance')
    
    args = parser.parse_args(sys.argv[1:])
    return args

def sample_coords(coords_input: np.ndarray, num_samples: int, num_clusters:int=1):
    coords = np.copy(coords_input)
    if num_clusters == 1:
        random_idxs = np.random.choice(coords.shape[0], size=num_samples, replace=False)
        return coords[random_idxs, :]

    # KMeans clustering
    kmeans = KMeans(n_clusters=20)
    labels = kmeans.fit_predict(coords)
    
    chosen_labels = np.random.choice(np.unique(labels), size=num_clusters, replace=False)
    num_coords_in_clusters = np.zeros((num_clusters), dtype=int)
    
    for i in range(num_samples):
        num_coords_in_clusters[i%len(chosen_labels)] += 1
    chosen_coords_list = []
    for ci in range(num_clusters):   
        label = chosen_labels[ci]
        coords_in_cluster = coords[labels==label]
        chosen_idxs = np.random.choice(len(coords_in_cluster), size=num_coords_in_clusters[ci])
        selected_coords = coords_in_cluster[chosen_idxs]
        chosen_coords_list += [selected_coords]
    chosen_coords = np.concatenate(chosen_coords_list, axis=0)
    return chosen_coords



def sample_locker_coords(customer_coords: np.ndarray,
                         minimart_coords: np.ndarray,
                         num_lockers: int) -> np.ndarray:
    """Place lockers at minimarts that are inside customer clusters and not too close to each other."""

    locker_coords = np.empty((num_lockers, 2))
    kms_per_radian = 6371.0088
    customer_coords_rad = np.radians(customer_coords)
    minimart_coords_rad = np.radians(minimart_coords)

    # --- Cluster customers ---
    db = DBSCAN(eps=2 / kms_per_radian,  # ~2km
                min_samples=2,
                algorithm='ball_tree',
                metric='haversine').fit(customer_coords_rad)
    labels = db.labels_

    # Filter valid clusters (exclude noise)
    valid_labels = np.unique(labels[labels != -1])
    if len(valid_labels) == 0:
        raise ValueError("No valid clusters found.")

    # Get all customer points in clusters
    clustered_customer_coords = customer_coords[np.isin(labels, valid_labels)]

    # For each minimart, check if it's inside any cluster (within eps radius from any clustered customer)
    distances = haversine_distances(minimart_coords_rad, np.radians(clustered_customer_coords)) * kms_per_radian
    within_cluster_mask = (distances < 10).any(axis=1)  # at least one customer within 2km

    # Filter only those minimarts inside clusters
    candidate_minimarts = minimart_coords[within_cluster_mask]
    if len(candidate_minimarts) < num_lockers:
        raise ValueError(f"Not enough minimarts inside customer clusters. Needed {num_lockers}, found {len(candidate_minimarts)}.")

    candidate_minimarts_rad = np.radians(candidate_minimarts)

    # --- Compute cluster centroids ---
    centroids = np.array([
        customer_coords[labels == label].mean(axis=0)
        for label in valid_labels
    ])

    # --- Place lockers ---
    for i in range(num_lockers):
        cluster_idx = i % len(centroids)
        centroid = centroids[[cluster_idx]]

        # Distance to centroid
        dist_to_centroid = haversine_distances(candidate_minimarts_rad, np.radians(centroid)).flatten() * kms_per_radian

        if i == 0:
            weights = np.exp(-dist_to_centroid)
        else:
            dist_to_lockers = haversine_distances(candidate_minimarts_rad, np.radians(locker_coords[:i])) * kms_per_radian
            closest_dist = dist_to_lockers.min(axis=1)
            weights = np.exp(-dist_to_centroid) * np.exp(closest_dist / 2)

        probs = weights / weights.sum()
        chosen_idx = np.random.choice(len(candidate_minimarts), p=probs)
        locker_coords[i] = candidate_minimarts[chosen_idx]

        # Optional: remove chosen minimart from future choices
        candidate_minimarts = np.delete(candidate_minimarts, chosen_idx, axis=0)
        candidate_minimarts_rad = np.radians(candidate_minimarts)

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