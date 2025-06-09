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
    # parser.add_argument('--locker-cost',
    #                     type=float,
    #                     default=100,
    #                     help='locker cost')

    # parser.add_argument('--mrt-line-cost',
    #                     type=float,
    #                     default=1,
    #                     help='mrt line cost per unit goods')
    
    
    # # vehicles
    # parser.add_argument('--num-vehicles',
    #                     type=int,
    #                     default=0,
    #                     help='0 means use same num vehicles as original, >0 means use this num instead')
    # parser.add_argument('--vehicle-variable-cost',
    #                     type=float,
    #                     default=1,
    #                     help='vehicle cost per unit travelled distance')
    
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

if __name__ == "__main__":
    args = prepare_args()
    # np.random.seed(42)
    # args = prepare_args()
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
    mrt_line_colors = []
    for mrt_station in mrt_stations:
        if mrt_station.line not in line_dict.keys():
            line_dict[mrt_station.line] = len(complete_mrt_lines)
            complete_mrt_lines.append([mrt_station])
            color = get_mrt_line_color(mrt_station.line)
            mrt_line_colors.append(color)
        else:
            i = line_dict[mrt_station.line]
            complete_mrt_lines[i].append(mrt_station)
    
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
    
    gdf = gpd.read_file("taipei.geojson")
    gdf_proj = gdf.to_crs(epsg=3826)
    # print(taipei_df.info())
    gdf["centroid"] = gdf_proj.geometry.centroid
    gdf["latitude"] = gdf.centroid.y
    gdf["longitude"] = gdf.centroid.x
    coords = gdf[["latitude","longitude"]].to_numpy()
    # coords = sample_coords(coords, 200, 1)

    chosen_coords = sample_coords(coords, args.num_customers, args.num_clusters)
    # customers: List[Customer] = []
    
    taipei_store_df = pd.read_csv("taipei_store.csv")
    store_coords = taipei_store_df[["Latitude","Longitude"]].to_numpy()
    
    locker_coords = sample_locker_coords(chosen_coords, store_coords, num_lockers=10)
    external_lockers: List[Locker] = []
    for lcoord in locker_coords:
        ext_locker = Locker(len(lockers), lcoord,
                            15, 
                            randint(50, 100),
                            100)
        lockers.append(ext_locker)
        external_lockers.append(ext_locker)
    
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
            cust_coord = np.radians(chosen_coords[[i]])
            distance_to_lockers = haversine_distances(cust_coord, np.radians(locker_coords))*6371.088
            distance_to_lockers  =distance_to_lockers.flatten()
            # print(distance_to_lockers)
            preferred_lockers = [locker for li, locker in enumerate(lockers) if distance_to_lockers[li] <= args.locker_preference_radius]
            if len(preferred_lockers)>5:
                shuffle(preferred_lockers)
                preferred_lockers = preferred_lockers[:5]
            # print(preferred_lockers)
        
        customer = Customer(i+1,
                            chosen_coords[i],
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
    # print(num_hd, num_sp, num_fx)
    
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot base Taipei map
    gdf.boundary.plot(ax=ax, color="gray", linewidth=0.5)

    # Plot customers
    customer_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in chosen_coords], crs="EPSG:4326")
    customer_gdf.to_crs(epsg=3857).plot(ax=ax, color="blue", markersize=10, label="Customers")

    # Plot lockers
    locker_gdf = gpd.GeoDataFrame(geometry=[Point(locker.coord[1], locker.coord[0]) for locker in external_lockers], crs="EPSG:4326")
    locker_gdf.to_crs(epsg=3857).plot(ax=ax, color="green", markersize=50, marker='s', label="Lockers")

    # Plot MRT terminals
    mrt_terminal_coords = [(s.longitude, s.latitude) for s in terminal_mrt_stations]
    terminal_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in mrt_terminal_coords], crs="EPSG:4326")
    terminal_gdf.to_crs(epsg=3857).plot(ax=ax, color="red", markersize=40, marker='^', label="MRT Terminals")

    # Optionally, plot MRT lines
    for mi, mrt_line in enumerate(complete_mrt_lines):
        coords = []
        for mrt_station in mrt_line:
            coords += [[mrt_station.longitude, mrt_station.latitude]]  # reverse to (lon, lat)
        line = LineString(coords)
        line_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
        line_gdf = line_gdf.to_crs(epsg=3857)
        mrt_line_color = mrt_line_colors[mi]
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