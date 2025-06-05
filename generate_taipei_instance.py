import argparse
import sys
from random import randint, sample, shuffle
from typing import List

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from problem.cvrpptpl import Cvrpptpl
from problem.locker import Locker
from problem.mrt_line import MrtLine
from problem.mrt_station import MrtStation
from shapely.geometry import Point
from sklearn.metrics.pairwise import haversine_distances


def prepare_args():
    parser = argparse.ArgumentParser(description='Taipei instance generation')
    parser.add_argument('--num-customers',
                        type=int,
                        help="the number of customers, must be between 1 and number of customers in the original problem instance, \
                            or if set to 0 means it follows the original problem instance")
    
    
    
    # customers
    parser.add_argument('--pickup-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of flexible customers/number of customers, used to determine number of flexible customers')
    
    
    # locker
    parser.add_argument('--num-external-lockers',
                        type=int,
                        default=4,
                        help='number of lockers outside of mrt stations')
    parser.add_argument('--min-locker-capacity',
                        type=int,
                        default=70,
                        help='min range of locker capacity to random')
    parser.add_argument('--max-locker-capacity',
                        type=int,
                        default=100,
                        help='max range of locker capacity to random')
    
    parser.add_argument('--locker-location-mode',
                        type=str,
                        default="c",
                        choices=["c","r","rc"],
                        help='lockers\' location distribution mode. \
                            r: randomly scattered \
                            c: each cluster of customers gets a locker if possible \
                            rc: half clustered half random')
    parser.add_argument('--locker-cost',
                        type=float,
                        default=100,
                        help='locker cost')

    parser.add_argument('--mrt-line-cost',
                        type=float,
                        default=1,
                        help='mrt line cost per unit goods')
    
    
    # vehicles
    parser.add_argument('--num-vehicles',
                        type=int,
                        default=0,
                        help='0 means use same num vehicles as original, >0 means use this num instead')
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=1,
                        help='vehicle cost per unit travelled distance')
    
    args = parser.parse_args(sys.argv[1:])
    return args

def sample_coords(coords_input: np.ndarray, num_samples: int, num_clusters:int=1):
    coords = np.copy(coords_input)
    if num_clusters == 1:
        random_idxs = np.random.choice(coords.shape[0], size=num_samples, replace=False)
        return coords[random_idxs, :]
    
    seed_coord_idxs = np.random.choice(len(coords), num_clusters, replace=False)
    seed_coords = coords[seed_coord_idxs, :]
    coords = np.delete(coords, seed_coord_idxs, axis=0)
    # seed_coords = np.empty((num_clusters,2), dtype=float)
    # for i in range(num_clusters):
    #     if i==0:
    #         seed_coord_idx = np.random.choice(len(coords))
    #         seed_coords[i]=coords[seed_coord_idx]
    #         coords = np.delete(coords, seed_coord_idx, axis=0)
    #         continue

    #     distance_to_seeds = haversine_distances(np.radians(coords), np.radians(seed_coords[:i]))*6371
    #     distance_to_closest_seed = np.min(distance_to_seeds, axis=1)
    #     probs = np.exp(distance_to_closest_seed)/np.sum(np.exp(distance_to_closest_seed))
    #     seed_coord_idx = np.random.choice(len(probs), p=probs)
    #     seed_coords[i] = coords[seed_coord_idx]
    #     coords = np.delete(coords, seed_coord_idx, axis=0)
    # print(haversine_distances(seed_coords))
    
    num_samples -= len(seed_coords)
    all_coords = []
    cluster_tightness = 3
    for i in range(num_samples):
        distance_to_seeds = haversine_distances(np.radians(coords), np.radians(seed_coords))*6371
        weights = np.exp(np.sum(-distance_to_seeds*cluster_tightness, axis=1))
        probs = weights/np.sum(weights)
        selected_coord_idx = np.random.choice(len(probs), p=probs)
        selected_coord = coords[selected_coord_idx]
        coords = np.delete(coords, selected_coord_idx, axis=0)
        all_coords.append(selected_coord)
    all_coords = np.asanyarray(all_coords)
    all_coords = np.concatenate((all_coords, seed_coords), axis=0)
    return all_coords


if __name__ == "__main__":
    np.random.seed(42)
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
    for mrt_station in mrt_stations:
        if mrt_station.line not in line_dict.keys():
            line_dict[mrt_station.line] = len(complete_mrt_lines)
            complete_mrt_lines.append([mrt_station])
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

    chosen_coords = sample_coords(coords, 30, 2)
    # plt.scatter(chosen_coords[:,0], chosen_coords[:, 1])
    # plt.show()

    # Convert coords from (lat, lon) to GeoDataFrame
    geometry = [Point(lon, lat) for lat, lon in chosen_coords]
    gdf_coords = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")  # WGS84

    # Project to Web Mercator for plotting with contextily
    gdf_coords = gdf_coords.to_crs(epsg=3857)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot non-seed coordinates (e.g., the clustered ones) in red
    gdf_coords[:-2].plot(ax=ax, color='red', alpha=0.7, markersize=40, label='Clustered')

    # Plot seed coordinates (e.g., the initial cluster centers) in blue
    gdf_coords[-2:].plot(ax=ax, color='blue', alpha=0.7, markersize=40, label='Seeds')
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)


    ax.set_title("Sampled Coordinates on Real Map (OSM)")
    plt.show()

    # print(gdf[["latitude","longitude"]])

    # problem = Cvrpptpl()
        
    
    
    # print(mrt_stations)