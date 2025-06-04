import argparse
import sys
from random import sample, shuffle, randint
from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from problem.mrt_station import MrtStation
from problem.locker import Locker
from problem.mrt_line import MrtLine
from problem.cvrpptpl import Cvrpptpl


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


if __name__ == "__main__":
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
    print(gdf[["latitude","longitude"]])

    # problem = Cvrpptpl()
        
    
    
    # print(mrt_stations)