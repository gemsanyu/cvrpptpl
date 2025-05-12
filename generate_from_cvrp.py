import argparse
import sys
from random import shuffle

import matplotlib.pyplot as plt
import networkx as nx
from problem.cust_locker_assignment import generate_customer_locker_preferences
from problem.cvrp import read_from_file
from problem.cvrpptpl import Cvrpptpl
from problem.locker import generate_lockers
from problem.mrt_line import generate_mrt_network


def prepare_args():
    parser = argparse.ArgumentParser(description='CVRP-PT-PL instance generation')
    
    # args for generating instance based on CVRP problem instances
    parser.add_argument('--cvrp-instance-name',
                        type=str,
                        default="A-n32-k5",
                        help="the cvrp instance name")
    
    
    
    # customers
    parser.add_argument('--pickup-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of flexible customers/number of customers, used to determine number of flexible customers')
    
    
    # depot
    parser.add_argument('--depot-location-mode',
                        type=str,
                        default="c",
                        choices=["c","r"],
                        help='depot\'s location mode, \
                            c: depot in the center of customers \
                            r: randomly scattered')
    
    
    # locker
    parser.add_argument('--num-lockers',
                        type=int,
                        default=6,
                        help='number of lockers')
    parser.add_argument('--locker-capacity-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of total custs\' demand qty that is divided to be lockers\' capacities')
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
    
    
    # mrt
    parser.add_argument('--num-mrt',
                        type=int,
                        default=6,
                        help='number of mrt stations, must be even and smaller than number of lockers')
    parser.add_argument('--freight-capacity-mode',
                        type=str,
                        default="e",
                        choices=["a","e"],
                        help='freight capacity generation mode,\
                            a: ample capacity (10000) \
                            e: enough capacity (U[0.2,0.8]*demands in end station)')
    parser.add_argument('--mrt-line-cost',
                        type=float,
                        default=5,
                        help='mrt line cost per unit goods')
    
    
    # vehicles
    parser.add_argument('--vehicle-cost-reference',
                        type=float,
                        default=0.6,
                        help='vehicle cost reference\
                             vehicle cost will relate to its capacity times this value')
    
    
    args = parser.parse_args(sys.argv[1:])
    return args

def generate(args):
    cvrp_instance_name = args.cvrp_instance_name
    filename = f"{cvrp_instance_name}.vrp"
    cvrp_problem = read_from_file(filename)
    customers = cvrp_problem.customers
    # randomizing customer types
    # [0,1,2] -> [hd, sp, fx]
    num_sp = int(args.pickup_ratio*len(customers))
    num_fx = int(args.flexible_ratio*len(customers))
    num_hd = len(customers)-num_sp-num_fx
    customer_types = [0]*num_hd + [1]*num_sp + [2]*num_fx
    shuffle(customer_types)
    for i, customer in enumerate(customers):
        if customer_types[i]==1:
            customer.is_self_pickup = True
        elif customer_types[i]==2:
            customer.is_flexible = True
    # re-order hd, sp, fx
    customer_types, customers = (list(t) for t in zip(*sorted(zip(customer_types, customers), key=lambda x: x[0])))
    for i, customer in enumerate(customers):
        customer.idx = i+1
        
    min_coord = cvrp_problem.coords.min(axis=0)
    max_coord = cvrp_problem.coords.max(axis=0)
    total_demands = cvrp_problem.demands.sum()
    mrt_lines_1_dicts = {"coordinate_mode": "cross-large","num_mrt_lines": 2}
    # mrt_lines_2_dicts = {"coordinate_mode": "vertical_line-small","num_mrt_lines": 1}
    args_dicts = [mrt_lines_1_dicts, 
                #   mrt_lines_2_dicts
                  ]
    mrt_lockers, mrt_lines = generate_mrt_network(args_dicts, 
                                                  total_demands, 
                                                  min_coord,
                                                  max_coord,
                                                  args.locker_cost, 
                                                  mrt_cost=args.mrt_line_cost)
    num_lockers_left = args.num_lockers - len(mrt_lockers)
    lockers = generate_lockers(num_lockers_left,
                               cvrp_problem.coords,
                               total_demands,
                               args.locker_capacity_ratio,
                               args.locker_cost,
                               args.locker_location_mode)
    
    # add mrt_lines lockers to generated lockers
    lockers = mrt_lockers + lockers
    for i, locker in enumerate(lockers):
        locker.idx = i + len(customers) + 1
    # generating preference matching for customers and lockers
    customers = generate_customer_locker_preferences(customers, lockers)
    cvrpptpl_problem = Cvrpptpl(cvrp_problem.depot,
                                customers,
                                lockers,
                                mrt_lines,
                                cvrp_problem.vehicles,
                                instance_name=cvrp_instance_name)
    cvrpptpl_problem.save_to_ampl_file(is_v2=True)
    cvrpptpl_problem.save_to_ampl_file(is_v2=False)
    cvrpptpl_problem.save_to_file()

if __name__ == "__main__":
    args = prepare_args()
    generate(args)
    