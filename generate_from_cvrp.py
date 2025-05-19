import argparse
import sys
from random import sample, shuffle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from problem.cust_locker_assignment import generate_customer_locker_preferences
from problem.cvrp import read_from_file
from problem.cvrpptpl import Cvrpptpl
from problem.locker import generate_lockers_v2
from problem.mrt_line import generate_mrt_network_soumen


def prepare_args():
    parser = argparse.ArgumentParser(description='CVRP-PT-PL instance generation')
    
    # args for generating instance based on CVRP problem instances
    parser.add_argument('--cvrp-instance-name',
                        type=str,
                        default="A-n32-k5",
                        help="the cvrp instance name")
    
    parser.add_argument('--num-customers',
                        type=int,
                        default=0,
                        help="the number of customers, must be between 1 and number of customers in the original problem instance, \
                            or if set to 0 means it follows the original problem instance")
    
    
    
    # customers
    parser.add_argument('--pickup-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=0.1,
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
    
    
    # mrt
    parser.add_argument('--num-mrt-lines',
                        type=int,
                        default=1,
                        help='number of mrt stations, must be even and smaller than number of lockers')
    
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

def generate(args):
    cvrp_instance_name = args.cvrp_instance_name
    filename = f"{cvrp_instance_name}.vrp"
    cvrp_problem = read_from_file(filename)
    customers = cvrp_problem.customers
    if args.num_customers >len(customers):
        raise ValueError(f"num-customers must be less than actual number of \
                         customers in original cvrp instance, got {args.num_customers}, expected < {len(customers)}")
    if args.num_customers > 0:
        customers = sample(customers, args.num_customers)

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
        
    mrt_lockers, mrt_lines = generate_mrt_network_soumen(args.num_mrt_lines,
                                                         args.locker_cost,
                                                         args.min_locker_capacity,
                                                         args.max_locker_capacity,
                                                         args.mrt_line_cost)
    customer_coords = np.asanyarray([customer.coord for customer in customers])
    lockers = generate_lockers_v2(args.num_external_lockers,
                               customer_coords,
                               args.min_locker_capacity,
                               args.max_locker_capacity,
                               args.locker_cost,
                               args.locker_location_mode)
    
    # add mrt_lines lockers to generated lockers
    lockers = mrt_lockers + lockers
    for i, locker in enumerate(lockers):
        locker.idx = i + len(customers) + 1
    # generating preference matching for customers and lockers
    customers = generate_customer_locker_preferences(customers, lockers)
    
    
    vehicles = cvrp_problem.vehicles
    if args.num_vehicles>0:
        vehicles = vehicles[:args.num_vehicles]
        
    instance_name = cvrp_instance_name
    if args.num_customers>0:
        instance_name = f"A-n{len(customers)}-k{len(vehicles)}"
    cvrpptpl_problem = Cvrpptpl(cvrp_problem.depot,
                                customers,
                                lockers,
                                mrt_lines,
                                vehicles,
                                instance_name=instance_name)
    # cvrpptpl_problem.visualize_graph()
    cvrpptpl_problem.save_to_ampl_file(is_v2=True)
    # cvrpptpl_problem.save_to_ampl_file(is_v2=False)
    cvrpptpl_problem.save_to_file()

if __name__ == "__main__":
    args = prepare_args()
    generate(args)
    