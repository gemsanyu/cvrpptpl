import argparse
from copy import deepcopy
import sys
from random import sample, shuffle, random
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance_matrix as dm_func

from problem.customer import Customer
from problem.cust_locker_assignment import generate_customer_locker_preferences
from problem.cvrp import read_from_file
from problem.cvrpptpl import Cvrpptpl
from problem.locker import generate_lockers_v2, Locker
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
                        default=1/3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=1/3,
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
    
    # mrt
    parser.add_argument('--mrt-line-cost',
                        type=float,
                        default=0.5,
                        help='mrt line cost per unit goods')
    
    
    # vehicles
    parser.add_argument('--num-vehicles',
                        type=int,
                        default=0,
                        help='0 means use same num vehicles as original, >0 means use this num instead')
    parser.add_argument('--vehicle-variable-cost',
                        type=float,
                        default=3,
                        help='vehicle cost per unit travelled distance')
    
    args = parser.parse_args(sys.argv[1:])
    return args

def generate_basic_instance(args)->Cvrpptpl:
    cvrp_instance_name = args.cvrp_instance_name
    filename = f"{cvrp_instance_name}.vrp"
    cvrp_problem = read_from_file(filename)
    customers = cvrp_problem.customers
    if args.num_customers>len(customers):
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
        
    
    customer_coords = np.asanyarray([customer.coord for customer in customers])
    lockers = generate_lockers_v2(args.num_external_lockers,
                               customer_coords,
                               args.min_locker_capacity,
                               args.max_locker_capacity,
                               args.locker_location_mode)
    
    # add mrt_lines lockers to generated lockers
    # lockers = mrt_lockers + lockers
    for i, locker in enumerate(lockers):
        locker.idx = i + len(customers) + 1
    # generating preference matching for customers and lockers
    customers = generate_customer_locker_preferences(customers, lockers)
    
    vehicles = cvrp_problem.vehicles
    if args.num_vehicles>0:
        vehicles = vehicles[:args.num_vehicles]
    for vehicle in vehicles:
        vehicle.cost = args.vehicle_variable_cost
    cvrpptpl_problem = Cvrpptpl(cvrp_problem.depot,
                                customers,
                                lockers,
                                [],
                                vehicles,
                                instance_name="AA")
    return cvrpptpl_problem

def add_mrt_lockers_to_preference(new_customers: List[Customer], mrt_lockers: List[Locker]) -> List[Customer]:
    if len(mrt_lockers)==0:
        return new_customers
    cust_coords = np.asanyarray([cust.coord for cust in new_customers])
    locker_coords = np.asanyarray([locker.coord for locker in mrt_lockers])
    dist_custs_to_lockers = dm_func(cust_coords, locker_coords)
    all_coords = [cust.coord for cust in new_customers] + [locker.coord for locker in mrt_lockers]
    all_coords = np.stack(all_coords)
    min_coord, max_coord = all_coords.min(axis=0), all_coords.max(axis=0)
    diag_range = np.linalg.norm(min_coord-max_coord)
    for ci, customer in enumerate(new_customers):
        if not(customer.is_self_pickup or customer.is_flexible):
            continue
        dist_to_lockers = dist_custs_to_lockers[ci,:].flatten()
        sorted_idxs = np.argsort(dist_to_lockers)
        sorted_dist_to_lockers = dist_to_lockers[sorted_idxs]
        # has a 50% chance to include the closest mrt lockers even if outside 
        # reasonable radius
        closest_dist = sorted_dist_to_lockers[0]
        if closest_dist/diag_range <= 0.2:
            customer.preferred_locker_idxs.append(sorted_idxs[0])
        elif random()<=0.5:
            customer.preferred_locker_idxs.append(sorted_idxs[0])
    return new_customers

if __name__ == "__main__":
    args = prepare_args()
    basic_problem = generate_basic_instance(args)
    
    
    instance_name = f"A-n{len(basic_problem.customers)}-k{len(basic_problem.vehicles)}-m{3}-b{len(basic_problem.non_mrt_lockers)}"
    # cvrpptpl_problem = Cvrpptpl(basic_problem.depot,
    #                         basic_problem.customers,
    #                         new_lockers,
    #                         mrt_lines,
    #                         basic_problem.vehicles,
    #                         instance_name=instance_name)
    for num_mrt_lines in range(4):
        instance_name = f"A-n{len(basic_problem.customers)}-k{len(basic_problem.vehicles)}-m{num_mrt_lines}-b{len(basic_problem.non_mrt_lockers)}"
        new_problem: Cvrpptpl
        mrt_lockers, mrt_lines = generate_mrt_network_soumen(max(1, num_mrt_lines),args.min_locker_capacity,args.max_locker_capacity,args.mrt_line_cost)
        for locker in mrt_lockers:
            locker.idx += len(basic_problem.customers)+1
        new_lockers = deepcopy(basic_problem.lockers)
        for locker in new_lockers:
            locker.idx += len(mrt_lockers)
        new_lockers = mrt_lockers + new_lockers

        new_customers = deepcopy(basic_problem.customers)
        for customer in new_customers:
            for li, locker_idx in enumerate(customer.preferred_locker_idxs):
                customer.preferred_locker_idxs[li] = locker_idx + len(mrt_lockers)
        new_customers = add_mrt_lockers_to_preference(new_customers, mrt_lockers)
        new_problem = Cvrpptpl(basic_problem.depot,
                                new_customers,
                                new_lockers, 
                                mrt_lines,
                                basic_problem.vehicles,
                                instance_name=instance_name)
        if num_mrt_lines == 0:
            new_problem.mrt_lines = []
            mrt_lockers_idx = []
            new_problem.non_mrt_lockers = new_lockers
            new_problem.mrt_line_stations_idx= []
            new_problem.mrt_line_costs = []
            new_problem.mrt_line_capacities = []
            
        # cvrpptpl_problem.visualize_graph()
        new_problem.save_to_ampl_file(is_v2=True)
        new_problem.save_to_ampl_file(is_v2=False)
        new_problem.save_to_file()
    