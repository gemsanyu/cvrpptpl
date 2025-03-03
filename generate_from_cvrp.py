from random import shuffle

import matplotlib.pyplot as plt
import networkx as nx

from problem.arguments import prepare_instance_generation_args
from problem.cvrp import read_from_file
from problem.cvrpptpl import Cvrpptpl
from problem.mrt_line import generate_combined_mrt_lines
from problem.locker import generate_lockers
from problem.cust_locker_assignment import generate_customer_locker_preferences


def generate(args):
    filename = "A-n32-k5.vrp"
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
    mrt_lines_2_dicts = {"coordinate_mode": "vertical_line-small","num_mrt_lines": 2}
    args_dicts = [mrt_lines_1_dicts, mrt_lines_2_dicts]
    mrt_lines = generate_combined_mrt_lines(args_dicts, total_demands, min_coord, max_coord)
    lockers = generate_lockers(args.num_lockers, cvrp_problem.coords, total_demands, 0.3, 1, "c")
    
    # add mrt_lines lockers to generated lockers
    mrt_line_lockers = []
    for mrt_line in mrt_lines:
        mrt_line_lockers += [mrt_line.station_a, mrt_line.station_b]
    lockers = mrt_line_lockers + lockers
    for i, locker in enumerate(lockers):
        locker.idx = i + len(customers) + 1
    
    # generating preference matching for customers and lockers
    customers = generate_customer_locker_preferences(customers, lockers)
    cvrpptpl_problem = Cvrpptpl(cvrp_problem.depot,
                                customers, 
                                lockers,
                                mrt_lines,
                                cvrp_problem.vehicles)    
    cvrpptpl_problem.visualize()

if __name__ == "__main__":
    args = prepare_instance_generation_args()
    generate(args)
    