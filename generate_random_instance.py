import numpy as np
from problem.cust_locker_assignment import generate_customer_locker_preferences
from visualize_instance import visualize_instance

from cvrpptpl.arguments import prepare_instance_generation_args
from cvrpptpl.customer import generate_customers
from cvrpptpl.cvrpptpl import Cvrpptpl
from cvrpptpl.depot import generate_depot_coord
from cvrpptpl.instance_preparation import (reindex_customer_by_delivery,
                                           reindex_mrt_line_lockers)
from cvrpptpl.locker import generate_lockers
from cvrpptpl.mrt_line import generate_mrt_lines
from cvrpptpl.vehicle import generate_vehicles


def run(args):
    customers = generate_customers(args.num_customers,
                                   args.customer_location_mode,
                                   args.num_clusters,
                                   args.cluster_dt,
                                   args.demand_generation_mode)
    customer_coords = [customer.coord for customer in customers]
    customer_coords = np.stack(customer_coords, axis=0)

    depot_coord = generate_depot_coord(customer_coords, 
                                       args.depot_location_mode)
    total_customer_demand = sum([customer.demand for customer in customers])
    lockers = generate_lockers(args.num_lockers,
                               customer_coords,
                               total_customer_demand,
                               args.locker_capacity_ratio,
                               args.locker_cost,
                               args.locker_location_mode)
    customers = generate_customer_locker_preferences(customers, lockers, args.pickup_ratio, args.flexible_ratio)
    mrt_lines = generate_mrt_lines(args.num_mrt,
                                   lockers,
                                   customers,
                                   args.mrt_line_cost,
                                   args.freight_capacity_mode)
    vehicles = generate_vehicles(args.num_vehicles,
                                 args.num_customers,
                                 total_customer_demand,
                                 args.vehicle_cost_reference)
    
    customers = reindex_customer_by_delivery(customers)
    customers, mrt_lines, lockers = reindex_mrt_line_lockers(customers, mrt_lines, lockers)
    for c_idx, customer in enumerate(customers):
        if customer.is_self_pickup or customer.is_flexible:
            customers[c_idx].preferred_locker_idxs.sort()
    problem = Cvrpptpl(depot_coord,
                       customers,
                       lockers,
                       mrt_lines,
                       vehicles,
                       args.depot_location_mode,
                       args.locker_capacity_ratio,
                       args.locker_location_mode,
                       args.pickup_ratio,
                       args.flexible_ratio,
                       args.freight_capacity_mode,
                       args.customer_location_mode)
    # visualize_instance(problem)
    problem.save_to_file()
    problem.save_to_ampl_file()
    
if __name__ == "__main__":
    args = prepare_instance_generation_args()
    run(args)