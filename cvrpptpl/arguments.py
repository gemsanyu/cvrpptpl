import sys

import argparse

def prepare_instance_generation_args():
    parser = argparse.ArgumentParser(description='CVRP-PT-PL instance generation')
    # customers
    parser.add_argument('--num-customers',
                        type=int,
                        default=100,
                        help='number of customers')
    parser.add_argument('--num-clusters',
                        type=int,
                        default=4,
                        help='number of clusters for customer, if customer locations are clustered')
    parser.add_argument('--pickup-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of pickup customers/number of customers, used to determine number of self pickup customers')
    parser.add_argument('--customer-location-mode',
                        type=str,
                        default="r",
                        choices=["c","r","rc"],
                        help='customers\' location distribution mode')
    parser.add_argument('--cluster-dt',
                        type=float,
                        default=30,
                        help='coefficient to determine customer locations\' cluster density')
    parser.add_argument('--demand-generation-mode',
                        type=str,
                        default="u",
                        help='demand generation mode')
    
    # depot
    parser.add_argument('--depot-location-mode',
                        type=str,
                        default="r",
                        choices=["c","r"],
                        help='depot\'s location mode')
    
    
    # locker
    parser.add_argument('--num-lockers',
                        type=int,
                        default=10,
                        help='number of lockers')
    parser.add_argument('--locker-capacity-ratio',
                        type=float,
                        default=0.6,
                        help='ratio of total custs\' demand qty that is divided to be lockers\' capacities')
    parser.add_argument('--locker-location-mode',
                        type=str,
                        default="r",
                        choices=["c","r","rc"],
                        help='lockers\' location distribution mode')
    parser.add_argument('--locker-cost',
                        type=float,
                        default=1,
                        help='locker cost')
    
    
    # mrt
    parser.add_argument('--num-mrt',
                        type=int,
                        default=4,
                        help='number of mrt stations, must be even and smaller than number of lockers')
    parser.add_argument('--freight-capacity-mode',
                        type=str,
                        default="e",
                        choices=["a","e"],
                        help='freight capacity generation mode, a for ample, e for enough')
    parser.add_argument('--mrt-line-cost',
                        type=float,
                        default=10,
                        help='mrt line cost per unit goods')
    
    
    # vehicles
    parser.add_argument('--num-vehicles',
                        type=int,
                        default=10,
                        help='number of vehicles')
    parser.add_argument('--vehicle-cost-reference',
                        type=float,
                        default=5,
                        help='vehicle cost reference')
    
    
    args = parser.parse_args(sys.argv[1:])
    return args