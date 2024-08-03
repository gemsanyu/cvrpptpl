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
    parser.add_argument('--flexible-ratio',
                        type=float,
                        default=0.3,
                        help='ratio of flexible customers/number of customers, used to determine number of flexible customers')
    
    parser.add_argument('--customer-location-mode',
                        type=str,
                        default="c",
                        choices=["c","r","rc"],
                        help='customers\' location distribution mode \
                             c: customer locations are clustered \
                             r: customer locations are randomly scattered \
                             rc: half clustered, half random')
    parser.add_argument('--cluster-dt',
                        type=float,
                        default=30,
                        help='coefficient to determine customer locations\' cluster density')
    parser.add_argument('--demand-generation-mode',
                        type=str,
                        default="5-20",
                        help="demand generation mode \
                            u: uniform, all is set to 1 \
                            q: quartile, the \
                                customers split into four quartiles based on coordinate then even quartiles\
                                get small demand odd quartile get big demand, \
                            sl: some get small demand, some get large \
                            L-U: (replace L and U with integer) means random from L to U")
    
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
                        default=10,
                        help='number of lockers')
    parser.add_argument('--locker-capacity-ratio',
                        type=float,
                        default=0.6,
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
                        help='freight capacity generation mode,\
                            a: ample capacity (10000) \
                            e: enough capacity (U[0.2,0.8]*demands in end station)')
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
                        help='vehicle cost reference\
                             vehicle cost will relate to its capacity times this value')
    
    
    args = parser.parse_args(sys.argv[1:])
    return args

def visualize_instance_args():
    parser = argparse.ArgumentParser(description='CVRP-PT-PL instance generation')
    parser.add_argument('--instance-name',
                        type=str,
                        help='instance filename')
    args = parser.parse_args(sys.argv[1:])
    return args