import sys

import argparse

def prepare_args():
    parser = argparse.ArgumentParser(description='CVRP-PT-PL instance generation')
    
    # args for generating instance based on CVRP problem instances
    parser.add_argument('--instance-filename',
                        type=str,
                        default="A-n32-k5_idx_0",
                        help="the cvrp-pt-pl instance name")
    args = parser.parse_args(sys.argv[1:])
    return args