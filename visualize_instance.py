import matplotlib.pyplot as plt
import numpy as np
from cvrpptpl.arguments import visualize_instance_args
from cvrpptpl.cvrpptpl import Cvrpptpl, read_from_file


def visualize_instance(problem: Cvrpptpl):
    depot_coord = problem.depot_coord
    customer_coords = np.stack([customer.coord for customer in problem.customers], axis=0)
    locker_coords = np.stack([locker.coord for locker in problem.lockers], axis=0)
    hd_coords = np.stack([customer.coord for customer in problem.customers if not (customer.is_flexible or customer.is_self_pickup)], axis=0)
    sp_coords = np.stack([customer.coord for customer in problem.customers if customer.is_self_pickup], axis=0)
    fx_coords = np.stack([customer.coord for customer in problem.customers if customer.is_flexible], axis=0)
    
    plt.scatter(depot_coord[0], depot_coord[1], marker="s", s=80, label="Depot")
    plt.scatter(hd_coords[:,0], hd_coords[:,1], label="Home delivery custs")
    plt.scatter(sp_coords[:,0], sp_coords[:,1], label="Self-pickup custs")
    plt.scatter(fx_coords[:,0], fx_coords[:,1], label="Flexible custs")
    plt.scatter(locker_coords[:,0], locker_coords[:,1], label="Lockers", marker="h", s=70)

    mrt_lines = problem.mrt_lines
    mrt_start_coords = np.stack([mrt_line.start_station.coord for mrt_line in mrt_lines], axis=0)
    mrt_end_coords = np.stack([mrt_line.end_station.coord for mrt_line in mrt_lines], axis=0)
    mrt_pair_coords = np.concatenate([mrt_start_coords[:,np.newaxis,:], mrt_end_coords[:,np.newaxis,:]], axis=1)
    plt.scatter(mrt_start_coords[:,0], mrt_start_coords[:,1], s=100, marker="^", label="Start MRT")
    plt.scatter(mrt_end_coords[:,0], mrt_end_coords[:,1], s=100, marker="v", label="End MRT")
    
    for i, mrt_line in enumerate(mrt_lines):
        if i==0:
            plt.plot(mrt_pair_coords[i, :, 0], mrt_pair_coords[i, :, 1], "k--", label="MRT Line")
        else:
            plt.plot(mrt_pair_coords[i, :, 0], mrt_pair_coords[i, :, 1], "k--")
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = visualize_instance_args()
    problem = read_from_file(args.instance_name)
    visualize_instance(problem)