
from heuristic.random_initialization import random_initialization
from problem.cvrpptpl import read_from_file

def main():
    cvrp_instance_name = "A-n32-k5"
    cvrpptpl_filename = f"{cvrp_instance_name}_idx_0.txt"
    problem = read_from_file(cvrpptpl_filename)
    random_initialization(problem)
    
if __name__ == "__main__":
    main()