from problem.cvrpptpl import read_from_file

def main():
    cvrp_instance_name = "A-n32-k5"
    filename = f"{cvrp_instance_name}_idx_0.txt"
    read_from_file(filename)

if __name__ == "__main__":
    main()