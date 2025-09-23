from arguments import prepare_args
from problem.cvrpptpl import read_from_file

if __name__ == "__main__":
    args = prepare_args()
    problem = read_from_file(args.instance_filename+".txt")
    problem.filename = args.instance_filename
    set_without_mrt = "m0" in args.instance_filename
    problem.save_to_ampl_file(set_without_mrt)

