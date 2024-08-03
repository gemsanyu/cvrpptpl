import pathlib

from cvrpptpl.arguments import visualize_instance_args
from cvrpptpl.cvrpptpl import Cvrpptpl, read_from_file

def parse_instance(problem: Cvrpptpl, parsed_instance_filepath: pathlib.Path):
    lines = []
    vec_idx_line = "set K:=\t" + "\t".join([str(i+1) for i in range(problem.num_vehicles)]) + ";"
    lines.append(vec_idx_line)
    ch_idx_line = "set C_H:=\t" + "\t".join([str(cust.idx) for cust in problem.customers if not cust.is_flexible and not cust.is_self_pickup]) + ";"
    lines.append(ch_idx_line)
    cs_idx_line = "set C_S:=\t" + "\t".join([str(cust.idx) for cust in problem.customers if cust.is_self_pickup]) + ";"
    lines.append(cs_idx_line)
    cf_idx_line = "set C_F:=\t" + "\t".join([str(cust.idx) for cust in problem.customers if cust.is_flexible]) + ";"
    lines.append(cf_idx_line)
    mrt_locker_idx_line = "set M:=\t" + "\t".join([str(mrt_line.start_station.idx)+"\t"+str(mrt_line.end_station.idx) for mrt_line in problem.mrt_lines]) + ";"
    lines.append(mrt_locker_idx_line)
    mrt_start_locker_idx_line = "set M_t:=\t" + "\t".join([str(mrt_line.start_station.idx) for mrt_line in problem.mrt_lines]) + ";"
    lines.append(mrt_start_locker_idx_line)
    mrt_locker_idxs = [mrt_line.start_station.idx for mrt_line in problem.mrt_lines] + [mrt_line.end_station.idx for mrt_line in problem.mrt_lines]
    non_mrt_locker_idxs = [locker.idx for locker in problem.lockers if locker.idx not in mrt_locker_idxs]
    non_mrt_locker_idx_line = "set L_B:=\t" + "\t".join([str(idx) for idx in non_mrt_locker_idxs]) + ";"
    lines.append(non_mrt_locker_idx_line)
    sp_idxs = [cust.idx for cust in problem.customers if cust.is_self_pickup]
    non_sp_idxs = [idx for idx in range(problem.num_nodes) if idx not in sp_idxs]
    lines.append("set A1:=")
    for i in non_sp_idxs:
        line = f"({i},*)\t" + "\t".join([str(j) for j in non_sp_idxs if j != i])
        lines.append(line)
    lines.append(";")
    
    lines.append("set A2:=")
    for mrt_line in problem.mrt_lines:
        line = f"({mrt_line.start_station.idx}, *)\t{mrt_line.end_station.idx}"
        lines.append(line)
    lines.append(";")

    lines.append("param BigM:=999;")
    lines.append("param r:=2;")
    
    lines.append("param d:=")
    
    
    for line in lines:
        print(line)
    exit()
    
    with open(parsed_instance_filepath.absolute, "w") as f:
        f.writelines(lines)

if __name__ == "__main__":
    args = visualize_instance_args()
    problem = read_from_file(args.instance_name)
    parsed_instance_filedir = pathlib.Path()/"parsed_instances"
    parsed_instance_filedir.mkdir(parents=True, exist_ok=True)
    parsed_instance_filepath = parsed_instance_filedir/args.instance_name
    parse_instance(problem, parsed_instance_filepath)