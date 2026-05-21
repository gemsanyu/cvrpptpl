import pathlib

from functools import partial

from new_model.callback import callback
from new_model.master import Master
from problem.cvrpptpl import read_from_file



instance_names = ["A-n6-k2-m0-b1.txt",
"A-n6-k2-m1-b1.txt",
"A-n6-k2-m2-b1.txt",
"A-n6-k2-m3-b1.txt",
"A-n7-k2-m0-b1.txt",
"A-n7-k2-m1-b1.txt",
"A-n7-k2-m2-b1.txt",
"A-n7-k2-m3-b1.txt",
"A-n8-k3-m0-b1.txt",
"A-n8-k3-m1-b1.txt",
"A-n8-k3-m2-b1.txt",
"A-n8-k3-m3-b1.txt",
"A-n9-k3-m0-b1.txt",
"A-n9-k3-m1-b1.txt",
"A-n9-k3-m2-b1.txt",
"A-n9-k3-m3-b1.txt",
"A-n10-k3-m0-b1.txt",
"A-n10-k3-m1-b1.txt",
"A-n10-k3-m2-b1.txt",
"A-n10-k3-m3-b1.txt",
"A-n11-k4-m0-b1.txt",
"A-n11-k4-m1-b1.txt",
"A-n11-k4-m2-b1.txt",
"A-n11-k4-m3-b1.txt",
"A-n12-k4-m0-b1.txt",
"A-n12-k4-m1-b1.txt",
"A-n12-k4-m2-b1.txt",
"A-n12-k4-m3-b1.txt",
"A-n13-k4-m0-b1.txt",
"A-n13-k4-m1-b1.txt",
"A-n13-k4-m2-b1.txt",
"A-n13-k4-m3-b1.txt",
"A-n14-k4-m0-b1.txt",
"A-n14-k4-m1-b1.txt",
"A-n14-k4-m2-b1.txt",
"A-n14-k4-m3-b1.txt",
"A-n15-k4-m0-b1.txt",
"A-n15-k4-m1-b1.txt",
"A-n15-k4-m2-b1.txt",
"A-n15-k4-m3-b1.txt",
"A-n16-k4-m0-b1.txt",
"A-n16-k4-m1-b1.txt",
"A-n16-k4-m2-b1.txt",
"A-n16-k4-m3-b1.txt",
"A-n17-k4-m0-b1.txt",
"A-n17-k4-m1-b1.txt",
"A-n17-k4-m2-b1.txt",
"A-n17-k4-m3-b1.txt",
"A-n18-k4-m0-b1.txt",
"A-n18-k4-m1-b1.txt",
"A-n18-k4-m2-b1.txt",
"A-n18-k4-m3-b1.txt",
"A-n19-k4-m0-b1.txt",
"A-n19-k4-m1-b1.txt",
"A-n19-k4-m2-b1.txt",
"A-n19-k4-m3-b1.txt",
"A-n20-k4-m0-b1.txt",
"A-n20-k4-m1-b1.txt",
"A-n20-k4-m2-b1.txt",
"A-n20-k4-m3-b1.txt",
"A-n21-k4-m0-b1.txt",
"A-n21-k4-m1-b1.txt",
"A-n21-k4-m2-b1.txt",
"A-n21-k4-m3-b1.txt",
"A-n22-k4-m0-b1.txt",
"A-n22-k4-m1-b1.txt",
"A-n22-k4-m2-b1.txt",
"A-n22-k4-m3-b1.txt",
"A-n23-k4-m0-b1.txt",
"A-n23-k4-m1-b1.txt",
"A-n23-k4-m2-b1.txt",
"A-n23-k4-m3-b1.txt",
"A-n24-k4-m0-b1.txt",
"A-n24-k4-m1-b1.txt",
"A-n24-k4-m2-b1.txt",
"A-n24-k4-m3-b1.txt",
"A-n25-k4-m0-b1.txt",
"A-n25-k4-m1-b1.txt",
"A-n25-k4-m2-b1.txt",
"A-n25-k4-m3-b1.txt",
"A-n26-k4-m0-b1.txt",
"A-n26-k4-m1-b1.txt",
"A-n26-k4-m2-b1.txt",
"A-n26-k4-m3-b1.txt",
"A-n32-k5-m0-b1.txt",
"A-n32-k5-m1-b1.txt",
"A-n32-k5-m2-b1.txt",
"A-n32-k5-m3-b1.txt",
"A-n33-k5-m0-b1.txt",
"A-n33-k5-m1-b1.txt",
"A-n33-k5-m2-b1.txt",
"A-n33-k5-m3-b1.txt",
"A-n33-k6-m0-b1.txt",
"A-n33-k6-m1-b1.txt",
"A-n33-k6-m2-b1.txt",
"A-n33-k6-m3-b1.txt",
"A-n34-k5-m0-b1.txt",
"A-n34-k5-m1-b1.txt",
"A-n34-k5-m2-b1.txt",
"A-n34-k5-m3-b1.txt",
"A-n36-k5-m0-b1.txt",
"A-n36-k5-m1-b1.txt",
"A-n36-k5-m2-b1.txt",
"A-n36-k5-m3-b1.txt",
"A-n37-k5-m0-b1.txt",
"A-n37-k5-m1-b1.txt",
"A-n37-k5-m2-b1.txt",
"A-n37-k5-m3-b1.txt",
"A-n37-k6-m0-b1.txt",
"A-n37-k6-m1-b1.txt",
"A-n37-k6-m2-b1.txt",
"A-n37-k6-m3-b1.txt",
"A-n38-k5-m0-b1.txt",
"A-n38-k5-m1-b1.txt",
"A-n38-k5-m2-b1.txt",
"A-n38-k5-m3-b1.txt",
"A-n39-k5-m0-b1.txt",
"A-n39-k5-m1-b1.txt",
"A-n39-k5-m2-b1.txt",
"A-n39-k5-m3-b1.txt",
"A-n39-k6-m0-b1.txt",
"A-n39-k6-m1-b1.txt",
"A-n39-k6-m2-b1.txt",
"A-n39-k6-m3-b1.txt",
"A-n44-k6-m0-b1.txt",
"A-n44-k6-m1-b1.txt",
"A-n44-k6-m2-b1.txt",
"A-n44-k6-m3-b1.txt",
"A-n45-k6-m0-b1.txt",
"A-n45-k6-m1-b1.txt",
"A-n45-k6-m2-b1.txt",
"A-n45-k6-m3-b1.txt",
"A-n45-k7-m0-b2.txt",
"A-n45-k7-m1-b2.txt",
"A-n45-k7-m2-b2.txt",
"A-n45-k7-m3-b2.txt",
"A-n46-k7-m0-b2.txt",
"A-n46-k7-m1-b2.txt",
"A-n46-k7-m2-b2.txt",
"A-n46-k7-m3-b2.txt",
"A-n48-k7-m0-b2.txt",
"A-n48-k7-m1-b2.txt",
"A-n48-k7-m2-b2.txt",
"A-n48-k7-m3-b2.txt",
"A-n53-k7-m0-b2.txt",
"A-n53-k7-m1-b2.txt",
"A-n53-k7-m2-b2.txt",
"A-n53-k7-m3-b2.txt",
"A-n54-k7-m0-b2.txt",
"A-n54-k7-m1-b2.txt",
"A-n54-k7-m2-b2.txt",
"A-n54-k7-m3-b2.txt",
"A-n55-k9-m0-b4.txt",
"A-n55-k9-m1-b4.txt",
"A-n55-k9-m2-b4.txt",
"A-n55-k9-m3-b4.txt",
"A-n60-k9-m0-b4.txt",
"A-n60-k9-m1-b4.txt",
"A-n60-k9-m2-b4.txt",
"A-n60-k9-m3-b4.txt",
"A-n61-k9-m0-b4.txt",
"A-n61-k9-m1-b4.txt",
"A-n61-k9-m2-b4.txt",
"A-n61-k9-m3-b4.txt",
"A-n62-k8-m0-b3.txt",
"A-n62-k8-m1-b3.txt",
"A-n62-k8-m2-b3.txt",
"A-n62-k8-m3-b3.txt",
"A-n63-k10-m0-b6.txt",
"A-n63-k10-m1-b6.txt",
"A-n63-k10-m2-b6.txt",
"A-n63-k10-m3-b6.txt",
"A-n63-k9-m0-b4.txt",
"A-n63-k9-m1-b4.txt",
"A-n63-k9-m2-b4.txt",
"A-n63-k9-m3-b4.txt",
"A-n64-k9-m0-b4.txt",
"A-n64-k9-m1-b4.txt",
"A-n64-k9-m2-b4.txt",
"A-n64-k9-m3-b4.txt",
"A-n65-k9-m0-b4.txt",
"A-n65-k9-m1-b4.txt",
"A-n65-k9-m2-b4.txt",
"A-n65-k9-m3-b4.txt",
"A-n69-k9-m0-b4.txt",
"A-n69-k9-m1-b4.txt",
"A-n69-k9-m2-b4.txt",
"A-n69-k9-m3-b4.txt",
"A-n80-k10-m0-b5.txt",
"A-n80-k10-m1-b5.txt",
"A-n80-k10-m2-b5.txt",
"A-n80-k10-m3-b5.txt"]

if __name__ == "__main__":
    for instance_filename in instance_names:
        problem = read_from_file(instance_filename)
        mp = Master(problem)
        mp.model.params.OutputFlag = 1
        mp_callback = partial(callback, mp=mp)
        mp.model.optimize(mp_callback)

        routes = []
        for i in mp.visitable_nodes:
            if i == mp.depot: continue
            if mp.x[mp.depot, i].X<0.5:
                continue
            route = [mp.depot, i]
            curr = i
            while curr!=mp.depot:
                for j in mp.visitable_nodes:
                    if curr==j or mp.x[curr, j].X<0.5: continue
                    curr = j
                    break
                route.append(curr)
            routes.append(route)

        output_string = ""
        output_string += (f"TOTAL COST: {mp.model.ObjVal}\n")
        output_string += (f"REGULAR VEHICLE COST: {mp.total_regular_vehicle_cost.getValue()}\n")
        output_string += (f"MRT OUTSOURCE COST: {mp.total_mrt_costs.getValue()}\n")
        output_string += (f"LOWER BOUND: {mp.model.ObjBound}\n")
        output_string += (f"GAP: {(mp.model.ObjVal-mp.model.ObjBound)/mp.model.ObjBound}\n")
        output_string += (f"RUNTIME: {mp.model.Runtime}\n")
        output_string += ("REGULAR VEHICLE ROUTES:\n")
        for ri, route in enumerate(routes):
            load = sum(mp.theta[i].X for i in route if i!=mp.depot)
            output_string+=(f"{route} ({load}<={mp.vehicle_capacity})\n")
        output_string+=("LOCKER ASSIGNMENT\n")
        for l in mp.lockers:
            assigned_customers = [i for i in mp.customers if (i,l) in mp.u.keys() and mp.u[i,l].X>0.5]
            if len(assigned_customers)>0:
                output_string+=(f"{l}: {assigned_customers}\n")
        output_string+=("MRT USAGE\n")
        for j,pj in mp.mrt_pair.items():
            if mp.z[j,pj].X>0:
                output_string+=(f"{j}->{pj} ({mp.z[j,pj].X}<={mp.mrt_line_capacities[j,pj]})\n")
        
        results_dir = pathlib.Path("results_new_model")
        results_dir.mkdir(parents=True, exist_ok=True)
        result_filepath = results_dir/instance_filename
        with open(result_filepath.absolute(), "w") as f:
            f.write(output_string)