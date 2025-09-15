import multiprocessing as mp
import os
import pathlib
import subprocess


def call_ampl(instance_name, time_limit):
    template: str
    template_filename: str
    model_filename: str
    if "m0" in instance_name:
        model_filename = "CVRP14042025.mod"
        template_filename = "CVRP_run_template"
    else:
        model_filename = "CVRPPT14042025.mod"
        template_filename = "CVRPPT_run_template"
    
    with open(template_filename, "r", encoding="utf-8") as f:
        template = f.read()
        
    run_script = template.replace("@INSTANCE@", instance_name)
    run_script = run_script.replace("@TIME_LIMIT@", str(time_limit))
    run_script = run_script.replace("@MODEL@", model_filename)
    run_script_filename = f"run_{instance_name}.run"
    with open(run_script_filename, "w+", encoding="utf-8") as f:
        f.write(run_script)
    cmd_args = ["ampl", run_script_filename]
    subprocess.run(cmd_args)
    if os.path.exists(run_script_filename):
        os.remove(run_script_filename)
        print(f"File {run_script_filename} has been removed.")

if __name__ == "__main__":
    output_dir = pathlib.Path()/"ampl_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    instances = [
        # "A-n11-k3-m0-b3ampl_.txt",
        # "A-n11-k3-m1-b3ampl_.txt",
        # "A-n11-k3-m2-b3ampl_.txt",
        # "A-n11-k3-m3-b3ampl_.txt",
        # "A-n12-k3-m0-b3ampl_.txt",
        # "A-n12-k3-m1-b3ampl_.txt",
        # "A-n12-k3-m2-b3ampl_.txt",
        # "A-n12-k3-m3-b3ampl_.txt",
        # "A-n13-k3-m0-b3ampl_.txt",
        # "A-n13-k3-m1-b3ampl_.txt",
        # "A-n13-k3-m2-b3ampl_.txt",
        # "A-n13-k3-m3-b3ampl_.txt",
        # "A-n14-k3-m0-b3ampl_.txt",
        # "A-n14-k3-m1-b3ampl_.txt",
        # "A-n14-k3-m2-b3ampl_.txt",
        # "A-n14-k3-m3-b3ampl_.txt",
        "A-n20-k4-m0-b4ampl_.txt",
        "A-n20-k4-m1-b4ampl_.txt",
        "A-n20-k4-m2-b4ampl_.txt",
        "A-n20-k4-m3-b4ampl_.txt",
        "A-n21-k4-m0-b4ampl_.txt",
        "A-n21-k4-m1-b4ampl_.txt",
        "A-n21-k4-m2-b4ampl_.txt",
        "A-n21-k4-m3-b4ampl_.txt",
        "A-n22-k4-m0-b4ampl_.txt",
        "A-n22-k4-m1-b4ampl_.txt",
        "A-n22-k4-m2-b4ampl_.txt",
        "A-n22-k4-m3-b4ampl_.txt",
        "A-n23-k4-m0-b4ampl_.txt",
        "A-n23-k4-m1-b4ampl_.txt",
        "A-n23-k4-m2-b4ampl_.txt",
        "A-n23-k4-m3-b4ampl_.txt",
        "A-n24-k4-m0-b4ampl_.txt",
        "A-n24-k4-m1-b4ampl_.txt",
        "A-n24-k4-m2-b4ampl_.txt",
        "A-n24-k4-m3-b4ampl_.txt",
        "A-n25-k4-m0-b4ampl_.txt",
        "A-n25-k4-m1-b4ampl_.txt",
        "A-n25-k4-m2-b4ampl_.txt",
        "A-n25-k4-m3-b4ampl_.txt",
        "A-n31-k5-m0-b5ampl_.txt",
        "A-n31-k5-m1-b5ampl_.txt",
        "A-n31-k5-m2-b5ampl_.txt",
        "A-n31-k5-m3-b5ampl_.txt",
        "A-n32-k5-m0-b5ampl_.txt",
        "A-n32-k5-m1-b5ampl_.txt",
        "A-n32-k5-m2-b5ampl_.txt",
        "A-n32-k5-m3-b5ampl_.txt",
        "A-n32-k6-m0-b6ampl_.txt",
        "A-n32-k6-m1-b6ampl_.txt",
        "A-n32-k6-m2-b6ampl_.txt",
        "A-n32-k6-m3-b6ampl_.txt",
        "A-n33-k5-m0-b5ampl_.txt",
        "A-n33-k5-m1-b5ampl_.txt",
        "A-n33-k5-m2-b5ampl_.txt",
        "A-n33-k5-m3-b5ampl_.txt",
        "A-n35-k5-m0-b5ampl_.txt",
        "A-n35-k5-m1-b5ampl_.txt",
        "A-n35-k5-m2-b5ampl_.txt",
        "A-n35-k5-m3-b5ampl_.txt",
        "A-n36-k5-m0-b5ampl_.txt",
        "A-n36-k5-m1-b5ampl_.txt",
        "A-n36-k5-m2-b5ampl_.txt",
        "A-n36-k5-m3-b5ampl_.txt",
        "A-n36-k6-m0-b6ampl_.txt",
        "A-n36-k6-m1-b6ampl_.txt",
        "A-n36-k6-m2-b6ampl_.txt",
        "A-n36-k6-m3-b6ampl_.txt",
        "A-n37-k5-m0-b5ampl_.txt",
        "A-n37-k5-m1-b5ampl_.txt",
        "A-n37-k5-m2-b5ampl_.txt",
        "A-n37-k5-m3-b5ampl_.txt",
        "A-n38-k5-m0-b5ampl_.txt",
        "A-n38-k5-m1-b5ampl_.txt",
        "A-n38-k5-m2-b5ampl_.txt",
        "A-n38-k5-m3-b5ampl_.txt",
        "A-n38-k6-m0-b6ampl_.txt",
        "A-n38-k6-m1-b6ampl_.txt",
        "A-n38-k6-m2-b6ampl_.txt",
        "A-n38-k6-m3-b6ampl_.txt",
        "A-n43-k6-m0-b6ampl_.txt",
        "A-n43-k6-m1-b6ampl_.txt",
        "A-n43-k6-m2-b6ampl_.txt",
        "A-n43-k6-m3-b6ampl_.txt",
        "A-n44-k6-m0-b6ampl_.txt",
        "A-n44-k6-m1-b6ampl_.txt",
        "A-n44-k6-m2-b6ampl_.txt",
        "A-n44-k6-m3-b6ampl_.txt",
        "A-n44-k7-m0-b7ampl_.txt",
        "A-n44-k7-m1-b7ampl_.txt",
        "A-n44-k7-m2-b7ampl_.txt",
        "A-n44-k7-m3-b7ampl_.txt",
        "A-n45-k7-m0-b7ampl_.txt",
        "A-n45-k7-m1-b7ampl_.txt",
        "A-n45-k7-m2-b7ampl_.txt",
        "A-n45-k7-m3-b7ampl_.txt",
        "A-n47-k7-m0-b7ampl_.txt",
        "A-n47-k7-m1-b7ampl_.txt",
        "A-n47-k7-m2-b7ampl_.txt",
        "A-n47-k7-m3-b7ampl_.txt",
        "A-n52-k7-m0-b7ampl_.txt",
        "A-n52-k7-m1-b7ampl_.txt",
        "A-n52-k7-m2-b7ampl_.txt",
        "A-n52-k7-m3-b7ampl_.txt",
        "A-n53-k7-m0-b7ampl_.txt",
        "A-n53-k7-m1-b7ampl_.txt",
        "A-n53-k7-m2-b7ampl_.txt",
        "A-n53-k7-m3-b7ampl_.txt",
        "A-n54-k9-m0-b9ampl_.txt",
        "A-n54-k9-m1-b9ampl_.txt",
        "A-n54-k9-m2-b9ampl_.txt",
        "A-n54-k9-m3-b9ampl_.txt",
        "A-n59-k9-m0-b9ampl_.txt",
        "A-n59-k9-m1-b9ampl_.txt",
        "A-n59-k9-m2-b9ampl_.txt",
        "A-n59-k9-m3-b9ampl_.txt",
        "A-n60-k9-m0-b9ampl_.txt",
        "A-n60-k9-m1-b9ampl_.txt",
        "A-n60-k9-m2-b9ampl_.txt",
        "A-n60-k9-m3-b9ampl_.txt",
        "A-n61-k8-m0-b8ampl_.txt",
        "A-n61-k8-m1-b8ampl_.txt",
        "A-n61-k8-m2-b8ampl_.txt",
        "A-n61-k8-m3-b8ampl_.txt",
        "A-n62-k10-m0-b10ampl_.txt",
        "A-n62-k10-m1-b10ampl_.txt",
        "A-n62-k10-m2-b10ampl_.txt",
        "A-n62-k10-m3-b10ampl_.txt",
        "A-n62-k9-m0-b9ampl_.txt",
        "A-n62-k9-m1-b9ampl_.txt",
        "A-n62-k9-m2-b9ampl_.txt",
        "A-n62-k9-m3-b9ampl_.txt",
        "A-n63-k9-m0-b9ampl_.txt",
        "A-n63-k9-m1-b9ampl_.txt",
        "A-n63-k9-m2-b9ampl_.txt",
        "A-n63-k9-m3-b9ampl_.txt",
        "A-n64-k9-m0-b9ampl_.txt",
        "A-n64-k9-m1-b9ampl_.txt",
        "A-n64-k9-m2-b9ampl_.txt",
        "A-n64-k9-m3-b9ampl_.txt",
        "A-n68-k9-m0-b9ampl_.txt",
        "A-n68-k9-m1-b9ampl_.txt",
        "A-n68-k9-m2-b9ampl_.txt",
        "A-n68-k9-m3-b9ampl_.txt",
        "A-n79-k10-m0-b10ampl_.txt",
        "A-n79-k10-m1-b10ampl_.txt",
        "A-n79-k10-m2-b10ampl_.txt",
        "A-n79-k10-m3-b10ampl_.txt",
        "taipei-n18-k5-m0-b2ampl.txt",
        "taipei-n18-k5-m3-b2ampl.txt",
        
    ]
    time_limit = 28800
    args_list = [(instance_name, time_limit) for instance_name in instances]
    with mp.Pool(4) as pool:
        pool.starmap(call_ampl, args_list)
    # for instance_name in instances:
    #     call_ampl(instance_name, 14400)