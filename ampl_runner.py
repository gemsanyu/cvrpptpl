import os
import subprocess


def call_ampl(instance_name, time_limit):
    template: str
    with open("CVRP_run_template", "r", encoding="utf-8") as f:
        template = f.read()
    run_script = template.replace("@INSTANCE@", instance_name)
    run_script = run_script.replace("@TIME_LIMIT@", str(time_limit))
    run_script_filename = f"run_{instance_name}.run"
    with open(run_script_filename, "w+", encoding="utf-8") as f:
        f.write(run_script)
    cmd_args = ["ampl", run_script_filename]
    subprocess.run(cmd_args)
    if os.path.exists(run_script_filename):
        os.remove(run_script_filename)
        print(f"File {run_script_filename} has been removed.")

if __name__ == "__main__":
    instance_names = ["A-n32-k5_idx_0_v1_ampl_.txt",
        # "A-n32-k5_idx_0_v2_ampl_.txt",
        # "A-n33-k5_idx_0_v1_ampl_.txt",
        # "A-n33-k5_idx_0_v2_ampl_.txt",
        # "A-n33-k6_idx_0_v1_ampl_.txt",
        # "A-n33-k6_idx_0_v2_ampl_.txt",
        # "A-n34-k5_idx_0_v1_ampl_.txt",
        # "A-n34-k5_idx_0_v2_ampl_.txt",
        # "A-n36-k5_idx_0_v1_ampl_.txt",
        # "A-n36-k5_idx_0_v2_ampl_.txt",
        # "A-n37-k5_idx_0_v1_ampl_.txt",
        # "A-n37-k5_idx_0_v2_ampl_.txt",
        # "A-n37-k6_idx_0_v1_ampl_.txt",
        # "A-n37-k6_idx_0_v2_ampl_.txt",
        # "A-n38-k5_idx_0_v1_ampl_.txt",
        # "A-n38-k5_idx_0_v2_ampl_.txt",
        # "A-n39-k5_idx_0_v1_ampl_.txt",
        # "A-n39-k5_idx_0_v2_ampl_.txt",
        # "A-n39-k6_idx_0_v1_ampl_.txt",
        # "A-n39-k6_idx_0_v2_ampl_.txt",
        # "A-n44-k6_idx_0_v1_ampl_.txt",
        # "A-n44-k6_idx_0_v2_ampl_.txt",
        # "A-n45-k6_idx_0_v1_ampl_.txt",
        # "A-n45-k6_idx_0_v2_ampl_.txt",
        # "A-n45-k7_idx_0_v1_ampl_.txt",
        # "A-n45-k7_idx_0_v2_ampl_.txt",
        # "A-n46-k7_idx_0_v1_ampl_.txt",
        # "A-n46-k7_idx_0_v2_ampl_.txt",
        # "A-n48-k7_idx_0_v1_ampl_.txt",
        # "A-n48-k7_idx_0_v2_ampl_.txt",
        # "A-n53-k7_idx_0_v1_ampl_.txt",
        # "A-n53-k7_idx_0_v2_ampl_.txt",
        # "A-n54-k7_idx_0_v1_ampl_.txt",
        # "A-n54-k7_idx_0_v2_ampl_.txt",
        # "A-n55-k9_idx_0_v1_ampl_.txt",
        # "A-n55-k9_idx_0_v2_ampl_.txt",
        # "A-n60-k9_idx_0_v1_ampl_.txt",
        # "A-n60-k9_idx_0_v2_ampl_.txt",
        # "A-n61-k9_idx_0_v1_ampl_.txt",
        # "A-n61-k9_idx_0_v2_ampl_.txt",
        # "A-n62-k8_idx_0_v1_ampl_.txt",
        # "A-n62-k8_idx_0_v2_ampl_.txt",
        # "A-n63-k10_idx_0_v1_ampl_.txt",
        # "A-n63-k10_idx_0_v2_ampl_.txt",
        # "A-n63-k9_idx_0_v1_ampl_.txt",
        # "A-n63-k9_idx_0_v2_ampl_.txt",
        # "A-n64-k9_idx_0_v1_ampl_.txt",
        # "A-n64-k9_idx_0_v2_ampl_.txt",
        # "A-n65-k9_idx_0_v1_ampl_.txt",
        # "A-n65-k9_idx_0_v2_ampl_.txt",
        # "A-n69-k9_idx_0_v1_ampl_.txt",
        # "A-n69-k9_idx_0_v2_ampl_.txt",
        # "A-n80-k10_idx_0_v1_ampl_.txt",
        # "A-n80-k10_idx_0_v2_ampl_.txt",
        ]
    time_limit = 10
    for instance_name in instance_names:
        call_ampl(instance_name, time_limit)