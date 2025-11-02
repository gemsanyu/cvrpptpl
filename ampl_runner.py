import multiprocessing as mp
import os
import pathlib
import re
import subprocess


def call_ampl(instance_name, time_limit):
    template: str
    template_filename: str
    model_filename: str
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
    instance_name = "A-n10-k3-m1-b1ampl_.txt"
    time_limit = 14400
    call_ampl(instance_name, time_limit)