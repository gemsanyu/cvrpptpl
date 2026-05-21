import os

import gurobipy as grp

def get_grb_model(name:str)->grp.Model:
    env = grp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("WLSAccessID", os.environ.get("WLSAccessID"))
    env.setParam("WLSSECRET",  os.environ.get("WLSSECRET"))
    env.setParam("LICENSEID", int(os.environ.get("GUROBI_LICENSEID")))
    env.start()
    model = grp.Model(name, env=env)
    return model