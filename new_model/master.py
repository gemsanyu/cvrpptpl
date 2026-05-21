from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from new_model.utils import get_grb_model
from problem.cvrpptpl import Cvrpptpl

class Master:
    # parameters
    travel_costs: dict[tuple[int, int], float]
    travel_time: dict[tuple[int, int], float]
    outsourcing_costs: dict[tuple[int, int], float]
    demands: np.ndarray
    customers: set[int]
    sp_customers: set[int]
    fx_customers: set[int]
    hd_customers: set[int]
    lockers: set[int]
    external_lockers: set[int]
    mrt_stations: set[int]
    mrt_pair: dict[int, int]
    destination_alternatives: list[list[int]]
    depot: int
    visitable_nodes: set[int]
    nodes: set[int]
    node_capacities: dict[int, float]
    locker_capacities: dict[int, float]
    mrt_line_capacities: dict[tuple[int, int], float]

    # decision variables
    a: dict[int, gp.Var]
    x: dict[tuple[int, int], gp.Var]
    z: dict[tuple[int, int], gp.Var]
    u: dict[tuple[int, int], gp.Var]
    y: dict[tuple[int, int], gp.Var]
    lamda: dict[int, gp.Var]
    theta: dict[int, gp.Var]
    def __init__(self, problem:Cvrpptpl) -> None:
        self._get_relevant_params(problem)
        self._setup_model()

    def _setup_model(self):
        self.model = get_grb_model("CVRP-PT-PL")
        self.model.Params.LazyConstraints = 1
        self._init_variables()
        self._add_constraints()
        self._set_objective()
        self._add_valid_inequalities()
        
    def _set_objective(self):
        self.total_regular_vehicle_cost = gp.quicksum(self.x[i,j]*self.travel_costs[i,j] for i,j in self.arcs)
        self.total_mrt_costs = gp.quicksum(self.outsourcing_costs[j,pj]*self.z[j,pj] for j,pj in self.mrt_pair.items())
        self.obj = self.total_regular_vehicle_cost + self.total_mrt_costs
        self.model.setObjective(self.obj, GRB.MINIMIZE)

    def _init_variables(self):
        self.a = {}
        for i in self.visitable_nodes:
            if i == self.depot:
                continue
            self.a[i] = self.model.addVar(0, 1, 0, GRB.BINARY, f"a[{i}]")

        self.u = {}
        for i in self.customers:
            for j in self.destination_alternatives[i]:
                self.u[i,j] = self.model.addVar(0,1,0,GRB.BINARY,f"u[{i},{j}]")
        
        self.lamda = {}
        self.theta = {}
        for i in self.nodes:
            if i == self.depot:
                continue
            self.lamda[i] = self.model.addVar(0, self.node_capacities[i], 0, GRB.CONTINUOUS, f"ld[{i}]")
            self.theta[i] = self.model.addVar(0, self.node_capacities[i], 0, GRB.CONTINUOUS, f"theta[{i}]")
        
        self.z = {}
        self.y = {}
        for (j, pj), capacity in self.mrt_line_capacities.items():
            self.y[j, pj] = self.model.addVar(0, 1, 0, GRB.BINARY, f"y[{j},{pj}]")
            self.z[j, pj] = self.model.addVar(0, capacity, 0, GRB.CONTINUOUS, f"z[{j},{pj}]")

        self.x = {}
        for (i,j) in self.arcs:
            self.x[i,j] = self.model.addVar(0, 1, 0, GRB.BINARY, f"x[{i},{j}]")

    def _add_valid_inequalities(self):
        # 1. BP
        # 2. SEC-2
        # self._add_bp_min_num_vec()
        self._add_sec2()
        self._add_sec3()
    
    # def _add_bp_min_num_vec(self):


    def _add_sec2(self):
        N3 = [i for i in self.visitable_nodes if i!=self.depot]
        combos = combinations(N3, 2)
        for (i,j) in combos:
            total_demand = self.theta[i] + self.theta[j]
            constr = self.x[i,j] + self.x[j,i] <= self.a[i] + self.a[j] - total_demand/self.vehicle_capacity
            self.model.addConstr(constr)
    
    def _add_sec3(self):
        N3 = [i for i in self.visitable_nodes if i!=self.depot]
        combos = combinations(N3, 3)
        for (i,j,k) in combos:
            total_demand = self.theta[i] + self.theta[j] + self.theta[k]
            all_arcs = self.x[i,j] + self.x[i,k] + self.x[j,i] + self.x[j,k] + self.x[k,i] + self.x[k,j]
            num_visits =self.a[i] + self.a[j] + self.a[k]
            constr = all_arcs <= num_visits - total_demand/self.vehicle_capacity
            self.model.addConstr(constr)

    def _add_constraints(self):
        # 1. PARCEL DESTINATION
        for i in self.customers:
            constr = gp.quicksum(self.u[i, j] for j in self.destination_alternatives[i]) == 1
            self.model.addConstr(constr)
        
        # 2. DESTINATION LOADS AGGREGATION
        for j in self.lamda.keys():
            total_load = gp.quicksum(self.demands[i]*self.u[i,j] for i in self.customers if j in self.destination_alternatives[i])
            constr = self.lamda[j] == total_load
            self.model.addConstr(constr)
        
        # 3. LOCKER CAPACITY
        for j in self.lockers:
            constr = self.lamda[j] <= self.locker_capacities[j]
            self.model.addConstr(constr)
        
        # 4. MRT LINE USAGE
        for j, pj in self.mrt_pair.items():
            is_end_station_used = gp.quicksum(self.u[i, pj] for i in self.customers if pj in self.destination_alternatives[i])
            constr = self.y[j, pj] <= is_end_station_used
            self.model.addConstr(constr)
        
        # 5. FLOW CONSTRAINTS
        # 5.1 active nodes
        for j in self.visitable_nodes:
            if j == self.depot:
                continue
            incoming_edges = gp.quicksum(self.x[i,j] for i in self.visitable_nodes if (i,j) in self.arcs)
            constr = self.a[j] == incoming_edges
            self.model.addConstr(constr)
        
        # 5.2 flow equality
        for j in self.visitable_nodes:
            if j == self.depot:
                continue
            incoming_edges = gp.quicksum(self.x[i,j] for i in self.visitable_nodes if (i,j) in self.arcs)
            outgoing_edges = gp.quicksum(self.x[j,k] for k in self.visitable_nodes if (j,k) in self.arcs)
            self.model.addConstr(incoming_edges == outgoing_edges)
        
        # 5.3 num vehicles
        num_vehicles_going_out = gp.quicksum(self.x[self.depot, j] for j in self.visitable_nodes if (self.depot, j) in self.arcs)
        constr = num_vehicles_going_out <= self.num_vehicles
        self.model.addConstr(constr)

        # 5.4 num vehicles consistency
        num_vehicles_going_in = gp.quicksum(self.x[j, self.depot] for j in self.visitable_nodes if (j,self.depot) in self.arcs)
        constr = num_vehicles_going_out == num_vehicles_going_in
        self.model.addConstr(constr)

        # 5.5 subtour elimination constraints is in the callback!

        # 6. DEMAND AGGREGATION CONSTRAINTS
        for i in self.customers:
            self.model.addConstr(self.theta[i] == self.lamda[i])
        for l in self.external_lockers:
            self.model.addConstr(self.theta[l] == self.lamda[l])
        
        # 7. MRT LINE CAPACITY
        for j, pj in self.mrt_pair.items():
            constr = self.z[j, pj] <= self.mrt_line_capacities[j, pj]*self.y[j, pj]
            self.model.addConstr(constr)
        
        # 8. MRT LINE LOAD
        for j, pj in self.mrt_pair.items():
            constr_a = self.z[j, pj] >= self.lamda[pj] - self.locker_capacities[pj]*(1-self.y[j, pj])
            self.model.addConstr(constr_a)

            constr_b = self.z[j, pj] <= self.lamda[pj]
            self.model.addConstr(constr_b)
        
        # 9. DEMAND AGGREGATION FOR MRT STATIONS
        for j in self.mrt_stations:
            pj = self.mrt_pair[j]
            constr = self.theta[pj] == self.lamda[pj] - self.z[j, pj] + self.z[pj, j]
            self.model.addConstr(constr)
        
        # 10. ACTIVE NODES DEMAND
        for i in self.visitable_nodes:
            if i == self.depot:
                continue
            constr_a = self.theta[i] >= self.a[i]
            self.model.addConstr(constr_a)

            constr_b = self.theta[i] <= self.vehicle_capacity*self.a[i]
            self.model.addConstr(constr_b)


    def _get_relevant_params(self, problem:Cvrpptpl):
        self.num_vehicles = problem.num_vehicles
        self.vehicle_capacity = problem.vehicle_capacities[0]
        self.demands = problem.demands
        self.customers = set([cust.idx for cust in problem.customers])
        self.fx_customers = set([cust.idx for cust in problem.customers if cust.is_flexible])
        self.sp_customers = set([cust.idx for cust in problem.customers if cust.is_self_pickup])
        self.hd_customers = set([cust.idx for cust in problem.customers if not cust.is_flexible and not cust.is_self_pickup])
        self.destination_alternatives = problem.destination_alternatives
        self.depot = problem.depot.idx
        visitable_nodes = [self.depot]
        visitable_nodes += [cust.idx for cust in problem.customers if not cust.is_self_pickup]
        visitable_nodes += [locker.idx for locker in problem.lockers]
        self.visitable_nodes = set(visitable_nodes)
        self.arcs = [(i,j) for i in self.visitable_nodes for j in self.visitable_nodes if i!=j]
        self.nodes = set([node.idx for node in problem.nodes])
        self.lockers = set([locker.idx for locker in problem.lockers])
        self.mrt_stations = set([mrt_line.start_station.idx for mrt_line in problem.mrt_lines] + [mrt_line.end_station.idx for mrt_line in problem.mrt_lines])
        self.mrt_pair = {mrt_line.start_station.idx:mrt_line.end_station.idx for mrt_line in problem.mrt_lines}
        self.locker_capacities = {l: problem.locker_capacities[l] for l in self.lockers}
        self.mrt_line_capacities = {(mrt_line.start_station.idx,mrt_line.end_station.idx):mrt_line.freight_capacity for mrt_line in problem.mrt_lines}
        self.outsourcing_costs = {(mrt_line.start_station.idx,mrt_line.end_station.idx):mrt_line.cost for mrt_line in problem.mrt_lines}
        self.external_lockers = set([l for l in self.lockers if l not in self.mrt_stations])
        
        self.node_capacities = {}
        self.node_capacities[self.depot] = 0
        for i in self.customers:
            if i in self.sp_customers:
                self.node_capacities[i] = 0 
            else:
                self.node_capacities[i] = self.demands[i]
        for l in self.lockers:
            self.node_capacities[l] = self.locker_capacities[l]

        self.travel_time = {(i,j):problem.distance_matrix[i,j] for i,j in self.arcs}
        self.travel_costs = {(i,j):self.travel_time[i,j]*problem.vehicle_costs[0] for i,j in self.arcs}


    