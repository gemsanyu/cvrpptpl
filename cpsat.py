from typing import Dict, List, Optional, Tuple

import numpy as np
from ortools.sat.python import cp_model

from problem.customer import Customer
from problem.locker import Locker


def get_model(customers: List[Customer], lockers: List[Locker], vehicle_capacity: int)->Tuple[cp_model.CpModel, List[List[Optional[cp_model.BoolVarT]]]]:
    model = cp_model.CpModel()
    x = []
    for i in range(len(customers)+5):
        x += [[]]
        for j in range(len(lockers)+len(customers)+20):
            x[i] += [None]
    total_demand = 0
    for customer in customers:
        if not (customer.is_flexible or customer.is_self_pickup):
            continue
        total_demand += customer.demand
        for l_idx in customer.preferred_locker_idxs:
            x[customer.idx][l_idx] = model.new_bool_var(f"x{customer.idx}->{l_idx}")
        model.add_exactly_one([x[customer.idx][l_idx] for l_idx in customer.preferred_locker_idxs])
    
    locker_caps_violation = [None for _ in range(len(lockers) + len(customers)+20)]

    for locker in lockers:
        l_idx = locker.idx
        relevant_xs = [x[c.idx][l_idx] for c in customers if x[c.idx][l_idx] is not None]
        relevant_demands = [c.demand for c in customers if x[c.idx][l_idx] is not None]
        if not relevant_xs:
            locker_caps_violation[l_idx] = cp_model.LinearExpr.constant(0)
            continue

        load = cp_model.LinearExpr.WeightedSum(relevant_xs, relevant_demands)
        model.add(load<=vehicle_capacity)
        max_possible_violation = sum(relevant_demands)  # safe upper bound
        over = model.NewIntVar(0, max_possible_violation, f"over_{l_idx}")
        
        # over = max(0, load - capacity)
        model.Add(over >= load - locker.capacity)
        
        locker_caps_violation[l_idx] = over
    
    total_violation = cp_model.LinearExpr.Sum(
        [v for v in locker_caps_violation if v is not None]
    )
    model.minimize(total_violation)
    return model, x

def best_fit_spfx_assignment(customers: List[Customer],
                             lockers: List[Locker],
                             vehicle_capacity: int) -> List[int]:
    model, x = get_model(customers, lockers, vehicle_capacity)
    solver = cp_model.CpSolver()
    status = solver.solve(model)
    customer_locker_assignments = [-1 for _ in range(len(customers)+10)]

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("⚠️ No feasible assignment found.")
        exit()
    for customer in customers:
        for locker in lockers:
            if x[customer.idx][locker.idx] is None:
                continue
            if solver.Value(x[customer.idx][locker.idx]):
                customer_locker_assignments[customer.idx] = locker.idx
                break
    return customer_locker_assignments        