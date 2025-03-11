import numpy as np

from problem.cvrpptpl import Cvrpptpl
from heuristic.l1_destroy_operator import L1DestroyOperator
from heuristic.solution import Solution

def compute_customer_removal_d_costs(problem: Cvrpptpl, solution: Solution):
    cust_d_costs: np.ndarray = np.zeros([problem.num_customers,], dtype=float)
    for i, customer in enumerate(problem.customers):
        cust_idx = customer.idx
        # check if it is home delivery, then also calculate its 
        # removal_d_cost from the route
        v_idx = solution.destination_vehicle_assignmests[cust_idx]
        if v_idx > -1:
            pos = solution.routes[v_idx].index(cust_idx)
            prev_dest_idx = solution.routes[v_idx][pos-1]
            next_dest_idx = solution.routes[v_idx][(pos+1)%len(solution.routes[v_idx])]
            related_arc_costs = problem.distance_matrix[[prev_dest_idx, cust_idx, prev_dest_idx],[cust_idx, next_dest_idx, next_dest_idx]]*problem.vehicle_costs[v_idx]
            prev_to_dest_cost, pos_to_dest_cost, prev_to_next_cost = related_arc_costs
            d_cost = prev_to_next_cost -(prev_to_dest_cost + pos_to_dest_cost)
            cust_d_costs[i] += d_cost
            continue
        locker_idx = solution.package_destinations[cust_idx]
        # if it is self pickup and removing this customer
        # leads to removing a locker,
        # then check also the cost for this locker
        # and also if using mrt line, then check the mrt line cost
        
        
        
        # if it is self pickup:
        # compute its locker cost
        # if the locker use mrt line
        # also compute mrt line usage cost


class WorstDestinationsRemoval(L1DestroyOperator):
    
    def apply(self, problem, solution):
        cust_removal_d_costs = compute_customer_removal_d_costs(problem, solution)
        # sorted_idx = np.argsort(removal_d_costs)
        # dests_in_routes = dests_in_routes[sorted_idx]

        # num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        # num_to_remove = min(num_to_remove, len(dests_in_routes))
        # dests_in_routes = dests_in_routes[:num_to_remove]
        
        # # pick randomly
        # dests_to_remove = np.random.choice(dests_in_routes, num_to_remove, replace=False)
        # for dest_idx in dests_to_remove:
        #     remove_a_customer(solution, dest_idx)
        # finally remove destinations from route with 0 demands?
        # how? empty lockers
        # empty mrt lines

# remove lockers
# remove mrt lines


