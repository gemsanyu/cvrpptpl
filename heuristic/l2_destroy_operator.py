from random import randint

import numpy as np

from problem.cvrpptpl import Cvrpptpl
from heuristic.d_op_utils import complete_customers_removal, compute_customer_removal_d_costs, complete_locker_removal, compute_locker_removal_d_costs
from heuristic.l1_destroy_operator import L1DestroyOperator
from heuristic.operator import Operator, OperationStatus
from heuristic.solution import Solution, NO_DESTINATION, NO_VEHICLE

class WorstCustomersRemoval(L1DestroyOperator):
    def apply(self, problem, solution):
        cust_removal_d_costs = compute_customer_removal_d_costs(problem, solution)
        sorted_idx = np.argsort(-cust_removal_d_costs)
        custs_idx = np.arange(problem.num_customers)+1
        custs_idx = custs_idx[sorted_idx]
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(custs_idx))
        custs_idx = custs_idx[:num_to_remove]
        complete_customers_removal(problem, solution, custs_idx)
        return OperationStatus.SUCCESS
        
class RandomCustomersRemoval(L1DestroyOperator):    
    def apply(self, problem, solution):
        custs_idx = np.arange(problem.num_customers)+1
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(custs_idx))
        custs_idx = np.random.choice(custs_idx, num_to_remove, replace=False)
        print(custs_idx)
        complete_customers_removal(problem, solution, custs_idx)
        return OperationStatus.SUCCESS
        
# remove lockers
class WorstLockersRemoval(L1DestroyOperator):
    def apply(self, problem, solution):
        used_lockers = np.where(solution.locker_loads>0)[0]
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(used_lockers))        
        if len(used_lockers) > num_to_remove:
            locker_removal_d_costs = compute_locker_removal_d_costs(problem, solution, used_lockers)
            sorted_idx = np.argsort(locker_removal_d_costs)
            used_lockers = used_lockers[sorted_idx]
        
        for locker_idx in used_lockers:
            complete_locker_removal(problem, solution, locker_idx)
        return OperationStatus.SUCCESS

class RandomLockersRemoval(L1DestroyOperator):
    def apply(self, problem, solution):
        used_lockers = np.where(solution.locker_loads>0)[0]
        num_to_remove = randint(self.min_to_remove, self.max_to_remove)
        num_to_remove = min(num_to_remove, len(used_lockers))        
        used_lockers = np.random.choice(used_lockers, num_to_remove)
        for locker_idx in used_lockers:
            complete_locker_removal(problem, solution, locker_idx)
        return OperationStatus.SUCCESS