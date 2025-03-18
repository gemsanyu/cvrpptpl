from enum import Enum
from math import exp
from random import random
from typing import List, Tuple

import numpy as np

from heuristic.operator import Operator, select_operator, OperationStatus
from heuristic.random_initialization import random_initialization
from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl


class LevelOperationStatus:
    SUCCESS = 1
    BETTER = 2
    BEST = 3
    FAILED = 4

L_DT_SCORE_MAP = {
    LevelOperationStatus.SUCCESS : 0.05,
    LevelOperationStatus.BETTER : 0.1,
    LevelOperationStatus.BEST : 0.2,
    LevelOperationStatus.FAILED : -0.05     
}

class ALNS():
    def __init__(self, 
                 problem: Cvrpptpl,
                 l1_destroy_operators: List[Operator],
                 l2_destroy_operators: List[Operator],
                 reassignment_operators: List[Operator],
                 reinsertion_operators: List[Operator],
                 initial_temp: float,
                 cooling_rate: float = 0.9,
                 max_iteration: int  = 1000,
                 ):
        self.problem: Cvrpptpl = problem
        self.l1_destroy_operators: List[Operator] = l1_destroy_operators
        self.l2_destroy_operators: List[Operator] = l2_destroy_operators
        self.reassignment_operators: List[Operator] = reassignment_operators
        self.reinsertion_operators: List[Operator] = reinsertion_operators
        
        self.l1_d_scores = np.asanyarray([0.1]*len(self.l1_destroy_operators), dtype=float)
        self.l2_d_scores = np.asanyarray([0.1]*len(self.l2_destroy_operators), dtype=float)
        self.ra_scores = np.asanyarray([0.1]*len(self.reassignment_operators), dtype=float)
        self.ri_scores = np.asanyarray([0.1]*len(self.reinsertion_operators), dtype=float)
        
        
        self.temp: float = initial_temp
        self.cooling_rate: float = cooling_rate
        self.max_iteration: int = max_iteration
        
        self.level_scores: np.ndarray = np.full([2, ], 0.1, dtype=float)
        self.level_operations = [self.level_1_operation, self.level_2_operation]
        self.best_solution: Solution = None
        self.curr_solution: Solution = None
    
    def update_score(self, 
                 operator: Operator, 
                 status: int, 
                 modified_solution_cost:float=None, 
                 current_solution_cost:float=None,
                 best_solution_cost:float=None):
        
        if status is not OperationStatus.SUCCESS:
            operator.score -= 0.1
            operator.score = max(min(operator.score, 2), 0.1)
            return
        if modified_solution_cost < best_solution_cost:
            operator.score += 0.2
        elif modified_solution_cost < current_solution_cost:
            operator.score += 0.1
        else:
            operator.score += 0.05
        operator.score = max(min(operator.score, 2), 0.1)
    
    def update_level_score(self, level_idx, status: LevelOperationStatus):
        dt_score = L_DT_SCORE_MAP[status]
        self.level_scores[level_idx] += dt_score
        self.level_scores = np.clip(self.level_scores, 0.1, 2)
    
    def solve(self)->Solution:
        initial_solution = random_initialization(self.problem)
        initial_solution.check_feasibility()
        self.best_solution = initial_solution
        self.curr_solution = initial_solution
        for it in range(self.max_iteration):
            print(f"iteration {it}")
            level_selection_probs = self.level_scores/self.level_scores.sum()
            level_idx = np.random.choice(2, size=1, p=level_selection_probs)[0]
            print(f"---Level {level_idx+1} operation")
            level_op = self.level_operations[level_idx]
            status = level_op()
            print(f"------Best Fitness:", self.best_solution.total_cost)
            self.update_level_score(level_idx, status)
            self.temp = self.temp*self.cooling_rate
        return self.best_solution
    
    def level_1_operation(self)->LevelOperationStatus:
        new_solution = self.curr_solution.copy()
        l1_d_op = select_operator(self.l1_destroy_operators, self.l1_d_scores)
        ri_op = select_operator(self.reinsertion_operators, self.ri_scores)
        l1_d_op.count += 1
        ri_op.count += 1
        
        d_status = l1_d_op.apply_with_check(self.problem, new_solution)
        ri_status = ri_op.apply_with_check(self.problem, new_solution)
        self.update_score(l1_d_op, d_status, new_solution.total_cost, self.curr_solution.total_cost, self.best_solution.total_cost)
        self.update_score(ri_op, ri_status, new_solution.total_cost, self.curr_solution.total_cost, self.best_solution.total_cost)
        if ri_status is not OperationStatus.SUCCESS:
            return LevelOperationStatus.FAILED
        status = self.accept_solution(new_solution)
        return status
    
    def level_2_operation(self)->LevelOperationStatus:
        new_solution = self.curr_solution.copy()
        l2_d_op = select_operator(self.l2_destroy_operators, self.l1_d_scores)
        ra_op = select_operator(self.reassignment_operators)
        ri_op = select_operator(self.reinsertion_operators, self.ri_scores)
        l2_d_op.count += 1
        ri_op.count += 1
        
        l2_d_status = l2_d_op.apply_with_check(self.problem, new_solution)
        ra_status = ra_op.apply_with_check(self.problem, new_solution)
        if ra_status is not OperationStatus.SUCCESS:
            self.update_score(ra_op, ra_status)    
            return LevelOperationStatus.FAILED
        ri_status = ri_op.apply_with_check(self.problem, new_solution)
        if ri_status is not OperationStatus.SUCCESS:
            self.update_score(ra_op, ri_status)
            return LevelOperationStatus.FAILED
        self.update_score(l2_d_op, l2_d_status, new_solution.total_cost, self.curr_solution.total_cost, self.best_solution.total_cost)
        self.update_score(ra_op, ra_status, new_solution.total_cost, self.curr_solution.total_cost, self.best_solution.total_cost)
        self.update_score(ri_op, ri_status, new_solution.total_cost, self.curr_solution.total_cost, self.best_solution.total_cost)
        status = self.accept_solution(new_solution)
        return status
        
    def accept_solution(self, new_solution: Solution)->LevelOperationStatus:
        status = LevelOperationStatus.SUCCESS
        if new_solution.total_cost < self.curr_solution.total_cost:
            self.curr_solution = new_solution
            status = LevelOperationStatus.BETTER
            if new_solution.total_cost < self.best_solution.total_cost:
                self.best_solution = new_solution
                status = LevelOperationStatus.BEST
        else:
            prob = exp(-(new_solution.total_cost-self.curr_solution.total_cost)/self.temp)
            if random() <= prob:
                self.curr_solution = new_solution
        return status
