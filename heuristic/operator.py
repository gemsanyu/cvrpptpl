from abc import ABC, abstractmethod
from enum import Enum
from random import randint
from typing import List

import numpy as np

from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

class OperationStatus(Enum):
    SUCCESS = 1
    FAILED = 2

class Operator(ABC):
    
    def __init__(self):
        super().__init__()
        self.score: float = 0.1
        self.count: int = 0
    
    @abstractmethod
    def apply(self, problem: Cvrpptpl, solution: Solution)->OperationStatus:
        raise NotImplementedError
    
    def apply_with_check(self, problem: Cvrpptpl, solution: Solution)->OperationStatus:
        status = self.apply(problem, solution)
        solution.check_validity()
        return status
    

def select_operator(operators: List[Operator], scores: np.asanyarray=None)->Operator:
    if scores is None:
        scores =  np.asanyarray([op.score for op in operators])
    probs = scores/scores.sum()
    selected_operator = np.random.choice(operators, p=probs, size=1)[0]
    return selected_operator