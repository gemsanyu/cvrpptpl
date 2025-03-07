from abc import ABC, abstractmethod
from random import randint

import numpy as np

from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

class Operator(ABC):
    
    def __init__(self):
        super().__init__()
        self.score = 0
        self.count = 0
    
    @abstractmethod
    def apply(self, problem: Cvrpptpl, solution: Solution):
        raise NotImplementedError