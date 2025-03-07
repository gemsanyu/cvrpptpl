from random import shuffle

import numpy as np

from heuristic.operator import Operator
from heuristic.solution import Solution
class ReinsertionOperator(Operator):
    def get_destinations_to_reinsert(self, solution: Solution):
        is_dest_need_visit = np.logical_and(solution.destination_total_demands > 0, solution.destination_vehicle_assignmests == -1)
        dests_to_reinsert = np.where(is_dest_need_visit)[0]
        return dests_to_reinsert
class RandomOrderBestPosition(ReinsertionOperator):
    def apply(self, problem, solution):
        dests_to_reinsert = self.get_destinations_to_reinsert(solution)
        shuffle(dests_to_reinsert)
        for dest_idx in dests_to_reinsert:
            for v_idx, route in enumerate(solution.routes):
                route_ = route[1:] + route[:1]
                dist_from = problem.distance_matrix[route, dest_idx]
                dist_to = problem.distance_matrix[dest_idx, route_]
                dist_in_route = problem.distance_matrix[route, route_]
                d_cost = dist_from + dist_to - dist_in_route
                print(d_cost)
                print("----------")
        exit()