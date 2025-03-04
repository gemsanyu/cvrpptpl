from typing import List

import numpy as np

class Solution:
    """_summary_
    """
    def __init__(self):
        self.package_destinations: np.ndarray
        self.mrt_usage_masks: np.ndarray
        self.task_vehicle_assignmests: np.ndarray
        self.destination_total_demands: np.ndarray
        self.destination_total_costs: np.ndarray
        self.routes: List[np.ndarray]
        