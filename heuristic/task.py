
class Task:
    def __init__(self,
                 destination_idx:int,
                 demand: int):
        self.destination_idx = destination_idx
        self.demand = demand
         
class SelfPickupTask(Task):
    def __init__(self, 
                 destination_idx, 
                 demand,
                 ):
        super().__init__(destination_idx, demand)
        