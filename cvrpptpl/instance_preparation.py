from typing import List, Tuple

from cvrpptpl.customer import Customer
from cvrpptpl.locker import Locker
from cvrpptpl.mrt_line import MrtLine


def reindex_customer_by_delivery(customers: List[Customer]) -> List[Customer]:
    new_customers: List[Customer] = []
    num_customers = len(customers)
    for cust in customers:
        if not cust.is_self_pickup:
            new_customers += [cust]
    
    for cust in customers:
        if cust.is_self_pickup:
            new_customers += [cust]
    for i in range(num_customers):
        new_customers[i].idx = i+1
    return new_customers


def reindex_mrt_line_lockers(customers: List[Customer],
                             mrt_lines: List[MrtLine],
                             lockers: List[Locker]) -> Tuple[List[Customer],List[MrtLine],List[Locker]]:
    # mrt stations have lower index than non mrt station lockers
    old_idx_to_locker_dict = {locker.idx: locker for locker in lockers}
    new_lockers: List[Locker] = sum([ [mrt_line.start_station, mrt_line.end_station] for mrt_line in mrt_lines], [])
    old_mrt_idxs = [mrt_station.idx for mrt_station in new_lockers]
    
    for locker in lockers:
        if locker.idx not in old_mrt_idxs:
            new_lockers += [locker]
    
    num_customers = len(customers)
    num_lockers = len(lockers)
    for i in range(num_lockers):
        new_lockers[i].idx = i+num_customers+1
    
    for i in range(num_customers):
        if not customers[i].is_self_pickup:
            continue
        for j, old_locker_idx in enumerate(customers[i].preferred_locker_idxs):
            new_locker_idx = old_idx_to_locker_dict[old_locker_idx].idx
            customers[i].preferred_locker_idxs[j] = new_locker_idx
            
    return customers, mrt_lines, new_lockers