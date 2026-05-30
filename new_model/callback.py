import igraph as ig
import gurobipy as gp
from gurobipy import GRB

from new_model.master import Master

def add_fractional_cuts(model: gp.Model, mp)->bool:
    # 1. CRITICAL: Only separate cuts if the LP relaxation at this node is optimal
    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return False
        
    # 2. Retrieve FRACTIONAL values using dictionary comprehensions
    x_vals = {arc: model.cbGetNodeRel(var) for arc, var in mp.x.items()}
    a_vals = {node: model.cbGetNodeRel(var) for node, var in mp.a.items()}
    theta_vals = {node: model.cbGetNodeRel(var) for node, var in mp.theta.items()}
    
    depot_node_name = mp.depot

    # -------------------------------------------------------------------------
    # METHOD 1: Connected Components on Support Graph (For Isolated Subtours)
    # -------------------------------------------------------------------------
    # Filter arcs that have any meaningful fractional flow
    active_arcs = [arc for arc, val in x_vals.items() if val > 0.01]
    if not active_arcs:
        return False
        
    g_support = ig.Graph.TupleList(active_arcs, directed=True)
    cuts_added = False
    for component in g_support.connected_components(mode="strong"):
        node_names = [g_support.vs[v]["name"] for v in component]
        
        # If the component doesn't include the depot, check it for a violation
        if depot_node_name not in node_names:
            sub_arcs = [(i, j) for i in node_names for j in node_names if i != j and (i, j) in mp.x]
            
            # Evaluate LHS vs RHS fractionally
            lhs_val = sum(x_vals[i, j] for i, j in sub_arcs)
            num_visits_val = sum(a_vals[i] for i in node_names if i in a_vals)
            total_demand_val = sum(theta_vals[i] for i in node_names if i in theta_vals)
            rhs_val = num_visits_val - (total_demand_val / mp.vehicle_capacity)
            
            # If violated by a meaningful precision threshold, cut it off!
            if lhs_val > rhs_val + 1e-4:
                total_demand_expr = gp.quicksum(mp.theta[i] for i in node_names)
                num_visits_expr = gp.quicksum(mp.a[i] for i in node_names)
                
                model.cbCut(gp.quicksum(mp.x[i, j] for i, j in sub_arcs) <= num_visits_expr - total_demand_expr / mp.vehicle_capacity)
                cuts_added = True

    # -------------------------------------------------------------------------
    # METHOD 2: Exact Min-Cut Separation (For Depot / Over-Capacity Routes)
    # -------------------------------------------------------------------------
    # Create a weighted graph where edge capacities equal their fractional x values
    edges_with_weights = [(i, j, val) for (i, j), val in x_vals.items() if val > 1e-4]
    if not edges_with_weights:
        return cuts_added
        
    g_weights = ig.Graph.TupleList(edges_with_weights, directed=True, edge_attrs="weight")
    
    # Check the max-flow/min-cut bottleneck from the depot to every other vertex
    for v in g_weights.vs:
        customer_id = v["name"]
        if customer_id == depot_node_name:
            continue 
            
        # Find the minimum cut separating the depot from this customer
        cut = g_weights.st_mincut(
            source=g_weights.vs.find(name=depot_node_name).index,
            target=v.index,
            capacity="weight"
        )
        
        # cut.partition[1] gives the target (customer) side of the cut
        cut_nodes = [g_weights.vs[idx]["name"] for idx in cut.partition[1]]
        
        # Disregard if the depot accidentally leaked into this partition side
        if depot_node_name in cut_nodes:
            continue
            
        # Check if this cut set violates the capacitated SEC
        sub_arcs = [(i, j) for i in cut_nodes for j in cut_nodes if i != j and (i, j) in mp.x]
        lhs_val = sum(x_vals[i, j] for i, j in sub_arcs)
        num_visits_val = sum(a_vals[i] for i in cut_nodes if i in a_vals)
        total_demand_val = sum(theta_vals[i] for i in cut_nodes if i in theta_vals)
        rhs_val = num_visits_val - (total_demand_val / mp.vehicle_capacity)
        
        if lhs_val > rhs_val + 1e-4:
            total_demand_expr = gp.quicksum(mp.theta[i] for i in cut_nodes)
            num_visits_expr = gp.quicksum(mp.a[i] for i in cut_nodes)
            
            model.cbCut(gp.quicksum(mp.x[i, j] for i, j in sub_arcs) <= num_visits_expr - total_demand_expr / mp.vehicle_capacity)
            cuts_added = True
    return cuts_added
    
def add_cycle_cuts(model: gp.Model, mp:Master)->bool:
    x_vals = model.cbGetSolution(mp.x)
    theta = model.cbGetSolution(mp.theta)
    a = model.cbGetSolution(mp.a)
    active_arcs = [arc for arc, val in x_vals.items() if val > 0.5]
    if not active_arcs:
        return False

    # 1. Build the network of active movements
    g = ig.Graph.TupleList(active_arcs, directed=True)
    
    # Track if we've already found the component containing the depot
    depot_node_name = mp.depot

    # 2. Analyze components.
    cuts_added = False
    for component in g.connected_components(mode="strong"):
        node_names = [g.vs[v]["name"] for v in component]
        # --- CASE 1: ISOLATED SUBTOURS (No Depot) ---
        if depot_node_name not in node_names:
            # Apply your capacitated SEC directly to this isolated loop
            sub_arcs = [(i, j) for i in node_names for j in node_names if i != j and (i, j) in mp.x.keys()]
            total_demand = gp.quicksum(mp.theta[i] for i in node_names)
            num_visits = sum(a[i] for i in node_names)
            model.cbLazy(gp.quicksum(mp.x[i, j] for i, j in sub_arcs) <= num_visits - total_demand / mp.vehicle_capacity)
            cuts_added =True
        # --- CASE 2: THE DEPOT COMPONENT ---
        else:
            # We must untangle the individual routes sharing this depot
            # Find all nodes that the vehicle directly visits immediately after leaving the depot
            departures = [j for (i, j) in active_arcs if i == depot_node_name]
            
            # Map out who connects to whom for easy pointer chasing
            next_node = {i: j for (i, j) in active_arcs if i != depot_node_name}
            for start_node in departures:
                route_customers = []
                curr = start_node
                
                # Trace the path until it returns to the depot
                while curr != depot_node_name and curr in next_node:
                    route_customers.append(curr)
                    curr = next_node[curr]
                
                if not route_customers:
                    continue
                
                # Now check this single route's capacity deterministically
                
                total_load = sum(theta[i] for i in route_customers)
                if total_load <= mp.vehicle_capacity:
                    continue
                sub_arcs = [(i, j) for i in route_customers for j in route_customers if i != j and (i, j) in mp.x.keys()]
                total_demand = gp.quicksum(mp.theta[i] for i in route_customers)
                num_visits = gp.quicksum(mp.a[i] for i in route_customers)
                model.cbLazy(gp.quicksum(mp.x[i, j] for i, j in sub_arcs) <= num_visits - total_demand / mp.vehicle_capacity)
                cuts_added = True
    
    return cuts_added

def callback(model: gp.Model, where, mp:Master):
    if where == GRB.Callback.MIPNODE:
        add_fractional_cuts(model, mp)

    if where == GRB.Callback.MIPSOL:
        # x_vals = model.cbGetSolution(mp.x)
        # u_vals = model.cbGetSolution(mp.u)
        # y_vals = model.cbGetSolution(mp.y)
        # for i, j in u_vals.keys():
        #     if u_vals[i,j]:
        #         print(f"{i}->{j}")
        # routes = []
        # for i in mp.visitable_nodes:
        #     if i == mp.depot: continue
        #     if not x_vals[mp.depot, i]: continue
        #     route = [0, i]
        #     curr = i
        #     while curr != mp.depot:
        #         for j in mp.visitable_nodes:
        #             if (curr,j) not in x_vals.keys() or not x_vals[curr,j]:continue
        #             curr = j
        #             break
        #         route.append(curr)
        #     routes.append(route)
        # for route in routes:
        #     print(route)
        # for j, pj in mp.y.keys():
        #     if y_vals[j, pj]:
        #         print(f"{j}->{pj}")
        
        cuts_added = add_cycle_cuts(model, mp)