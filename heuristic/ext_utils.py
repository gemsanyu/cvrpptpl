import matplotlib.pyplot as plt
import networkx as nx

from heuristic.solution import Solution
from problem.cvrpptpl import Cvrpptpl

def visualize_solution(problem: Cvrpptpl, solution: Solution):
    g = problem.graph
    legend_handles = problem.graph_legend_handles
    pos = nx.get_node_attributes(g, "pos")
    for node, data in g.nodes(data=True):
        nx.draw_networkx_nodes(g, pos, nodelist=[node], node_size=100, node_color=data["color"], node_shape=data['shape'])
    # add route edges
    for route in solution.routes:
        if len(route) == 1:
            continue
        for i in range(len(route)):
            u = route[i]
            v = route[(i+1)%len(route)]
            edge = (u,v)
            nx.draw_networkx_edges(g, pos, edgelist=[edge], edge_color="black", style="-", arrows=True, arrowstyle='->', arrowsize=20)
    # add locker assignment edges
    for customer in problem.customers:
        if solution.package_destinations[customer.idx] == customer.idx:
            continue
        u, v = customer.idx, solution.package_destinations[customer.idx]
        edge = (u,v)
        if v==-1:
            continue
        nx.draw_networkx_edges(g, pos, edgelist=[edge], edge_color="black", style=":")
    
    # add mrt line usage edges
    for u, v, key, data in g.edges(keys=True, data=True):
        incoming_mrt_line_idx = problem.incoming_mrt_lines_idx[v]
        if not (key=="mrt-line") or not solution.mrt_usage_masks[incoming_mrt_line_idx]:
            continue
        edge = (u,v)
        nx.draw_networkx_edges(g, pos, edgelist=[edge], edge_color=data["color"], style=data["style"], arrows=True, arrowstyle='->', arrowsize=20)
    plt.legend(handles=legend_handles, title="Graph Information", loc="upper left", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)
    plt.show()