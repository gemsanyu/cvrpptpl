import os
import pathlib
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from problem.customer import Customer
from problem.locker import Locker
from problem.mrt_line import MrtLine
from problem.node import Node
from problem.vehicle import Vehicle
from problem.mrt_station import MrtStation
from scipy.spatial import distance_matrix as dm_func


class Cvrpptpl:
    def __init__(self,
                 depot: Node,
                 customers: List[Customer],
                 lockers: List[Locker],
                 mrt_lines : List[MrtLine],
                 vehicles: List[Vehicle],
                 distance_matrix: Optional[np.ndarray] = None,
                 instance_name:str = None,
                 complete_mrt_lines:Optional[List[List[MrtStation]]] = None
                 ) -> None:
        self.depot = depot
        self.depot_coord = depot.coord
        self.customers = customers
        self.lockers = lockers
        self.nodes = [depot] + customers + lockers
        self.mrt_lines = mrt_lines
        self.complete_mrt_lines = complete_mrt_lines
        mrt_lockers_idx = [mrt_line.start_station.idx for mrt_line in mrt_lines] + [mrt_line.end_station.idx for mrt_line in mrt_lines]
        self.non_mrt_lockers = [locker for locker in lockers if not locker.idx in mrt_lockers_idx]
        self.vehicles = vehicles
        for vi, vec in enumerate(self.vehicles):
            vec.idx = vi+1
        self.num_customers = len(customers)
        self.num_lockers = len(lockers)
        self.num_vehicles = len(vehicles)
        self.num_nodes = len(self.nodes)
        coords = [node.coord for node in self.nodes]
        self.coords = np.stack(coords, axis=0)
        self.distance_matrix = distance_matrix
        if distance_matrix is None:
            self.distance_matrix = dm_func(self.coords, self.coords)
            self.distance_matrix = np.around(self.distance_matrix, decimals=2)
        self.filename = self.init_filename(instance_name)
        
        # this is for solver actually
        # infos spread into list or np.ndarray for easier/faster access later
        self.service_times: np.ndarray = np.asanyarray([node.service_time if (isinstance(node, Customer) or isinstance(node, Locker)) else 0 for node in self.nodes], dtype=float)
        self.demands: np.ndarray = np.zeros([self.num_nodes,],dtype=int)
        for customer in customers:
            self.demands[customer.idx] = customer.demand
        self.mrt_line_stations_idx: np.ndarray = np.empty([len(mrt_lines),2], dtype=int)
        for i, mrt_line in enumerate(mrt_lines):
            self.mrt_line_stations_idx[i, :] = (mrt_line.start_station.idx, mrt_line.end_station.idx)        
        self.incoming_mrt_lines_idx: List[int] = [None for _ in range(self.num_nodes)]
        for i, mrt_line in enumerate(mrt_lines):
            self.incoming_mrt_lines_idx[mrt_line.end_station.idx] = i
        self.vehicle_capacities: np.ndarray = np.asanyarray([vehicle.capacity for vehicle in self.vehicles], dtype=int)
        self.vehicle_costs: np.ndarray = np.asanyarray([vehicle.cost for vehicle in self.vehicles], dtype=float)
        self.mrt_line_costs: np.ndarray = np.asanyarray([mrt_line.cost for mrt_line in self.mrt_lines], dtype=float)
        self.mrt_line_capacities: np.ndarray = np.asanyarray([mrt_line.freight_capacity for mrt_line in self.mrt_lines], dtype=int)
        self.destination_alternatives : List[List[int]] = []
        for node in self.nodes:
            alternatives = [node.idx]
            if isinstance(node, Customer):
                if node.is_self_pickup:
                    alternatives = node.preferred_locker_idxs
                elif node.is_flexible:
                    alternatives = [node.idx]+node.preferred_locker_idxs
            self.destination_alternatives += [alternatives]
        self.locker_capacities: np.ndarray = np.asanyarray([node.capacity if isinstance(node, Locker) else 0 for node in self.nodes])
    
        graph_and_legends = self.generate_graph()
        self.graph: nx.MultiGraph = graph_and_legends[0]
        self.graph_legend_handles: List[Line2D] = graph_and_legends[1]
        
        
    def visualize_graph(self):
        g = self.graph
        legend_handles = self.graph_legend_handles
        pos = nx.get_node_attributes(g, "pos")
        for node, data in g.nodes(data=True):
            nx.draw_networkx_nodes(g, pos, nodelist=[node], node_size=100, node_color=data["color"], node_shape=data['shape'])
        
        # add locker assignment edges
        for customer in self.customers:
            if not (customer.is_flexible or customer.is_self_pickup):
                continue
            for dest_idx in self.destination_alternatives[customer.idx]:
                if dest_idx == customer.idx:
                    continue
                u, v = customer.idx, dest_idx
                edge = (u,v)
                if v==-1:
                    continue
                nx.draw_networkx_edges(g, pos, edgelist=[edge], edge_color="black", style=":")
        
        # add mrt line usage edges
        if len(self.mrt_lines)>0:
            for u, v, key, data in g.edges(keys=True, data=True):
                incoming_mrt_line_idx = self.incoming_mrt_lines_idx[v]
                if not (key=="mrt-line"):
                    continue
                edge = (u,v)
                nx.draw_networkx_edges(g, pos, edgelist=[edge], edge_color=data["color"], style=data["style"], arrows=True, arrowstyle='->', arrowsize=20)
        
        plt.legend(handles=legend_handles, title="Graph Information", loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.7)
        plt.show()
        
        
        
    def generate_graph(self):
        g = nx.MultiDiGraph()
        shapes = []
        colors = []
        labels = []
        legend_handles = []
        unique_legend_handles = []
        for node in self.nodes:
            # default for depot
            shape = "s"
            color = "green"
            label = "depot"
            if isinstance(node, Customer):
                shape = "o"
                if node.is_self_pickup:
                    color = "red"
                    label = "Self-pickup Cust"
                elif node.is_flexible:
                    color = "yellow"
                    label = "Flexible Cust"
                else:
                    color = "blue"
                    label = "Home-delivery Cust"
            elif isinstance(node, Locker):
                if node.idx in self.mrt_line_stations_idx:
                    shape = "H"
                    label = "Terminal"
                else:
                    shape = "^"
                    label = "Locker"
                color = "brown"
            shapes += [shape]
            colors += [color]
            labels += [label]
            legend_handle = Line2D([0], [0], marker=shape, color="w", 
                         markerfacecolor=color, markersize=10, 
                         label=label, markeredgecolor="black")
            legend_handle_str = f"{shape}-{color}-{label}"
            if legend_handle_str not in unique_legend_handles:
                legend_handles += [legend_handle]
                unique_legend_handles += [legend_handle_str]
        for node_1 in self.nodes:
            for node_2 in self.nodes:
                if node_1.idx == node_2.idx:
                    continue
                g.add_edge(node_1.idx, node_2.idx, key="general-edge", style="-", color="black")
                
        for customer in self.customers:
            for l_idx in customer.preferred_locker_idxs:
                g.add_edge(customer.idx, l_idx, key="locker-preference", style=":", color="gray")
        for mrt_line in self.mrt_lines:
            g.add_edge(mrt_line.start_station.idx, mrt_line.end_station.idx, key="mrt-line", style="--", color="blue")        
            g.add_node(mrt_line.start_station.idx)
        g.add_nodes_from([(node.idx, {"pos":node.coord, "shape":shapes[node.idx],"color":colors[node.idx]}) for node in self.nodes])
        edge_styles = {
            "Route": {"color": "black", "style": "-", "arrowstyle": "->"},
            "MRT Line": {"color": "blue", "style": "--", "arrowstyle": "->"},
            "Locker Usage": {"color": "black", "style": ":"}
        }
        edge_legend_handles = [
            Line2D([0], [0], color=style["color"], linestyle=style["style"], linewidth=2, 
                marker="", label=edge_type) 
            for edge_type, style in edge_styles.items()
        ]
        legend_handles += edge_legend_handles
        return g, legend_handles
    
    # def visualize(self):
        
    #     pos = nx.get_node_attributes(g, "pos")
    #     for node, data in g.nodes(data=True):
    #         nx.draw_networkx_nodes(g, pos, nodelist=[node], node_size=100, node_color=data["color"], node_shape=data['shape'])
    #     for u, v, key, data in g.edges(keys=True, data=True):
    #         edge = (u,v)
    #         nx.draw_networkx_edges(g, pos, edgelist=[edge], edge_color=data["color"], style=data["style"])
    #     plt.show()
        
    def init_filename(self, instance_name):
        instance_dir = pathlib.Path(".")/"instances"
        final_filename = None
        txt_filepath = None
        ampl_filepath = None
        txt_filepath = instance_dir/(instance_name+".txt")
        ampl_filepath = instance_dir/(instance_name+"_ampl.txt")
        if not os.path.exists(txt_filepath.absolute()) and not os.path.exists(ampl_filepath.absolute()):
            return instance_name
        for save_idx in range(100000):
            final_filename = instance_name+"_idx_"+str(save_idx)
            txt_filepath = instance_dir/(final_filename+".txt")
            ampl_filepath = instance_dir/(final_filename+"_ampl.txt")
            if not os.path.exists(txt_filepath.absolute()) and not os.path.exists(ampl_filepath.absolute()):
                break
        return final_filename
        
    def save_to_file(self):
        instance_dir = pathlib.Path(".")/"instances"
        filepath = instance_dir/(self.filename+".txt")
        instance_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        lines += ["vehicles\n"]
        lines += ["vehicle_idx, capacity, cost_per_unit_distance\n"]
        for vehicle in self.vehicles:
            lines += [str(vehicle)]
        lines += ["depot\n"]
        lines += ["node_idx,x,y\n"]
        lines += ["0,"+str(self.depot.coord[0])+","+str(self.depot.coord[1])+"\n"]
        lines += ["home delivery customers\n"]
        lines += ["node_idx,x,y,service_time,demand\n"]
        for customer in self.customers:
            if not (customer.is_self_pickup or customer.is_flexible):
                lines += [str(customer)]
        lines += ["self pickup customers\n"]
        lines += ["node_idx,x,y,service_time,demand,locker_idxs\n"]
        for customer in self.customers:
            if customer.is_self_pickup:
                lines += [str(customer)]
        lines += ["flexible customers\n"]
        lines += ["node_idx,x,y,service_time,demand,locker_idxs\n"]
        for customer in self.customers:
            if customer.is_flexible:
                lines += [str(customer)]
        lines += ["lockers\n"]
        lines += ["node_idx,x,y,service_time,capacity,cost_per_unit_good\n"]
        for locker in self.lockers:
            lines += [str(locker)]
        lines += ["mrt lines\n"]
        lines += ["start_node_idx,end_node_idx,freight_capacity,cost_per_unit_good\n"]
        for mrt_line in self.mrt_lines:
            lines += [str(mrt_line)]
        lines += ["distance matrix\n"]
        node_idxs_str = ","+"".join([str(i)+"," for i in range(self.num_nodes)])+"\n"
        lines += node_idxs_str
        for i in range(self.num_nodes):
            line = str(i)+","
            for j in range(self.num_nodes):
                line += str(self.distance_matrix[i,j])
                if j < self.num_nodes-1:
                    line += ","
            lines += [line+"\n"]
        
        with open(filepath.absolute(), "w") as save_file:
            save_file.writelines(lines)

    def save_to_ampl_file(self, is_v2: bool):
        lines = []
        vehicles_idx_str = "\t".join([str(vehicle.idx) for vehicle in self.vehicles])
        lines += ["set K:= "+vehicles_idx_str+";\n"]
        hd_custs_idx = [customer.idx for customer in self.customers if not (customer.is_flexible or customer.is_self_pickup)]
        hd_custs_idx_str = "\t".join([str(c_idx) for c_idx in hd_custs_idx])
        lines += ["set C_H:= "+hd_custs_idx_str+";\n"]
        sp_custs_idx = [customer.idx for customer in self.customers if customer.is_self_pickup]
        sp_custs_idx_str = "\t".join([str(c_idx) for c_idx in sp_custs_idx])
        lines += ["set C_S:= "+sp_custs_idx_str+";\n"]
        f_custs_idx = [customer.idx for customer in self.customers if customer.is_flexible]
        f_custs_idx_str = "\t".join([str(c_idx) for c_idx in f_custs_idx])
        lines += ["set C_F:= "+f_custs_idx_str+";\n"]
        
        mrts_idx = []
        if len(self.mrt_line_stations_idx)>0:
            mrts_idx = list(set(np.concat(self.mrt_line_stations_idx).tolist()))
            mrts_idx.sort()
            mrts_idx_str = "\t".join([str(mrt_idx) for mrt_idx in mrts_idx])
            lines += ["set M:= "+mrts_idx_str+";\n"]

        non_mrt_lockers_idx = [locker.idx for locker in self.non_mrt_lockers]
        non_mrt_lockers_idx_str = "\t".join([str(l_idx) for l_idx in non_mrt_lockers_idx])
        lines += ["set L_B:= "+non_mrt_lockers_idx_str+";\n"]
        lines += ["set A1:=\n"]
        
        
        # if version 1, dont add lines between mrt lines for regular vehicles
        # if version 2, add lines between mrt lines for regular vehicles
        reachable_nodes_idx = [0] + hd_custs_idx + f_custs_idx + mrts_idx + non_mrt_lockers_idx
        for i in reachable_nodes_idx:
            reachable_nodes_idx_str = [str(idx) for idx in reachable_nodes_idx if idx!=i]    
            if i in mrts_idx and not is_v2:
                end_station_idx:int 
                for mrt_line_station_idx in self.mrt_line_stations_idx:
                    if mrt_line_station_idx[0]==i:
                        end_station_idx = mrt_line_station_idx[1]
                        break
                reachable_nodes_idx_str = [str(idx) for idx in reachable_nodes_idx if idx!=i and idx!=end_station_idx]
            line = f"({i},*) "+" ".join(reachable_nodes_idx_str)+"\n"
            lines += [line]
        lines+=[";\n"]
        
        if len(self.mrt_lines)>0:
            lines += ["set A2:=\n"]
            for mrt_line in self.mrt_lines:
                lines += [f"({mrt_line.start_station.idx},*) {mrt_line.end_station.idx}\n"]
            lines+=[";\n"]
        
        lines+= ["param BigM:=99999;\n"] 
        lines+= [f"param n:= {self.num_vehicles};\n"]
        
        lines+= ["param d:=\n"]
        for customer in self.customers:
            lines+= [f"{customer.idx}\t{customer.demand}\n"]
        lines+= [";\n"]
        
        lines+= [f"param Q:={self.vehicles[0].capacity};\n"]
        
        if len(self.mrt_lines)>0:
            lines+= ["param w:=\n"]
            for mrt_line in self.mrt_lines:
                lines+= [f"[{mrt_line.start_station.idx},*] {mrt_line.end_station.idx} {mrt_line.cost}\n"]
            lines+= [";\n"]
        
        if len(self.mrt_lines)>=1:
            lines+= ["param V:=\n"]
            for mrt_line in self.mrt_lines:
                lines+= [f"[{mrt_line.start_station.idx},*] {mrt_line.end_station.idx} {mrt_line.freight_capacity}\n"]
            lines+= [";\n"]
        
        # lines+= ["param f:=\n"]
        # for locker in self.lockers:
        #     lines+= [f"{locker.idx}\t{locker.cost}\n"]
        # lines+= [";\n"]
        
        lines+= ["param e:\n"]
        lockers_idx = [locker.idx for locker in self.lockers]
        lines+= ["\t"+"\t".join([str(l_idx) for l_idx in lockers_idx])+":=\n"]
        for customer in self.customers:
            if not (customer.is_self_pickup or customer.is_flexible):
                continue
            line = f"{customer.idx}\t"
            for locker in self.lockers:
                if locker.idx in customer.preferred_locker_idxs:
                    line+= "1\t"
                else:
                    line+= "0\t"
            line += "\n"
            lines+=[line]
        lines+= [";\n"]    
            
        # lines+= ["param r:\n"]
        # lines+= ["\t"+mrts_idx_str+":=\n"]
        # for mrt_idx_1 in mrts_idx:
        #     line = str(mrt_idx_1)+"\t"
        #     for mrt_idx_2 in mrts_idx:
        #         is_connected = "0"
        #         for mrt_line in self.mrt_lines:
        #             if mrt_line.start_station.idx == mrt_idx_1 and mrt_line.end_station.idx == mrt_idx_2:
        #                 is_connected = "1"
        #         line += is_connected+"\t"
        #     lines+=[line+"\n"]
        # lines+= [";\n"] 
        
        lines+= ["param p:=\n"]
        for vehicle in self.vehicles:
            lines+= [f"{vehicle.idx}\t{vehicle.cost}\n"]
        lines+= [";\n"]
        
        lines+= ["param G:=\n"]
        for locker in self.lockers:
            lines+= [f"{locker.idx}\t{locker.capacity}\n"]
        lines+= [";\n"]
        
        lines+= ["param s:=\n"]
        lines+= ["0\t0\n"]
        for customer in self.customers:
            if customer.is_self_pickup:
                continue
            lines+= [f"{customer.idx}\t{customer.service_time}\n"]
        for locker in self.lockers:
            lines+= [f"{locker.idx}\t{locker.service_time}\n"]
        lines+= [";\n"]
        
        lines+= ["param t:\n"]
        line = "\t"+"\t".join([str(i) for i in range(self.num_nodes)])+":=\n"
        lines+= [line]
        for i in range(self.num_nodes):
            lines+= [f"{str(i)}\t"+"\t".join(str(self.distance_matrix[i,j]) for j in range(self.num_nodes)  )+"\n"]
        lines+= [";\n"]
        instance_dir = pathlib.Path(".")/"instances"
        filename = self.filename
        if is_v2:
            filename += "_v2_"
        else:
            filename += "_v1_"

        filepath = instance_dir/(filename+"ampl_.txt")
        instance_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath.absolute(), "w") as save_file:
            save_file.writelines(lines)
        

def read_from_file(filename:str)->Cvrpptpl:
    directory = pathlib.Path("")/"instances"
    filepath = directory/filename
    vehicles:List[Vehicle] = []
    customers:List[Customer] = []
    lockers:List[Locker] = []
    mrt_lines:List[MrtLine] = []
    locker_idx_dict: Dict[int, Locker] = {}
    with open(filepath.absolute(), "r") as f:
        lines = f.readlines()
        line_idx = 2
        # 1 vehicles
        while "depot" not in lines[line_idx]:
            line = lines[line_idx]
            line = line.split(",")
            idx, capacity, cost = int(line[0]), int(line[1]), int(line[2])
            vehicle = Vehicle(idx, capacity, cost)
            vehicles += [vehicle]
            line_idx += 1
        line_idx += 2
        line = lines[line_idx].split(",")
        # 2 depot coord
        depot_coord = np.asanyarray([int(line[1]), int(line[2])], dtype=int)
        line_idx += 3
        # 3 hd customers
        while "self pickup" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, service_time, demand = [int(v) for v in line]
           coord = np.asanyarray([x,y], dtype=float)
           hd_customer = Customer(idx, coord, service_time, demand)
           customers += [hd_customer]
           line_idx += 1
        line_idx += 2
        # 4 self pickup
        while "flexible" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, service_time, demand, locker_idxs_str = line
           idx, x, y, service_time, demand = int(idx), int(x), int(y), int(service_time), int(demand)
           locker_idxs_str = locker_idxs_str.split("-")
           locker_idxs = [int(locker_idx_str) for locker_idx_str in locker_idxs_str]
           coord = np.asanyarray([x,y], dtype=float)
           sp_customer = Customer(idx, coord, service_time, demand, is_self_pickup=True, preferred_locker_idxs=locker_idxs)
           customers += [sp_customer]
           line_idx += 1
        line_idx += 2
        # 5 flexible
        while "lockers" not in lines[line_idx]:
           line = lines[line_idx].split(",")
           idx, x, y, service_time, demand, locker_idxs_str = line
           idx, x, y, service_time, demand = int(idx), int(x), int(y), int(service_time), int(demand)
           locker_idxs_str = locker_idxs_str.split("-")
           locker_idxs = [int(locker_idx_str) for locker_idx_str in locker_idxs_str]
           coord = np.asanyarray([x,y], dtype=float)
           fx_customer = Customer(idx, coord, service_time, demand, is_flexible=True, preferred_locker_idxs=locker_idxs)
           customers += [fx_customer]
           line_idx += 1
        line_idx += 2
        # 6 lockers
        while "mrt" not in lines[line_idx]:
            line = lines[line_idx].split(",")
            idx,x,y,service_time,capacity,cost = int(line[0]),float(line[1]),float(line[2]),int(line[3]),int(line[4]),int(line[5])
            coord = np.asanyarray([x,y], dtype=float)
            locker = Locker(idx, coord, service_time, capacity, cost)
            locker_idx_dict[idx] = locker
            lockers += [locker]
            line_idx += 1
        line_idx += 2
        # 7 mrt lines
        while "distance" not in lines[line_idx]:
            line = lines[line_idx].split(",")
            start_idx, end_idx, capacity, cost = [int(v) for v in line]
            start_station, end_station = locker_idx_dict[start_idx], locker_idx_dict[end_idx]
            mrt_line = MrtLine(start_station, end_station, start_station.service_time, cost, capacity)
            mrt_lines += [mrt_line]
            line_idx += 1
        line_idx += 2
        # 8 precomputed distance matrix
        num_nodes = len(customers) + len(lockers) + 1
        distance_matrix = np.zeros([num_nodes, num_nodes], dtype=float)
        for i in range(num_nodes):
            line = lines[line_idx].split(",")
            line = [float(v) for v in line]
            idx, distances = line[0], np.asanyarray(line[1:])
            distance_matrix[i, :] = distances
            line_idx += 1
    depot = Node(0, depot_coord)
    problem = Cvrpptpl(depot,
                       customers,
                       lockers,
                       mrt_lines,
                       vehicles,
                       distance_matrix,
                       instance_name="")
    return problem
    
    
    