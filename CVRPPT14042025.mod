set K;														#Set of vehicle
set o = {0};  												#Set representing the depot
set C_H;													#Set of home delivery customers 
set C_S;													#Set of self-pickup customers
set C_F;													#Set of flexible customers 
set M; 														#Set of MRT stations 
set L_B;													#Set of parcel lockers (outside of stations)
set L:=L_B union M;								            #Set of all parcel lockers
set C:=C_H union C_S union C_F;								#Set of customers
param nplus1 := card(C) + card(L) + 1;                      #dummy end depot index    
set o_end := {nplus1};                          
set N:=o union C union L union o_end;									#Set of node
set	N1:=o union C_H  union C_F union L;         #Set of nodes related to depot, home delivery, parcel lockers 
set N2:=C_H union C_F union L;
set A1 within {(N1 union o_end), (N1 union o_end) };										#Set of arcs between a set of nodes N
set A2 within {M,M};										#Set of MRT lines
param n; 													#Number of vehicles
param t{N,N};												#Travelling time between nodes 
param w{A2};												#Transportation cost on MRT lines
param V{A2};												#Carrying capa of each MRT line
param e{C_S union C_F,L};									#i customer selects to j parcel locker, then 1 otherwise 0
#param f{L};												#Compensation cost for each PL
param d{C}; 												#demand for customer 
param p{K};													#Routing cost for each vehicle
param s{N1};												#sevice time at each node 
param Q;													#Carrying capa of each vehicle
param G{L};													#Carrying capa of each PL
param BigM;													#An arbitrary large number


var x{A1,K} binary;											#Arc used by regular vehicle
var y{A2} binary; 											#Arc used by MRT line
var u{C_S union C_F,L} binary; 								#i customer is assigned to j parcel locker
var gm{C_H union C_F} binary; 								#i customer is assigned to home delivery service then 1; otherwise 0
var h{C_S union C_F} binary; 								#i customer's demand is assigned to parcel locker then 1; otherwise 0
var node_active{N2} binary; 								#i node is served by vehicle then 1; otherwise 0
var ntd{N1}>=0;                                             #Node total demand
var lockerload{L}>=0;                                       #var to help holding locker load
var b{N1}>=0;												#Vehicle depart from node i
var vehicle_assignment{N2, K} binary;                       #node to vehicle assignment flag
var n_cust{K}>=0;                                           #Number of customers assigned to vehicle k

#Objective
minimize TC: sum {i in N1} sum{j in N2 union o_end: (i,j) in A1} sum{k in K}p[k]*t[i,j]*x[i,j,k]+sum{(ss,es) in A2}(y[ss,es]*lockerload[es]*w[ss,es]);
# +sum{i in C_S union C_F}(sum{j in L}f[j]*u[i,j]);

# customer assignment
subject to C2 {j in C_H}:
    gm[j]=1;
subject to C3 {j in C_F}:
    gm[j] + h[j] = 1;
subject to C4 {j in C_S}:
    h[j]=1;
subject to C5 {j in C_S union C_F}:
    sum{l in L: e[j,l]=1}(u[j, l])=h[j];
subject to C6 {j in C_S union C_F, l in L}:
    u[j,l] <= e[j,l];

# locker capacity
subject to C7 {l in L}:
    lockerload[l] = sum{j in C_S union C_F}(u[j,l]*d[j]);
subject to C8 {l in L}:
    lockerload[l] <=G[l];

# MRT arc usage (ss startStation, es endStation)
subject to C9 {(ss, es) in A2}: # if mrt is active, the end station must have a load
    y[ss, es] <= lockerload[es];
subject to C9b {(ss, es) in A2}: # if mrt is active, the end station' load must be less than MRT capacity
    lockerload[es] <= (1-y[ss,es])*Q + y[ss,es]*V[ss,es];


# demand aggregation to every node
subject to C10 {j in C_H union C_F}:
    ntd[j] = d[j]*gm[j];
subject to C11 {l in L}: #aggregate locker load including mrt loads
    ntd[l] = lockerload[l] 
            - sum{ss in L: (ss, l) in A2}y[ss,l]*lockerload[l]
            + sum{es in L: (l, es) in A2}y[l,es]*lockerload[es];

# finding out which node is active or has to be served by a vehicle
subject to C12 {j in N2}:
    ntd[j]<=Q*node_active[j];
subject to C13 {j in N2}:
    ntd[j]>=node_active[j];

#routing constraint
subject to C14_Inflow {j in N2}:
    sum{i in o union N2: (i,j) in A1 and i!=j} (sum{k in K} x[i,j,k]) = node_active[j];
subject to C15_Outflow {j in N2}:
    sum{i in N2 union o_end: (j,i) in A1 and i!=j} (sum{k in K} x[j,i,k]) = node_active[j];
subject to C15c_NoDepartFromEndDepot:
    sum{j in N2 union o} sum{k in K} x[nplus1,j,k]=0;
subject to C15b_PerVehFlow {j in N2, k in K}:
    sum {i in N: (i,j) in A1 and i != j} x[i,j,k]
  = sum {i in N: (j,i) in A1 and i != j} x[j,i,k];
subject to C16_StartDepot {k in K}: # vehicle go from N2 or go to depot directly (unused vehicle)
    sum {j in N2 union o_end: (0,j) in A1} x[0,j,k] = 1;
subject to C17_ReturnDepot {k in K}:
    sum {j in N2 union o: (j,nplus1) in A1} x[j,nplus1,k] = 1;
subject to C18 {i in N1, j in N2: (i,j) in A1}: 
	b[j] >= b[i] + t[i,j] + s[j]-BigM*(1-sum{k in K}x[i,j,k]);
    

#vehicle capacity
subject to C19 {k in K}: 
    sum {i in N1} (sum {j in N2: (i,j) in A1} x[i,j,k]*ntd[j]) <= Q; 

#add symmetry breaking but only if vehicle costs and capacity are the same
subject to C20 {j in N2, k in K}:
    sum {i in N1: (i,j) in A1} x[i,j,k] = vehicle_assignment[j,k];
subject to C21 {k in K}:
    sum {i in N2} vehicle_assignment[i,k] = n_cust[k]; 
# COS
# subject to C22 {k in K: k>1}:
#     sum {i in N1} sum{j in N2 union o_end: (i,j) in A1} p[k]*t[i,j]*x[i,j,k] <= sum {i in N1} sum{j in N2 union o_end: (i,j) in A1} p[k-1]*t[i,j]*x[i,j,k-1];
# VC
subject to C22 {k in K: k>1}:
    n_cust[k] <= n_cust[k-1];
# # VR
# subject to C23 {i in N2}:
#     sum {k in K: k>i} vehicle_assignment[i,k] = 0;
# # Hierarchical constraints Type 1 (HC1)
# subject to C23 {i in C_H, k in K: k>1}:
#     vehicle_assignment[i,k] <= sum {j in N2: j < i} vehicle_assignment[j, k-1];
