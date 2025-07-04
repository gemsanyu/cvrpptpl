set K;														#Set of vehicle
set o = {0};  												#Set representing the depot
set C_H;													#Set of home delivery customers 
set C_S;													#Set of self-pickup customers
set C_F;													#Set of flexible customers 
set M; 														#Set of MRT stations with PLs
set M_B; 													#Set of PLs (inside of stations) dummy index
set L_B;													#Set of parcel lockers (outside of stations)
set L:=L_B union M union M_B;								#Set of all parcel lockers
set C:=C_H union C_S union C_F;								#Set of customers
set N:=o union C union L;									#Set of node
set	N1:=o union C_H  union C_F union L;						#Set of nodes related to depot, home delivery, parcel lockers 
set N2:=C_H union C_F union L;
set A1 within {N1,N1};										#Set of arcs between a set of nodes N
param n; 													#Number of vehicles
param t{N,N};												#Travelling time between nodes 
param e{C_S union C_F,L};									#i customer selects to j parcel locker, then 1 otherwise 0
param d{C}; 												#demand for customer 
param p{K};													#Routing cost for each vehicle
param s{N1};												#sevice time at each node 
param Q;													#Carrying capa of each vehicle
param G{L};													#Carrying capa of each PL
param BigM;													#An arbitrary large number
var x{A1,K} binary;											#Arc used by regular vehicle
var u{C_S union C_F,L} binary; 								#i customer is assigned to j parcel locker
var gm{C_H union C_F} binary; 								#i customer is assigned to home delivery service then 1; otherwise 0
var g{N2} binary; 											#i node is served by vehicle then 1; otherwise 0
var h{C_S union C_F} binary; 								#i customer's demand is assigned to parcel locker then 1; otherwise 0
var a{K}>=0;												#Vehicle return to depot
var b{N1}>=0;												#Vehicle depart from node i


#Objective
minimize TC: sum {(i,j) in A1}sum{k in K}p[k]*t[i,j]*x[i,j,k];
					

#flow constraints

subject to C2 {j in N2}:
	sum{i in N1: (i,j) in A1}(sum{k in K} x[i,j,k])=g[j];
	
subject to C3 {k in K}:
	sum{j in N2: (0,j) in A1} x[0,j,k]<=1;
	
subject to C4 {k in K}:
	sum{i in N2: (i,0) in A1}x[i,0,k]<=1;
	
subject to C5 {j in N1, k in K}:
	sum{i in N1: (i,j) in A1 and i != j}x[i,j,k] - sum{i in N1: (j,i) in A1 and i != j}x[j,i,k] = 0;
	
subject to C6 {j in M}:
    sum {i in N1 : (i,j) in A1 and i != j} (sum {k in K} x[i,j,k]) <= 1;

subject to C7:
	sum{j in N1: (0,j) in A1}(sum{k in K}x[0,j,k])<=n;
	
subject to C8:
	sum{i in N1: (i,0) in A1}(sum{k in K}x[i,0,k])<=n;
	
subject to C9 {j in C_H}:
	sum{i in N1: (i,j) in A1 and i != j}(sum{k in K}x[i,j,k])=1;	
	
subject to C10 {j in C_S}:
	sum{i in N1: (i,j) in A1 and i != j}(sum{k in K}x[i,j,k])=0;	

subject to C11 {j in C_F}:
	sum{i in N1: (i,j) in A1 and i != j}(sum{k in K}x[i,j,k])=gm[j];	
	
  
#Scheduling constraints

subject to C13 {i in N1, j in N2: (i,j) in A1}: 
	b[j] >= b[i] + t[i,j] + s[j]-BigM*(1-sum{k in K}x[i,j,k]);

subject to C14 {i in N2, k in K}: 
	a[k] >= b[i] + t[i,0] + s[0];

subject to C15 {k in K}:
	sum{i in N1,j in C_H union C_F: (i,j) in A1}(d[j]*x[i,j,k])+sum{i in N1,j in L: (i,j)in A1}(sum{c in C_S union C_F}d[c]*u[c,j]*x[i,j,k])<= Q; 
	
subject to C17 {l in L}:
	sum{j in C_S}(d[j]*u[j,l])+sum{j in C_F}(d[j]*u[j,l]*(1-gm[j])) <= G[l];


#Synchronization constraints

subject to C18 {i in C_H}: 
		gm[i] = 1;

subject to C19 {i in C_S}: 
		h[i] = 1;
		
subject to C20 {i in C_F}: 
    	gm[i]+h[i] = 1;	

subject to C21 {i in C_S union C_F}: 
		sum{j in L}u[i,j] = h[i];

subject to C22 {i in C_S union C_F, l in L}: 
		u[i,l]<=sum{(j,l) in A1}(sum{k in K}x[j,l,k]);
		
subject to C23 {i in C_H union C_F}: 
		gm[i]<=sum{(i,j) in A1}(sum{k in K}x[i,j,k]);
	
subject to C24 {i in C_S union C_F, j in L }: 
		u[i,j]<=e[i,j];	
	
  
  








