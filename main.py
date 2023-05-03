import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import permutations
from include.castar import aStarSearch


# def createAdjMatrix(a,b):
#     adj_matrix = np.zeros((a*b, a*b))
#     #[0,1,2......b-2,b-1]
#     #[b,b+1,b+2....2b-2,2b-1]
#     #[2b,.............3b-1]
#     #.....
#     #[ab-b,ab-b+1... ab-2,ab-1]
#     #four coner has two ways
#     adj_matrix[0][1] = 1
#     adj_matrix[0][b] = 1
#     adj_matrix[b-1][b-2] = 1
#     adj_matrix[b-1][2*b-1] = 1
#     adj_matrix[a*b-b][a*b-b+1] = 1
#     adj_matrix[a * b - b][a*b - 2*b] = 1
#     adj_matrix[a * b - 1][a * b - 2] = 1
#     adj_matrix[a * b - 1][a * b - 1 - b] = 1
#     #top level node
#     for i in range(1, b-1):
#         adj_matrix[i][i-1] = 1 #left
#         adj_matrix[i][i+1] = 1 #right
#         adj_matrix[i][i+b] = 1 #down
#     #bottom level node
#     for i in range(a*b-b+1, a * b - 1):
#         adj_matrix[i][i-1] = 1 #left
#         adj_matrix[i][i+1] = 1 #right
#         adj_matrix[i][i-b] = 1 #up
#     #left side node
#     for i in range(b, a*b-b, b):
#         adj_matrix[i][i - b] = 1 #up
#         adj_matrix[i][i + b] = 1 #down
#         adj_matrix[i][i + 1] = 1 #right
#     #right
#     for i in range(2*b-1, a*b-1, b):
#         adj_matrix[i][i - b] = 1 #up
#         adj_matrix[i][i + b] = 1 #down
#         adj_matrix[i][i - 1] = 1 #left
#     for i in range(b+1, a * b - b - 1):
#         if (i)%b != 0 and (i + 1)%b != 0:
#             adj_matrix[i][i - b] = 1  # up
#             adj_matrix[i][i + b] = 1  # down
#             adj_matrix[i][i - 1] = 1  # left
#             adj_matrix[i][i + 1] = 1  # right
#     return adj_matrix

def grid_adjacency_matrix(n):
    adj_matrix = np.zeros((n**2, n**2), dtype=int)

    for i in range(n):
        for j in range(n):
            node = i*n + j
            if j < n-1:
                adj_matrix[node, node+1] = 1
                adj_matrix[node+1, node] = 1
            if i < n-1:
                adj_matrix[node, node+n] = 1
                adj_matrix[node+n, node] = 1

    return adj_matrix

def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        # find the shortest cycle in the selected edge list
        tour = subtour(vals)
        print('subtour:')
        print(tour)
        if tour[-1] != end:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j] for i, j in permutations(tour, 2) if adj_matrix[i][j] > 0)<= len(tour)-1)

# Given a tuplelist of edges, find the subtour

def subtour(vals):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys()
                         if vals[i, j] > 0.5);
    path = [start]
    current = start
    while current != end:
        neighbors = [j for i, j in edges.select(current, '*')]
        current = neighbors[0]
        if current == start or current ==[]:
            break
        else:
            path.append(neighbors[0])  
    return path

def csp_risk(adj_matrix, start, end, risk_map, safety_level):
    n = len(adj_matrix)
    
    # Create a Gurobi model
    model = gp.Model()

    # Define decision variables
    x = {}
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] > 0:
                x[i, j] = model.addVar(obj=adj_matrix[i][j], vtype=GRB.BINARY, name='x_{0}_{1}'.format(i, j))

    # Define constraints # log of risk
    model.addConstr(gp.quicksum(x[i, j]*np.log (1-risk_map[j]) for i in range(n) for j in range(n) if adj_matrix[i][j] > 0 and i != j) >= np.log(safety_level))

    # Add flow conservation constraints
    for i in range(n):
        if i != start and i != end:
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if adj_matrix[i][j] > 0) == gp.quicksum(x[j, i] for j in range(n) if adj_matrix[j][i] > 0))

    # Add start and end constraints
    model.addConstr(gp.quicksum(x[start, j] for j in range(n) if adj_matrix[start][j] > 0) == 1)
    model.addConstr(gp.quicksum(x[i, end] for i in range(n) if adj_matrix[i][end] > 0) == 1)

    # Add subcontour remove constraints


    # Set objective function
    model.modelSense = GRB.MINIMIZE

    # Solve the model
    model._vars = x
    model.Params.LazyConstraints = 1
    model.optimize(subtourelim)

    # Print the solution
    if model.status == GRB.OPTIMAL:
        vals = model.getAttr('X', x)
        tour = subtour(vals)
        print('Optimal tour: %s' % str(tour))
        print('Total cost:', model.objVal)
    else:
        print('No solution found.')
        
a = 9
b = 9
start = 0
end = 72
graph = grid_adjacency_matrix(a)
#create a graph
# Define the graph as an adjacency matrix
adj_matrix = graph
safe_c = 0.9

risk = []
if a == 4:
    risk = [0, 0, 0, 0.11, 0.5, 0.5, 0.01, 0, 0.5,0.5,0.5,0,0,0,0,0]
if a == 3:
    risk = [0.01, 0.02, 0, 0.1, 0.1, 0.01, 0, 0, 0]
if a == 9:
    risk.append([0.001, 0, 0, 0, 0, 0, 0, 0, 0])
    risk.append([1, 1, 1, 1, 1, 1, 1, 0, 0])
    risk.append([1, 1, 1, 1, 1, 1, 1, 0, 0])
    risk.append([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 1])
    risk.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
    risk.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
    risk.append([0, 0, 0, 0, 0, 0, 1, 1, 0])
    risk.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
    risk.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
    flatten_list = [j for sub in risk for j in sub]
    risk = flatten_list
    risk = [x+0.001 for x in risk]
#risk = [0.001]*a**2
safe_c = 0.9
path, nodeList, count, explored = aStarSearch(start,end, adj_matrix, risk, safe_c)
print(path)
csp_risk(adj_matrix, start, end, risk, safe_c)
print(path)
#plot
# n = len(graph)  # number of nodes in the graph
# adj_matrix = np.array(adj_matrix)
# # create a graph from the adjacency matrix
# path, nodeList, count, explored = aStarSearch(start,end, adj_matrix, risk, safe_c)
# print(path)
#
# # plot the graph
# pos = {(int(i), int(j)): [i, j] for i in range(3) for j in range(3)}
# G = nx.from_numpy_array(adj_matrix)
# pos = nx.spring_layout(G, pos=pos)
# nx.draw_networkx_nodes(G, pos, node_size=500)
# nx.draw_networkx_edges(G, pos)
# nx.draw_networkx_labels(G, pos, font_size=16, font_color='white')
#
# # set the plot title and axis labels
# plt.title('Graph of the Adjacency Matrix')
# plt.axis('off')
#
# # show the plot
# plt.show()


# Define the number of nodes and the start and end nodes
n = len(graph[0])


# Create a Gurobi model
model = gp.Model()

# Define decision variables
# x = {}
# for i in range(n):
#     for j in range(n):
#         if graph[i][j] > 0:
#             x[i, j] = model.addVar(obj=graph[i][j], vtype=GRB.BINARY, name='x_{0}_{1}'.format(i, j))
#
# # Define constraints # need to change the nonlinear one
# model.addConstr(gp.quicksum(
#     x[i, j] * np.log(1 - risk[j]) for i in range(n) for j in range(n) if graph[i][j] > 0 and i != j) >= np.log(safe_c))
#
# # Add flow conservation constraints
# for i in range(n):
#     if i != start and i != end:
#         model.addConstr(gp.quicksum(x[i, j] for j in range(n) if graph[i][j] > 0) == gp.quicksum(
#             x[j, i] for j in range(n) if graph[j][i] > 0))
#
# # Add start and end constraints
# model.addConstr(gp.quicksum(x[start, j] for j in range(n) if graph[start][j] > 0) == 1)
# model.addConstr(gp.quicksum(x[i, end] for i in range(n) if graph[i][end] > 0) == 1)
#
# # Add subcontour remove constraints
# sub_contours = [(i, j, l, m) for i in range(n) for j in range(n) for l in range(n) for m in range(n)
#                 if i != j and j != l and l != m and m != i and
#                 (adj_matrix[i][j] == 1 or adj_matrix[j][i] == 1) and
#                 (adj_matrix[j][l] == 1 or adj_matrix[l][j] == 1) and
#                 (adj_matrix[l][m] == 1 or adj_matrix[m][l] == 1) and
#                 (adj_matrix[m][i] == 1 or adj_matrix[i][m] == 1)]
#
# for i, j, l, m in sub_contours:
#     model.addConstr(x[i, j] + x[j, l] + x[l, m] + x[m, i] <= 3)
#
# # Set objective function
# model.modelSense = GRB.MINIMIZE
#
# # Solve the model
# model.optimize()
#
# # Print the solution
# if model.status == GRB.OPTIMAL:
#     print('Optimal solution:')
#     path = [start]
#     node = start
#     while node != end:
#         for j in range(n):
#             if graph[node][j] > 0 and x[node, j].x > 0.5:
#                 path.append(j)
#                 print(path)
#                 node = j
#                 break
#     print('->'.join(str(node) for node in path))
#     print('Total cost:', model.objVal)
# else:
#     print('No solution found.')
#
# nodeList = {}  # {property：node}
# nodeList[(0,0.99)] = 1
# a = nodeList[(0,0.99)]
# print(a)
# path, nodeList, count, explored = aStarSearch(start,end, adj_matrix, risk, safe_c)
# print(path)
