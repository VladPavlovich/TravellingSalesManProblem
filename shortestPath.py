import matplotlib.pyplot as plt
import networkx as nx
import heapq
import math
import time

G = nx.Graph()

G.add_edge("a", "b", weight=6)
G.add_edge("a", "h", weight=6.40)
G.add_edge("a", "i", weight=5)
G.add_edge("b", "c", weight=4.12)
G.add_edge("b", "d", weight=5)
G.add_edge("c", "d", weight=4)
G.add_edge("c", "e", weight=5.66)
G.add_edge("c", "n", weight=3)
G.add_edge("d", "e", weight=4)
G.add_edge("e", "f", weight=4)
G.add_edge("e", "g", weight=6.40)
G.add_edge("f", "g", weight=5)
G.add_edge("h", "f", weight=9.49)
G.add_edge("h", "k", weight=6.40)
G.add_edge("h", "j", weight=3)
G.add_edge("i", "j", weight=5.39)
G.add_edge("j", "k", weight=5.10)
G.add_edge("k", "l", weight=1)
G.add_edge("k", "g", weight=9.06)
G.add_edge("l", "m", weight=6)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos = {"a": (1, 1), "b": (1, 7), "c": (0, 11), "d": (4, 11), "e": (4, 15), "f": (8, 15), "g": (8, 20),
       "h": (5, 6), "i": (6, 1), "j": (8, 6), "k": (9, 11), "l": (10, 11), "m": (10, 5), "n": (0, 14)}
g_dict = nx.get_edge_attributes(G, "weight")

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()

# This is how to get the weight of an edge Graph[firstNode][secondNode]["weight"]:
#   print(G["a"]["b"]["weight"])
# dijkstra's: finds the shortest path between a starting node and every other node in the graph
def dijkstra(graph, src):
    Q = []
    # dist: {'a': inf, 'b': inf, 'c': inf, 'd': inf, 'e': inf, 'f': inf}
    dist = dict(graph.nodes(data="dist", default=math.inf))
    # visited: {'a': False, 'b': False, 'c': False, 'd': False, 'e': False, 'f': False}
    visited = dict(graph.nodes(data="visited", default=False))
    prev = dict(graph.nodes(data="prev", default=None))
    
    dist[src] = 0
    # using heapq for our priority queue
    heapq.heappush(Q, (dist[src], src))
    
    while Q:
        v = heapq.heappop(Q)[1]
        
        for node in list(graph.neighbors(v)):
            if not visited[node]:
                alt = dist[v] + graph[v][node]["weight"]
                
                if alt < dist[node]:
                    dist[node] = alt
                    prev[node] = v
                    
                heapq.heappush(Q, (dist[node], node))
            
        visited[v] = True
    
    return dist, prev

def dijkstra_helper(graph, src, goal):
    dist, prev = dijkstra(graph, src)
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = prev[current]
    return path[::-1], dist[goal]

# r - vertex; t - target
def dfs(graph, v, t, visited, path=[], dist = 0):
    if v == t:
        path.append(v)
        return path, dist
    
    visited[v] = True    
    path.append(v)
    
    for node in list(graph.neighbors(v)):
        if not visited[node]: 
            finalPath = dfs(graph, node, t, visited, path, dist+graph[v][node]["weight"])
            if finalPath is not None:
                return finalPath

# h value for a given node
def manhattan_distance(start_node, end_node):
    return abs(pos[start_node][0] - pos[end_node][0]) + abs(pos[start_node][1] - pos[end_node][1])

def aStar(graph, start_node, end_node): 
    #open_list - nodes we can visit/queue node; closed_lsit - nodes we have visited
    open_list = []
    closed_list = []
    #dictionaries
    f_val = dict(graph.nodes(data="val", default=math.inf))
    g_val = {}
    prev = dict(graph.nodes(data="prev", default=None))
    
    f_val[start_node] = 0
    g_val[start_node] = 0
    #pushing the start_node to open_list w/ a f value of 0
    heapq.heappush(open_list, (f_val[start_node], start_node))
    
    while len(open_list) > 0:
        #pop current node from open_list
        current_node = heapq.heappop(open_list)[1]
        #append current node to closed_list
        closed_list.append(current_node)
        
        #check if current node is the goal node. if it is, return path and distance travelled
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current)
                current = prev[current]
            return path[::-1], g_val[end_node]
        
        #otherwise iterate through current node's neighbors
        for node in graph.neighbors(current_node):
            #as long as node hasn't been visited/not in closed_list
            if node not in closed_list:
                g_val[node] = g_val[current_node] + graph[current_node][node]["weight"]
                h = manhattan_distance(node, end_node)
                f = g_val[node] + h
                #calculate f values for each neighbor and find the minimum
                if f < f_val[node]:
                    f_val[node] = f
                    prev[node] = current_node
                
                #push the minimum f node to the open_list; loop continues until goal is reached
                heapq.heappush(open_list, (f_val[node], node))
                
def uniform_cost_search(graph, src, target):
    Queue = []
    dist = dict(graph.nodes(data="dist", default=math.inf))
    visited = dict(graph.nodes(data="visited", default=False))
    prev = dict(graph.nodes(data="prev", default=None))

    dist[src] = 0
    heapq.heappush(Queue, (dist[src], src))

    while Queue:
        v = heapq.heappop(Queue)[1]

        if v == target:
            break

        for node in list(G.neighbors(v)):
            if not visited[node]:
                alt = dist[v] + graph[v][node]["weight"]

                if alt < dist[node]:
                    dist[node] = alt
                    prev[node] = v

                heapq.heappush(Queue, (dist[node], node))

        visited[v] = True

    return dist, prev

def get_shortest_path(prev, src, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = prev[current]
    return path[::-1]

def iterativeDeepeningSearch(graph, start_node, end_node):
    for depth_limit in range(len(graph)):
        result = depthLimitedSearch(graph, start_node, end_node, depth_limit)
        if result is not None:
            return result
    return None

def depthLimitedSearch(graph, node, end_node, depth_limit):
    if node == end_node:
        return [node]
    if depth_limit <= 0:
        return None
    for neighbor in graph.neighbors(node):
        result = depthLimitedSearch(graph, neighbor, end_node, depth_limit - 1)
        if result is not None:
            return [node] + result
    return None

start_time = time.time()
print()

dijkstraResult = dijkstra_helper(G, "a", "g")
print("Dijkstra's: ", dijkstraResult)
dfsResult = dfs(G, "a", "g", visited = dict(G.nodes(data="visited", default=False)))
print("DFS: ", dfsResult)
aStarResult = (aStar(G, "a", "g"))
print("A*: ", aStarResult)
uniformCostResult = uniform_cost_search(G, "a", "g")
ucShortestPath = get_shortest_path(uniformCostResult[1], "a", "g")
print("Uniform Cost: " + str(ucShortestPath))
idsResult = iterativeDeepeningSearch(G,"a","g")
print("IDS: " + str(idsResult))

print("--- %s seconds ---" % (time.time() - start_time))
print()
# plt.show()


path = dijkstraResult[0]
path_edges = list(zip(path,path[1:]))
nx.draw_networkx_nodes(G,pos,nodelist=path,node_size=700,node_color='g')
nx.draw_networkx_edges(G,pos,edgelist=path_edges,edge_color='g',width=6)
plt.show()


# Travelling Salesman Algorithm Graph

import random

R = nx.Graph()
nodes = ["a", "b", "c", "d", "e", "f"]
R.add_nodes_from(nodes)

for n in nodes:
    for m in nodes:
        if n != m:
            R.add_edge(n, m, weight=random.randint(1, 11))


elarge = [(u, v) for (u, v, d) in R.edges(data=True) if d["weight"] > 0.5]
esmall = [(u, v) for (u, v, d) in R.edges(data=True) if d["weight"] <= 0.5]

pos = nx.spring_layout(R, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(R, pos, node_size=700)

# edges
nx.draw_networkx_edges(R, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    R, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(R, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(R, "weight")
nx.draw_networkx_edge_labels(R, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()

