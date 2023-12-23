import matplotlib.pyplot as plt
import networkx as nx
import heapq
import math

G = nx.Graph()

G.add_edge("a", "b", weight=6)
G.add_edge("a", "c", weight=2)
G.add_edge("c", "d", weight=1)
G.add_edge("c", "e", weight=7)
G.add_edge("c", "f", weight=9)
G.add_edge("a", "d", weight=3)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

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
    prev = dict(graph.nodes(data="prev", default="start"))
    
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

# r - vertex; t - target
def dfs(graph, v, t, visited, path=""):
    if v == t:
        print(v)
        return
    
    visited[v] = True
    
    print(v, end=' ')
    
    for node in list(graph.neighbors(v)):
        if not visited[node]: 
            dfs(graph, node, t, visited)

result1 = dijkstra(G, "a")
print("Distances: " + str(result1[0]) + "; Parent Nodes: " + str(result1[1]))
result2 = dfs(G, "a", "f", visited = dict(G.nodes(data="visited", default=False)))
plt.show() 

