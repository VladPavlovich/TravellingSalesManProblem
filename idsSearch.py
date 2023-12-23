import matplotlib.pyplot as plt
import networkx as nx

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


result = iterativeDeepeningSearch(G,"a","g")
print(result)
plt.show()