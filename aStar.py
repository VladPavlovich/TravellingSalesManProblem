import matplotlib.pyplot as plt
import networkx as nx

#initial graph w/ no nodes or edges
R = nx.Graph()

#adding nodes w/ weights
#weights for nodes represent hueristic values stores in a dict
nodes = ["Bloomington", "Muncie", "West Lafayette", "Indianapolis", "Evansville", "Fort Wayne"]
h_dict = {"Bloomington" : 10, "Muncie" : 9, "West Lafayette" : 6, "Indianapolis" : 20, "Evansville" : 12, "Fort Wayne" : 4}
R.add_nodes_from(nodes)

#adding edges w/ weights
#weights represent edge costs
R.add_edge("Evansville", "Bloomington", weight = 112)
#R.add_edge("Evansville", "West Lafayette", weight=233)
#R.add_edge("Evansville", "Indianapolis", weight = 176)
#R.add_edge("Evansville", "Muncie", weight =)
#R.add_edge("Evansville", "Fort Wayne", weight =)
R.add_edge("Bloomington", "Indianapolis", weight = 62)
#R.add_edge("Bloomington", "West Lafayette", weight = 130)
R.add_edge("Indianapolis", "Muncie", weight = 65)
R.add_edge("Muncie", "Fort Wayne", weight = 93)
R.add_edge("Indianapolis", "West Lafayette", weight = 67)

#node positions for consistency
pos = nx.spring_layout(R, seed=2)

#drawing nodes w/ labels
nx.draw_networkx_nodes(R, pos, node_size=100)
nx.draw_networkx_labels(R, pos, font_size=10, font_family="sans-serif")

#drawing edges w/ weights
nx.draw_networkx_edges(R, pos)
g_dict = nx.get_edge_attributes(R, "weight")
nx.draw_networkx_edge_labels(R, pos, g_dict)

#formatting
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")
plt.show()

#function calculating f = h + g where h represents the heuristic value of a node, and g represents the edge cost of moving to that node
def calc_f(start_node, end_node):

    edge = (start_node, end_node)
    if edge not in g_dict:
        edge = (end_node, start_node)

    f = h_dict[start_node] + g_dict[edge]
    return f

##### ADD EXPLORED NODES LIST
def aStar(graph, start_node, end_node):
   
    path = []
    path.append(start_node)

    explored_nodes = []
    explored_nodes.append(start_node)
   
    queue_node = start_node
   
    enter_loop = True
    while enter_loop:
        min_node = ""
        min_f = 10000
        
        for node in graph.neighbors(queue_node):
            if node not in explored_nodes:
                if (calc_f(queue_node, node) < min_f):
                    min_node = node
                    min_f = calc_f(queue_node, node)
        queue_node = min_node
        path.append(queue_node)
        explored_nodes.append(queue_node)

        if (queue_node == end_node):
            enter_loop = False

    return path

print(aStar(R, "Bloomington", "Evansville"))
print(aStar(R, "Evansville", "Fort Wayne"))